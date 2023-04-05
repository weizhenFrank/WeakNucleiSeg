import json
import os
from functools import reduce

import imageio
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.utils.data import DataLoader

import network.lib.utils.utils as utils
from network.lib.datasets.dataset import DataFolder
from network.lib.models.model import *
from network.lib.utils.cal_metric import compute_metrics
from network.lib.utils.image_transform import get_transforms
from network.lib.utils.inference import instance_inference, get_probmaps
from network.lib.utils.load_model import load_param_from_file
from network.lib.utils.losses import *
from network.lib.utils.meter import TFBoardWriter, setup_logging
from network.lib.utils.utils import save_checkpoint, copydir
from tqdm import tqdm

class BioSeg(object):

    def __init__(self, option):

        super(BioSeg, self).__init__()
        self.opt = option
        self.split = self.read_split()

        self.data_transforms = {'train': get_transforms(self.opt.transform_train),
                                'val': get_transforms(self.opt.transform_val)}

        self.logger, self.logger_results = setup_logging(self.opt)
        self.logger.info(utils.save_config(self.opt, self.opt.log_dir))
        if self.opt.isTrain:
            self.tf_writer_train = TFBoardWriter(self.opt.tb_dir, type='train')
            self.tf_writer_val = TFBoardWriter(self.opt.tb_dir, type="val")
        self.label_type = self.opt.train.label_type
        self.label_type_channel = self.opt.train.label_type_channel
        self.loss_type_weight = self.opt.train.loss_type_weight
        self.data_loader = self.build_dataloader()

        self.model = self.build_model()
        self.best_model_wts = None
        self.best_epoch = -1
        self.metrics = self.opt.train.metrics
        self.selection_best_metric = self.opt.train.select
        self.best_score = 0

        self.criteria, self.optimizer, self.scheduler = self.build_optimization()
        if self.selection_best_metric not in self.metrics:
            raise Exception(f"{self.selection_best_metric} not in metrics!")

    def read_split(self):
        """
        The train, val and test split
        Returns
        -------

        """
        with open(os.path.join(self.opt.data_dir, self.opt.dataname, "train_val_test.json")) as f:
            train_val_test = json.load(f)
        return train_val_test

    def build_dataloader(self):
        """
        Build the training dataloader
        Returns
        -------

        """
        img_dir = f'{self.opt.data.img_dir}/train'

        dir_list = [img_dir, ]
        num_channels = [3, ]
        sub_label_dirs = {}
        post_fix = []

        for idx, label_type in enumerate(self.label_type):
            sub_label_dirs[label_type] = os.path.join(self.opt.data.train_data_dir, f"labels_{label_type}",
                                                      "train")

            if label_type == 'voronoi':
                label_postfix = '_vor'
            elif label_type == 'instance':
                label_postfix = ''
            else:
                label_postfix = f"_{label_type}"

            if label_type in ['prob', 'instance']:
                ext = 'tiff'
            else:
                ext = 'png'

            to_append = f"label{label_postfix}.{ext}"

            post_fix.append(to_append)
            dir_list.append(sub_label_dirs[label_type])
            num_channels.append(self.label_type_channel[idx])
            
        train_set = DataFolder(dir_list, post_fix, num_channels, self.data_transforms['train'])
        train_loader = DataLoader(train_set, batch_size=self.opt.train.n_batches, shuffle=True,
                                  num_workers=self.opt.data.num_workers)

        return train_loader

    def choose_model_arch(self):

        os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.gpus
        if self.opt.model.network == "ResUNet34":
            model = ResUNet34(unet_arch=self.opt.model.unet)
        else:
            raise Exception("Please give valid model name")

        return model

    def build_model(self):

        model = self.choose_model_arch()
        if self.opt.train.checkpoint:
            if os.path.isfile(self.opt.train.checkpoint):
                self.logger.info(f"=> loading checkpoint {self.opt.train.checkpoint}")
                checkpoint = torch.load(self.opt.train.checkpoint)

                self.opt.train.start_epoch = checkpoint['epoch']

                model = load_param_from_file(model, self.opt.train.checkpoint, partially=True, logger=self.logger)

                self.logger.info(f"=> loaded checkpoint {self.opt.train['checkpoint']} (epoch {checkpoint['epoch']})")

            else:
                self.logger.info("=> no checkpoint found at '{}'".format(self.opt.train['checkpoint']))

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
        model = model.cuda()

        self.logger.info(f"model parameters status: \n{utils.show_params_status(model)}")
        return model

    def build_optimization(self):

        criteria = dict()

        if self.opt.use_instance_seg:
            criteria['discriminative'] = DiscriminativeLoss(0.5, 1.5, 0.001)
        else:
            for label_type_name in self.label_type:
                criteria[label_type_name] = torch.nn.NLLLoss(weight=torch.tensor(self.opt.train.loss_weight),
                                                             ignore_index=2).cuda()

        if self.opt.train.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.train.lr,
                                         weight_decay=self.opt.train.weight_decay,
                                         betas=(0.9, 0.99))

        elif self.opt.train.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.opt.train.lr,
                                        momentum=self.opt.train.momentum, weight_decay=self.opt.train.weight_decay)
        else:
            raise Exception("Please Give valid Optimizer!")

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.opt.train.milestone)
        return criteria, optimizer, scheduler

    def train_epoch(self, model, logger):

        n_loss = len(self.label_type) + 1
        if self.opt.use_instance_seg:
            n_loss = 4
        results = utils.AverageMeter(n_loss)
        model = model.cuda()
        loss_name_list = []

        model.train()
        for i, sample in enumerate(self.data_loader):
            iteration = i + 1
            loss_dict = {'total': 0}

            generated_labels = sample[1:]
            generated_labels = [label.squeeze(1) for label in generated_labels]
            inputs = sample[0]
            inputs = inputs.cuda()
            output = model(inputs)

            if self.opt.use_instance_seg:

                foreground_prob = generated_labels[self.label_type.index("prob")]
                fg_cutoff = self.opt.train.cutoff
                foreground = utils.generate_hard_label(foreground_prob, (fg_cutoff, fg_cutoff),
                                                       self.opt.post['min_area'])

                if self.opt.train.add_bg:
                    bg_cutoff1, bg_cutoff2 = self.opt.train.bg_cutoff
                    bg = utils.get_bg(foreground_prob, bg_cutoff1, bg_cutoff2)
                    bg_weight = self.opt.train.bg_weight
                else:
                    bg = None
                    bg_weight = None
                if 'instance' in self.label_type:
                    instance = generated_labels[self.label_type.index("instance")].cpu().numpy()
                    dis_loss = self.criteria['discriminative'](embedding=output,
                                                               foregrounds=instance, labels=None, bg=bg, alpha=bg_weight)
                else:
                    vor = generated_labels[self.label_type.index("voronoi")].cpu().numpy()
                    dis_loss = self.criteria['discriminative'](embedding=output,
                                                               foregrounds=foreground, labels=vor, bg=bg, alpha=bg_weight)
                loss_dict['variance'] = dis_loss[0]
                loss_dict['distance'] = dis_loss[1]
                loss_dict['reg'] = dis_loss[2]

            else:
                log_prob_maps = F.log_softmax(output, dim=1)
                # generated_labels is the same order with label_type parameter
                for idx, label in enumerate(generated_labels):
                    loss_dict[self.label_type[idx]] = self.criteria[self.label_type[idx]](log_prob_maps, label.cuda())

            loss = reduce((lambda x, y: x + y), loss_dict.values())
            loss_dict['total'] = loss
            result = []
            for single_loss in loss_dict.values():
                result.append(single_loss.item())
            results.update(result, inputs.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            torch.cuda.empty_cache()

            results_avg_dict = dict()
            for idx, loss_name in enumerate(loss_dict.keys()):
                results_avg_dict[loss_name] = results.avg[idx]

            loss_str = '\t'.join(
                f"Loss_{loss_name} {loss_val:.4f}" for loss_name, loss_val in results_avg_dict.items())

            if iteration % self.opt.train.log_interval == 0 or iteration == len(self.data_loader):
                logger.info(f'\tIteration: [{iteration}/{len(self.data_loader)}]\t{loss_str}\t')

            loss_name_list = loss_dict.keys()

        results_avg_dict = dict()
        for idx, loss_name in enumerate(loss_name_list):
            results_avg_dict[loss_name] = results.avg[idx]

        return results_avg_dict

    def train_phase(self):

        selection_value = 0
        best_score = 0
        num_epochs = self.opt.train.n_epochs

        val_results = dict()
        train_results = dict()
        
        for epoch in tqdm(range(self.opt.train.start_epoch, num_epochs)):
            self.logger.info(f'Epoch: [{epoch + 1}/{num_epochs}]')

            train_loss_dict = self.train_epoch(self.model, self.logger)

            self.scheduler.step(epoch+1)

            lr_rate = self.optimizer.param_groups[0]['lr']
            self.tf_writer_train.write_data(epoch + 1, lr_rate, "lr/lr_epoch")

            for loss_name, loss_value in train_loss_dict.items():
                self.tf_writer_train.write_data(epoch + 1, loss_value, f'loss/loss_{loss_name}')

            if (epoch + 1) % self.opt.snapshot == 0:
                with torch.no_grad():
                    val_metrics = self.test_phase('val', epoch + 1)
                    val_results[epoch + 1] = val_metrics

                for value, key in zip(val_metrics, self.metrics):
                    self.tf_writer_val.write_data(epoch + 1, value, f'eval/{key}')
                    if key == self.selection_best_metric:
                        selection_value = value

            if self.opt.use_instance_seg:
                interval = self.opt.snapshot * 10
            else:
                interval = self.opt.snapshot

            if (epoch + 1) % interval == 0:
                with torch.no_grad():
                    train_metrics = self.test_phase('train', epoch + 1)
                    train_results[epoch + 1] = train_metrics

                for value, key in zip(train_metrics, self.metrics):
                    self.tf_writer_train.write_data(epoch + 1, value, f'eval/{key}')

            is_best = selection_value > best_score
            best_score = max(selection_value, best_score)
            if is_best:
                self.best_epoch = epoch + 1
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                copydir(f'{self.opt.output_dir}/{epoch + 1}', f'{self.opt.output_dir}/best')

            else:
                self.best_epoch = self.best_epoch
                self.best_model_wts = self.best_model_wts

            cp_flag = (epoch + 1) % self.opt.snapshot == 0

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, epoch + 1, self.opt.save_checkpoint_dir, is_best, cp_flag)

            logger_results_str = f'{epoch + 1}' + "\t".join(
                f"{value:.4f}" for key, value in train_loss_dict.items())

            self.logger_results.info(logger_results_str)
        if self.best_epoch != -1:
            self.logger_results.info(
                f"Best {self.selection_best_metric} on val on epoch {self.best_epoch}")

            string = "\t".join(f"{k} : {v:.4f}" for k, v in zip(self.metrics, val_results[self.best_epoch]))
            self.logger_results.info(f'val best:{string}')

            for value, key in zip(val_results[self.best_epoch], self.metrics):
                self.tf_writer_val.write_data(self.best_epoch, value, f'eval_best/{key}')

    def test_phase(self, img_set, test_epoch=None):
        if test_epoch is None:
            test_epoch = self.opt.test.test_epoch
        img_dir = os.path.join(self.opt.data_dir, self.opt.dataname, "images")
        label_dir = self.opt.data.label_dir
        save_dir = f'{self.opt.output_dir}/{test_epoch}'

        ins_seg_folder = f'{save_dir}/{img_set}_ins_seg'
        if not os.path.exists(ins_seg_folder):
            os.makedirs(ins_seg_folder)

        seg_folder = f'{save_dir}/{img_set}_segmentation'
        if not os.path.exists(seg_folder):
            os.makedirs(seg_folder)

        metric_names = self.metrics
        test_results = dict()
        all_result = utils.AverageMeter(len(metric_names))

        test_transform = get_transforms(self.opt.transform_val)
        model = self.choose_model_arch()
        model = model.cuda()
        cudnn.benchmark = True

        self.logger.info('***** Inference starts *****')
        if os.path.isfile(self.opt.test.model_path):
            model_path = str(self.opt.test.model_path)
            self.logger.info(f"=> loading trained model at {model_path}")
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['state_dict'])
            self.logger.info(f"=> loaded model at epoch {checkpoint['epoch']}")

        else:
            model = self.model

        if os.path.isfile(self.opt.test.seg_model_path):
            model_path = str(self.opt.test.seg_model_path)
            self.logger.info(f"=> loading trained model at {model_path}")
            checkpoint = torch.load(model_path)
            seg_model = ResUNet34(unet_arch={"add_coord": False, "filters": [512, 256, 128, 64, 64, 64, 2]})
            seg_model = seg_model.cuda()
            seg_model.load_state_dict(checkpoint['state_dict'])
            self.logger.info(f"=> loaded model at epoch {checkpoint['epoch']}")
            seg_model.eval()
        else:
            seg_model = None

        model.eval()
        counter = 0

        for img_name in self.split[img_set]:

            self.logger.info(f'=> Processing image {img_name}')
            img_path = f'{img_dir}/{img_name}'
            img = Image.open(img_path)
            name = os.path.splitext(img_name)[0]

            label_path = f'{label_dir}/{name}_label.png'
            gt = imageio.imread(label_path)

            input_img = test_transform((img,))[0].unsqueeze(0)
            point = utils.get_centroid(gt)

            if self.opt.use_instance_seg:
                if not os.path.isfile(self.opt.test.seg_model_path):
                    se_fg_path = os.path.join(self.opt.data.mask_dir, f"{name}_label_prob.tiff")
                    se_fg = imageio.imread(se_fg_path)
                    probmap = np.stack([1-se_fg, se_fg], axis=0)
                else:
                    se_fg = None
                    probmap = get_probmaps(input_img, seg_model, self.opt)

                inference_result = instance_inference(model, seg_model, input_img, point, self.opt, name, se_fg)

                instance_seg_colored = utils.mk_colored(inference_result)

            else:
                probmap = get_probmaps(input_img, model, self.opt)
                inference_result = utils.cca(probmap, self.opt)
                instance_seg_colored = utils.mk_colored(inference_result)

            metrics = compute_metrics(inference_result, gt, metric_names)

            test_results[name] = []
            for metric_name in metric_names:
                test_results[name].append(metrics[metric_name])

            message = f'{name} metric: \n'

            for k, v in metrics.items():
                message += f'\t{k}: {v:.4f}\n'
            self.logger.info(message)
            all_result.update(test_results[name])

            self.logger.info('\tSaving image results...')

            imageio.imwrite(f'{ins_seg_folder}/{name}_ins_seg_colored.png', instance_seg_colored)
            imageio.imwrite(f'{ins_seg_folder}/{name}_ins_seg.tiff', inference_result.astype(np.int32))
            imageio.imwrite(f'{seg_folder}/{name}_seg.png', (inference_result > 0).astype(np.uint8) * 255)
            imageio.imwrite(f'{seg_folder}/{name}_seg_prob.tiff', probmap[1, :, :].astype(np.float32))

            counter += 1

        message = f'{counter} images average metric: \n'
        for i in range(len(metric_names)):
            message += f'\t{metric_names[i]}: {all_result.avg[i]:.4f}\n'
        self.logger.info(message)

        utils.save_results(metric_names, all_result.avg, test_results, f'{save_dir}/{img_set}_results.txt')

        return all_result.avg

