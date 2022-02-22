from easydict import EasyDict as edict

# at most two sub level
config = edict()

# 1. data_dir
config.prepare_data = False
config.isTrain = True
config.snapshot = 5
config.use_instance_seg = False

config.dataname = ''
config.data_dir = './data'
config.save_checkpoint_dir = './checkpoints'
config.log_dir = './log'
config.output_dir = './output'
config.intermediate_dir = './inter'
config.tb_dir = './tb'

# 2. data related
config.data = edict()
config.data.num_workers = 0

# 3. model related
config.model = edict()
config.model.network = 'ResUNet34'
config.model.unet = {"add_coord": True, "filters": [512, 256, 128, 64, 64, 64, 16]}

# 4. training params
config.train = edict()
config.train.random_seed = 1

config.train.n_epochs = 60
config.train.n_batches = 8
config.train.select = 'aji'
config.train.metrics = ['acc', 'p_iou', 'p_F1', 'aji', 'dice']
config.train.label_type = ['voronoi', 'cluster']  # ['binary', 'voronoi', 'cluster', 'prob', 'pseudo']
config.train.label_type_channel = [3, 3]

config.train.loss_weight = [1.0, 1.0]  # background and foreground weight
config.train.loss_type_weight = [1.0, 1.0]  # [cluster, voronoi, pseudo]

config.train.log_interval = 30
config.train.lr = 1e-4
config.train.momentum = 0.9
config.train.weight_decay = 1e-4
config.train.optimizer = "Adam"  # SGD
config.train.scheduler = "MultiStep"
config.train.milestone = [99999999, 999999999]  # if above is MultiStep
config.train.cutoff = 0.5
config.train.add_bg = False
config.train.bg_cutoff = 0.01
config.train.bg_weight = 1

config.train.start_epoch = 0
config.train.checkpoint = None

# 5. test params
config.test = edict()
config.test.test_epoch = 'best'
config.test.save_flag = True
config.test.patch_size = 224
config.test.overlap = 80
config.test.model_path = None
config.test.cluster = False
config.test.seg_model_path = ''

config.test.rawout = False
config.test.only_patch = False

# 6. post processing
config.post = dict()
config.post.min_area = 20  # minimum area for an object
config.post.cutoff = 0.5  # for two-stage, the cutoff for get the foreground from semantic probmap

config.post.cluster_method = "Kmeans"  # Kmeans mean_shift
config.post.bandwidth = 1.5  # if mean_shift, this works
config.post.ins_post = False

config.post.whole_patch = False
config.post.cluster_all = True

# 7. define data transforms for training
config.transform_train = edict()
config.transform_train.random_resize = [0.8, 1.25]
config.transform_train.horizontal_flip = True
config.transform_train.random_affine = 0.3  # Bound should be in range [0, 0.5)
config.transform_train.random_rotation = 90
config.transform_train.random_crop = 224
config.transform_train.label_encoding = 2  # indicate last 3 is labels images for LabelEncoding 3 channel to 1
config.transform_train.to_tensor = 1  # the index of first label image for to_tensor to distinguish the label and img
config.transform_train.normalize = [[0.7477097, 0.54574299, 0.67144866],
                                    [0.1580947, 0.19779244, 0.15096586]]

config.transform_val = edict()
config.transform_val.to_tensor = 1
config.transform_val.normalize = [[0.7477097, 0.54574299, 0.67144866],
                                  [0.1580947, 0.19779244, 0.15096586]]
