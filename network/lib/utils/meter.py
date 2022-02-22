import logging
import os

from torch.utils.tensorboard import SummaryWriter


def setup_logging(option):
    mode = 'a' if option.train['checkpoint'] else 'w'

    # create logger for training information
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.DEBUG)
    # create console handler and file handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    if not os.path.exists('{:s}/'.format(option.log_dir)):
        os.makedirs('{:s}/'.format(option.log_dir))

    file_handler = logging.FileHandler('{:s}/train_log.txt'.format(option.log_dir), mode=mode)
    file_handler.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s\t%(message)s', datefmt='%m-%d %I:%M')
    # add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # create logger for epoch results
    logger_results = logging.getLogger('results')
    logger_results.setLevel(logging.DEBUG)

    file_handler2 = logging.FileHandler('{:s}/epoch_results.txt'.format(option.log_dir), mode=mode)

    file_handler2.setFormatter(logging.Formatter('%(message)s'))
    logger_results.addHandler(file_handler2)

    logger.info('***** Training starts *****')
    logger.info('save directory: {:s}'.format(option.log_dir))
    if mode == 'w':
        logger_results.info('epoch\ttrain_loss\ttrain_loss_vor\ttrain_loss_cluster\tval_acc\tval_AJI')

    return logger, logger_results


class LossMeter(object):
    def __init__(self, loss_func):
        self.running_loss = []
        self.count = 0  # count = number of all received values, including those cleared.
        self.loss_func = loss_func

    def update(self, pred, gt, get_result=False, fg_weight=None):
        self.count += 1
        if fg_weight is not None:
            loss = self.loss_func(pred, gt, weight=[1, fg_weight])
        else:
            loss = self.loss_func(pred, gt)
        self.running_loss.append(loss.detach())

        if get_result:
            return loss

    def get_metric(self):
        avg = 0
        for p in self.running_loss:
            avg += p
        loss_avg = avg*1.0 / len(self.running_loss) if len(self.running_loss)!=0 else None
        return loss_avg

    def reset(self):
        self.running_loss = []


class MultiLossMeter(object):
    def __init__(self):
        self.running_loss = {}
        self.loss_names = None
        self.count = 0

    def reset(self):
        self.count = 0
        self.running_loss = {}
        if self.loss_names is not None:
            for term in self.loss_names:
                self.running_loss[term] = 0.0

    def update(self, losses, loss_names):
        if self.loss_names is None:
            self.loss_names = loss_names
            self.reset()
        self.count += 1
        loss_terms = dict(zip(loss_names, losses))

        # update running loss
        for term in self.running_loss.keys():
            if term in loss_terms.keys():
                self.running_loss[term] += loss_terms[term].detach()

    def get_metric(self):
        keys = self.running_loss.keys()
        avg_terms = {}
        for key in keys:
            avg_terms[key] = 0.0
        for key in keys:
            avg_terms[key] = self.running_loss[key] * 1.0 / self.count

        return avg_terms


class TFBoardWriter:
    def __init__(self, log_dir, type=None):
        if type is not None:
            tfbd_dir = os.path.join(log_dir, 'tfboard', type)
        else:
            tfbd_dir = os.path.join(log_dir, 'tfboard')

        if not os.path.exists(tfbd_dir):
            os.makedirs(tfbd_dir)

        self.tf_writer = SummaryWriter(log_dir=tfbd_dir,
                                       flush_secs=20)
        self.enable = True

    def write_data(self, iter, meter, key=None):
        if isinstance(iter, str):
            model = meter[0]
            input = meter[1]
            self.tf_writer.add_graph(model, input)
        else:
            if key is None:
                for each in meter.keys():
                    val = meter[each]
                    self.tf_writer.add_scalar(each, val, iter)
            else:
                self.tf_writer.add_scalar(key, meter, iter)

    def add_pr_curve_raw(self, iter, true_positive_counts, false_positive_counts, true_negative_counts, false_negative_counts, precision, recall, key):
        self.tf_writer.add_pr_curve_raw(key, true_positive_counts, false_positive_counts, true_negative_counts, false_negative_counts, precision, recall, iter)

    def add_pr_curve(self, iter, predictions, labels, key):

        self.tf_writer.add_pr_curve(key, labels, predictions, global_step=iter, num_thresholds=127, weights=None, walltime=None)

    def add_histogram(self, iter, predictions, key):
        self.tf_writer.add_histogram(key, predictions, global_step=iter, bins='tensorflow', walltime=None, max_bins=None)

    def close(self):
        if self.enable:
            self.tf_writer.close()


def l1_loss(pred, gt):
    return abs(pred - gt)

