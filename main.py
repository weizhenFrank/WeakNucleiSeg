import random

import numpy as np
import torch

from network.data_process.process_pipeline import main as prepare_data
from network.lib.bio_seg import BioSeg
from network.lib.configs.parse_arg import opt

if __name__ == '__main__':
    if opt.train['random_seed'] >= 0:
        # logger.info("=> Using random seed {:d}".format(opt.train['random_seed']))
        torch.manual_seed(opt.train.random_seed)
        torch.cuda.manual_seed(opt.train.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(opt.train.random_seed)
        random.seed(opt.train.random_seed)
    else:
        torch.backends.cudnn.benchmark = True

    if opt.prepare_data:
        prepare_data(opt)

    if opt.isTrain:
        bio_seg = BioSeg(opt)
        bio_seg.train_phase()
    else:
        bio_seg = BioSeg(opt)
        bio_seg.test_phase('test')

    print("All finished")
