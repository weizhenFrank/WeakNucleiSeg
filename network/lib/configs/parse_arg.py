import argparse
import os
import yaml
from easydict import EasyDict as edict
from network.lib.configs.config import config


def parse_args(description=''):
    parser = argparse.ArgumentParser(description=description)
    # general
    parser.add_argument('--cfg', default='',
                        help='experiment configure file name', type=str)
    parser.add_argument('--id', default='exp', type=str, help='Experiment ID')
    parser.add_argument('--gpu', default='0', type=str, help='GPU')
    parser.add_argument('--partial', default=1, type=float, help='Ratio of sampling for number of points')

    args = parser.parse_args()

    return args


# update configure from yaml file
def update_config(arguments, configure):
    # config file
    if os.path.isfile(arguments.cfg):
        with open(arguments.cfg) as f:
            exp_config = edict(yaml.safe_load(f))
            for k, v in exp_config.items():
                if k in configure:
                    if isinstance(v, dict):
                        for vk, vv in v.items():
                            configure[k][vk] = vv
                    else:
                        configure[k] = v
                else:
                    configure[k] = v
    # arguments as command line arguments has higher priority
    for k, v in arguments.__dict__.items():
        configure[k] = v
    return configure


def config_complete(arguments, configure):
    configure = update_config(arguments, configure)

    configure.id = f'{arguments.id}'

    exp_dir = f"{configure.dataname}/{configure.id}"

    configure.data.train_data_dir = f'./data_for_train/{configure.dataname}/{configure.partial:.2f}'
    configure.data.img_dir = f'{configure.data.train_data_dir}/images'

    configure.data.label_dir = f'./data/{configure.dataname}/{configure.partial:.2f}/labels_instance'
    configure.data.mask_dir = f'./data/{configure.dataname}/{configure.partial:.2f}/labels_prob'

    configure.data.label_point_dir = f'./data/{configure.dataname}/{configure.partial:.2f}/labels_point'

    configure.save_checkpoint_dir = f"{configure.save_checkpoint_dir}/{exp_dir}"
    configure.log_dir = f"{configure.log_dir}/{exp_dir}"

    configure.output_dir = f"{configure.output_dir}/{exp_dir}"

    if configure.isTrain:
        configure.tb_dir = f"{configure.tb_dir}/{exp_dir}"
    configure.intermediate_dir = f"{configure.intermediate_dir}/{exp_dir}"

    configure.gpus = str(arguments.gpu)
    return configure

args = parse_args()
opt = config_complete(args, config)
