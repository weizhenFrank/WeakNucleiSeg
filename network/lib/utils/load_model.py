from collections import OrderedDict

import torch


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def load_param_from_file(model, f: str, partially=False, module_namelist=None, logger=None):
    # logger = logging.getLogger(logger_name)
    if partially:
        logger.info("partially load weight from %s" % f) if logger is not None else print("partially load weight from %s" % f)

        model = load_weight_partially_from_file(model, f, module_namelist, logger)
    # else:
    #     logger.info("load weight from %s" % f)
    #     state_dict = torch.load(f, map_location=torch.device("cpu"))
    #     state_dict = state_dict['state_dict']
    #     model = load_state_dict(model, state_dict)

    return model


def load_weight_partially_from_file(model, f: str, module_namelist, logger=None):
    # logger = logging.getLogger(logger_name)
    # state_dict = torch.load(f, map_location=torch.device("cpu"))['state_dict']
    state_dict = torch.load(f)['state_dict']
    # ipdb.set_trace()
    own_state = model.state_dict()
    if module_namelist is not None:

        for to_load_name in module_namelist:
            param = state_dict[to_load_name]
            if to_load_name.startswith('module'):
                # ipdb.set_trace()
                to_load_name = to_load_name.replace('module.', '')
                # ipdb.set_trace()
            try:
                if isinstance(param, torch.nn.Parameter):
                    param = param.data
                    # backwards compatibility for serialized parameters
                own_state[to_load_name].copy_(param)
                # print("[Copied]: {}".format(to_load_name))
                logger.info("[Copied]: {}".format(to_load_name)) if logger is not None else print("[Copied]: {}".format(to_load_name))

            except RuntimeError:
                logger.info('[Missed] Size Mismatch... : {}'.format(to_load_name)) if logger is not None else print('[Missed] Size Mismatch... : {}'.format(to_load_name))

    else:

        for name, param in state_dict.items():
            if name.startswith('module'):
                name = name.replace('module.', '')
            try:

                if name not in own_state:
                    logger.info('[Missed]: {}'.format(name)) if logger is not None else print('[Missed]: {}'.format(name))

                    continue
                if isinstance(param, torch.nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                own_state[name].copy_(param)
                # ipdb.set_trace()
                # print("[Copied]: {}".format(name))
                logger.info("[Copied]: {}".format(name)) if logger is not None else print("[Copied]: {}".format(name))
            except RuntimeError:
                logger.info('[Missed] Size Mismatch... : {}'.format(name)) if logger is not None else print('[Missed] Size Mismatch... : {}'.format(name))
        logger.info("load the pretrain model") if logger is not None else print("load the pretrain model")

    return model


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

