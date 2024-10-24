import os.path
os.environ["CUDA_VISIBLE_DEVICES"] = "2,0,3,5,1"
import sys
import logging
import copy
import time
import torch
from utils import factory
from utils.toolkit import count_parameters

def train(args):
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])
    for seed in seed_list:
        args['seed'] = seed
        args['device'] = device
        _train(args)

def _train(args):
    logfilename = './logs/{}_{}_{}_'.format(args['prefix'], args['model_name'], args['net_type']) + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    _set_random()
    _set_device(args)
    print_args(args)
    model = factory.get_model(args['model_name'], args)
    model.train_phase()
    logging.info('All params: {}'.format(count_parameters(model._network)))
    logging.info('Trainable params: {}'.format(count_parameters(model._network, True)))
    ckp_name = logfilename + '.pkl'
    torch.save(model, ckp_name)

def _set_device(args):
    device_type = args['device']
    gpus = []
    for device in device_type:
        if device_type == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))
        gpus.append(device)
    args['device'] = gpus

def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))
