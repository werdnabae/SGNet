import sys
import os
import os.path as osp
import numpy as np
import time
import random
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data

import lib.utils as utl
from configs.pie import parse_sgnet_args as parse_args
from lib.models import build_model
from lib.losses import rmse_loss
from lib.utils.jaadpie_train_utils import train, val, test
import pickle as pkl


def main(args):
    this_dir = osp.dirname(__file__)
    model_name = args.model
    save_dir = osp.join(this_dir, 'checkpoints', model_name, str(args.seed))
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utl.set_seed(int(args.seed))

    model = build_model(args)

    if osp.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = nn.DataParallel(model)
    model = model.to(device)
    criterion = rmse_loss().to(device)
    # args.batch_size = 1  # Added line, so we can see the true SD
    test_gen = utl.build_data_loader(args, 'test')
    print("Number of test samples:", test_gen.__len__())

    # test
    # #test_loss, MSE_15, MSE_05, MSE_10, FMSE, FIOU, CMSE, CFMSE, SD, MSE_dict = test(model, test_gen, criterion, device)
    # print(test_loss)
    # print("MSE_05: %4f;  MSE_10: %4f;  MSE_15: %4f;   FMSE: %4f;   FIOU: %4f\n" % (MSE_05, MSE_10, MSE_15, FMSE, FIOU))
    # print("CFMSE: %4f;   CMSE: %4f;  \n" % (CFMSE, CMSE))
    MEAN, SD, MSE = test(model, test_gen, criterion, device)
    # print(f'MEAM: {MEAN}')
    # print(f'SD: {SD}')

    # print(f'SD: {SD}')

    if (args.dataset == 'JAAD'):
        out_dir = '/home/anbae/Documents/Research/SGNet.pytorch/outputs/JAAD/Gender'
    else:
        out_dir = '/home/anbae/Documents/Research/SGNet.pytorch/outputs/PIE/Gender'
    outDic = {'MEAN': MEAN, 'SD': SD, 'MSE': MSE}
    output_file = os.path.join(out_dir, f'SGNet_{args.dataset}_{args.split}__{args.gender}_{args.age}.pkl')
    print('Writing outputs to:', output_file)
    pkl.dump(outDic, open(output_file, 'wb'))
    print('done')

if __name__ == '__main__':
    main(parse_args())
