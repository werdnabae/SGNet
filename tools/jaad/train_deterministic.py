import os
import os.path as osp
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F

import lib.utils as utl
from configs.jaad import parse_sgd_args as parse_args
from lib.models import build_model
from lib.losses import rmse_loss
from lib.utils.jaadpie_train_utils import train, val, test


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
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5,
                                                        min_lr=1e-10, verbose=1)
    if osp.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        args.start_epoch += checkpoint['epoch']

    criterion = rmse_loss().to(device)

    train_gen = utl.build_data_loader(args, 'train')
    val_gen = utl.build_data_loader(args, 'val')
    print("Number of validation samples:", val_gen.__len__())
    print("Number of test samples:", test_gen.__len__())

    for epoch in range(args.start_epoch, args.epochs + args.start_epoch):
        print("Number of training samples:", len(train_gen))

        # train
        train_goal_loss, train_dec_loss, total_train_loss = train(model, train_gen, criterion, optimizer, device)

        print('Train Epoch: {} \t Goal loss: {:.4f}\t Decoder loss: {:.4f}\t Total: {:.4f}'.format(
            epoch, train_goal_loss, train_dec_loss, total_train_loss))

        # val
        val_loss = val(model, val_gen, criterion, device)


if __name__ == '__main__':
    main(parse_args())