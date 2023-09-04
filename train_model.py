# -*- coding: utf-8 -*-
import torch.optim
import torch.nn as nn
import time
import torch.utils.data
from tensorboardX import SummaryWriter
import os
import numpy as np
import random
from torch.backends import cudnn
import Config
from Load_Dataset import RandomGenerator, ValGenerator, ImageToImage2D, LV2D, ECA, Poly_dataset
from nets.LViT import LViT, LViTW, STPNet
from torch.utils.data import DataLoader
import logging
from Train_one_epoch import train_one_epoch, print_summary
import Config as config
from torchvision import transforms
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE, WeightedDiceCE, read_text, read_text_LV, save_on_batch, read_attr
from thop import profile

def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr


def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if best_model:
        filename = save_path + '/' + \
                   'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '/' + \
                   'model-{}-{:02d}.pth.tar'.format(model, epoch)
    torch.save(state, filename)


def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)


##################################################################################
# =================================================================================
#          Main Loop: load model,
# =================================================================================
##################################################################################
def main_loop(batch_size=config.batch_size, model_type='', tensorboard=True):
    # Load train and val data
    train_tf = transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    if config.task_name == 'MoNuSeg':
        train_text = read_text(config.train_dataset + 'Train_text.xlsx')
        val_text = read_text(config.val_dataset + 'Val_text.xlsx')
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_text, train_tf,
                                       image_size=config.img_size)
        val_dataset = ImageToImage2D(config.val_dataset, config.task_name, val_text, val_tf, image_size=config.img_size)
    elif config.task_name == 'Covid19':
        text = read_attr(config.task_dataset + 'Train_Val_text.xlsx')
        test_text = read_attr(config.test_dataset + 'Test_text.xlsx')
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, text, train_tf,
                                       image_size=config.img_size)
        val_dataset = ImageToImage2D(config.val_dataset, config.task_name, text, val_tf, image_size=config.img_size)
        test_dataset = ImageToImage2D(config.test_dataset, config.task_name, test_text, val_tf, image_size=config.img_size)
    elif config.task_name == 'COVID19_CT_new':
        train_text = read_attr(config.train_dataset + 'Train_text.xlsx')
        val_text = read_attr(config.val_dataset + 'Val_text.xlsx')
        test_text = read_attr(config.test_dataset + 'Test_text.xlsx')
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_text, train_tf,
                                       image_size=config.img_size)
        val_dataset = ImageToImage2D(config.val_dataset, config.task_name, val_text, val_tf, image_size=config.img_size)
        test_dataset = ImageToImage2D(config.test_dataset, config.task_name, test_text, val_tf, image_size=config.img_size)
    elif config.task_name == 'ECA':
        Dataset = ECA(config.ECA_dataset)
        torch.manual_seed(666)
        # train_dataset, test_dataset = torch.utils.data.random_split(Dataset, [60, 14])
        # train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [30, 30])
        train_dataset, test_dataset = torch.utils.data.random_split(Dataset, [228, 58])
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [182, 46])
    elif config.task_name == 'Kvasir-sessile':
        # Dataset = Kvasir_sessile_dataset(config.Kvasir_sessile_dataset, config.task_name,train_tf,
        #                                image_size=config.img_size)
        train_dataset = Poly_dataset(config.train_dataset, config.task_name,train_tf,
                                       image_size=config.img_size)
        val_dataset = Poly_dataset(config.val_dataset, config.task_name,train_tf,
                                       image_size=config.img_size)
        test_dataset = Poly_dataset(config.test_dataset, config.task_name,val_tf,
                                       image_size=config.img_size)
    elif config.task_name == 'CVC_ClinicDB':
        train_dataset = Poly_dataset(config.train_dataset, config.task_name,train_tf,
                                       image_size=config.img_size)
        val_dataset = Poly_dataset(config.val_dataset, config.task_name,train_tf,
                                       image_size=config.img_size)
        test_dataset = Poly_dataset(config.test_dataset, config.task_name,val_tf,
                                       image_size=config.img_size)
    elif config.task_name == 'Kvasir-SEG':
        train_dataset = Poly_dataset(config.train_dataset, config.task_name,train_tf,
                                       image_size=config.img_size)
        val_dataset = Poly_dataset(config.val_dataset, config.task_name,val_tf,
                                       image_size=config.img_size)
        test_dataset = val_dataset
    elif config.task_name == 'BKAI':
        train_dataset = Poly_dataset(config.train_dataset, config.task_name,train_tf,
                                       image_size=config.img_size)
        val_dataset = Poly_dataset(config.val_dataset, config.task_name,val_tf,
                                       image_size=config.img_size)
        test_dataset = Poly_dataset(config.test_dataset, config.task_name,val_tf,
                                       image_size=config.img_size)

    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              worker_init_fn=worker_init_fn,
                              num_workers=8,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            worker_init_fn=worker_init_fn,
                            num_workers=8,
                            pin_memory=True)
                            
    
    test_loader = DataLoader(test_dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            worker_init_fn=worker_init_fn,
                            num_workers=8,
                            pin_memory=True)
    
    
    lr = config.learning_rate
    logger.info(model_type)

    if model_type == 'STPNet':
        config_vit = config.get_CTranS_config()
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
        model = STPNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels, img_size=config.img_size)
        # model = LViT(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)  # MoNuSeg & Covid19

    elif model_type == 'STPNet_pretrain':
        config_vit = config.get_CTranS_config()
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
        model = STPNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels, img_size=config.img_size)
        pretrained_UNet_model_path = "Covid19/STPNet/Test_session_07.23_20h26/models/best_model-STPNet.pth.tar"
        pretrained_UNet = torch.load(pretrained_UNet_model_path, map_location='cuda')
        pretrained_UNet = pretrained_UNet['state_dict']
        model2_dict = model.state_dict()
        state_dict = {k: v for k, v in pretrained_UNet.items() if k in model2_dict.keys()}
        print(state_dict.keys())
        model2_dict.update(state_dict)
        model.load_state_dict(model2_dict)
        logger.info('Load successful!')

    else:
        raise TypeError('Please enter a valid name for the model type')
    input = torch.randn(2, 3, 224, 224)
    text = torch.randn(2, 1, 224, 224)
    # flops, params = profile(model, inputs=(input,))
    flops, params = profile(model, inputs=(input, text, ))  # MoNuSeg & Covid19
    print('flops:{}'.format(flops))
    print('params:{}'.format(params))
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)  # Choose optimize
    if config.cosineLR is True:
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    else:
        lr_scheduler = None
    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info('log dir: '.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    max_dice = 0.0
    best_epoch = 1
    for epoch in range(config.epochs):  # loop over the dataset multiple times
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs + 1))
        logger.info(config.session_name)
        # train for one epoch
        model.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))
        train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, None, model_type, logger)  # sup
        train_one_epoch(val_loader, model, criterion, optimizer, writer, epoch, None, model_type, logger)  # sup


        # evaluate on validation set
        logger.info('Validation')
        with torch.no_grad():
            model.eval()
            val_loss, val_dice = train_one_epoch(test_loader, model, criterion,
                                                 optimizer, writer, epoch, lr_scheduler, model_type, logger)
        # =============================================================
        #       Save best model
        # =============================================================
        if val_dice > max_dice:
            if epoch + 1 > 5:
                logger.info(
                    '\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice, val_dice))
                max_dice = val_dice
                best_epoch = epoch + 1
                save_checkpoint({'epoch': epoch,
                                 'best_model': True,
                                 'model': model_type,
                                 'state_dict': model.state_dict(),
                                 'val_loss': val_loss,
                                 'optimizer': optimizer.state_dict()}, config.model_path)
        else:
            logger.info('\t Mean dice:{:.4f} does not increase, '
                        'the best is still: {:.4f} in epoch {}'.format(val_dice, max_dice, best_epoch))
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count, config.early_stopping_patience))

        if early_stopping_count > config.early_stopping_patience:
            logger.info('\t early_stopping!')
            break

    return model


if __name__ == '__main__':
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    logger = logger_config(log_path=config.logger_path)
    model = main_loop(model_type=config.model_name, tensorboard=True)
