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
from Load_dataset import RandomGenerator, ValGenerator, ImageToImage2D, Poly_dataset
from nets.STPNet import STPNet #STPNet_resnetimgemb
from torch.utils.data import DataLoader
import logging
import Config as config
from torchvision import transforms
from utils import  read_text, read_attr
from thop import profile
import cv2
from sklearn.metrics import roc_auc_score, jaccard_score
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

def show_image_with_dice(predict_save, labs, save_path):
    tmp_lbl = labs.to(dtype=torch.float32).detach().cpu().numpy()
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1), tmp_3dunet.reshape(-1))
    if config.task_name == "MoNuSeg":
        predict_save = cv2.pyrUp(predict_save, (448, 448))
        predict_save = cv2.resize(predict_save, (2000, 2000))
        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #定义一个核
        # predict_save = cv2.filter2D(predict_save, -1, kernel=kernel)
        cv2.imwrite(save_path, predict_save * 255)
    else:
        cv2.imwrite(save_path, predict_save * 255)
    return dice_pred, iou_pred



##################################################################################
# =================================================================================
#          Main Loop: load model,
# =================================================================================
##################################################################################
def main_loop(batch_size=config.batch_size, model_type='', tensorboard=True):
    # Load train and val data
    train_tf = transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    if config.task_name == 'Covid19':
        text = read_attr(config.task_dataset + 'Train_val_text.xlsx')
        test_text = read_attr(config.test_dataset + 'Test_text.xlsx')
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, text, train_tf,
                                       image_size=config.img_size,is_train=True)
        val_dataset = ImageToImage2D(config.val_dataset, config.task_name, text, val_tf, image_size=config.img_size)
        test_dataset = ImageToImage2D(config.test_dataset, config.task_name, test_text, val_tf, image_size=config.img_size)
        test_num = 2113
    elif config.task_name == 'COVID19_CT_new':
        train_text = read_attr(config.train_dataset + 'Train_text_sentence.xlsx')
        val_text = read_attr(config.val_dataset + 'Val_text_sentence.xlsx')
        test_text = read_attr(config.test_dataset + 'Test_text_sentence.xlsx')
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_text, train_tf,
                                       image_size=config.img_size)
        val_dataset = ImageToImage2D(config.val_dataset, config.task_name, val_text, val_tf, image_size=config.img_size)
        test_dataset = ImageToImage2D(config.test_dataset, config.task_name, test_text, val_tf, image_size=config.img_size)
        test_num = 273
    elif config.task_name == 'Kvasir-SEG':
        train_text = read_attr(config.train_dataset + 'Train_text_sentence.xlsx')
        val_text = read_attr(config.val_dataset + 'Val_text_sentence.xlsx')
        test_text = read_attr(config.test_dataset + 'Test_text_sentence.xlsx')
        val_dataset = Poly_dataset(config.val_dataset, config.task_name, val_text, val_tf,
                                       image_size=config.img_size)
        test_dataset = val_dataset
        test_num = 120
    
    test_loader = DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=True,
                            worker_init_fn=worker_init_fn,
                            num_workers=8,
                            pin_memory=True)
    
    
    lr = config.learning_rate
    logger.info(model_type)


    config_vit = config.get_CTranS_config()
    logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
    logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
    logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
    model = STPNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels, img_size=config.img_size)
    pretrained_UNet_model_path = "./Test_session_02.18_04h13/models/best_model-STPNet.pth.tar"
    pretrained_UNet = torch.load(pretrained_UNet_model_path, map_location='cuda')
    pretrained_UNet = pretrained_UNet['state_dict']
    model2_dict = model.state_dict()
    state_dict = {k: v for k, v in pretrained_UNet.items() if k in model2_dict.keys()}
    model2_dict.update(state_dict)
    model.load_state_dict(model2_dict)
    logger.info('Load successful!')

    text = torch.tensor([[101, 17758, 21908,  8985,  1010,  2048, 10372,  2752,  1010,   103,
         2187, 11192,  1998,  2157, 11192,  1012,   102,     0,     0,     0,
            0],[101, 17758, 21908,  8985,  1010,  2048, 10372,  2752,  1010,   103,
         2187, 11192,  1998,  2157, 11192,  1012,   102,     0,     0,     0,
            0]])
    length = torch.tensor([17.0, 17.0])
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
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
    
    dice_pred = 0
    iou_pred = 0
    # evaluate on validation set
    logger.info('Validation')
    with torch.no_grad():
        model.eval()
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            images, masks= sampled_batch['image'].float(), sampled_batch['label']
            images, masks= images.cuda(), masks.cuda()
            preds, img_emb, Unilateral_img_emb,num_img_emb,left_loc_img_emb,right_loc_img_emb,Unilateral_emb_norm,num_emb_norm,left_loc_emb_norm,right_loc_emb_norm= model(images)
            pred_class = torch.where(preds > 0.5, torch.ones_like(preds), torch.zeros_like(preds))
            predict_save = pred_class[0].cpu().data.numpy()
            predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
            vis_save_path = './predict/'
            dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, masks,
                                                    save_path=vis_save_path + '_predict' + names[0])
            dice_pred += dice_pred_tmp
            iou_pred += iou_tmp
            torch.cuda.empty_cache()
    print("dice_pred", dice_pred / test_num)
    print("iou_pred", iou_pred / test_num)
    return dice_pred_tmp, iou_tmp


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
