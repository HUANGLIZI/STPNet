# -*- coding: utf-8 -*-
import torch.optim
import os
import time
from utils import *
import numpy as np
import Config as config
import warnings
from torchinfo import summary
# from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings("ignore")
import torch.nn.functional as F


def print_summary(epoch, i, nb_batch, names, loss, loss_name, batch_time,
                  average_loss, average_text_l1,average_text_l2,average_text_l3,average_text_l4,average_time, iou, average_iou,
                  dice, average_dice, acc, average_acc, mode, lr, logger):
    '''
        mode = Train or Test
    '''
    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(
        epoch, i, nb_batch)
    string = ''
    # string += str(names)
    string += ' Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    string += '(Unilateral_Loss {:.4f}) '.format(average_text_l1)
    string += '(num_Loss {:.4f}) '.format(average_text_l2)
    string += '(left_loc_Loss {:.4f}) '.format(average_text_l3)
    string += '(right_loc_Loss {:.4f}) '.format(average_text_l4)
    string += 'IoU:{:.3f} '.format(iou)
    string += '(Avg {:.4f}) '.format(average_iou)
    string += 'Dice:{:.4f} '.format(dice)
    string += '(Avg {:.4f}) '.format(average_dice)
    # string += 'Acc:{:.3f} '.format(acc)
    # string += '(Avg {:.4f}) '.format(average_acc)
    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr)
    # string += 'Time {:.1f} '.format(batch_time)
    string += '(AvgTime {:.1f})   '.format(average_time)
    summary += string
    logger.info(summary)
    # print summary

##################################################################################
#=================================================================================
#          Train One Epoch
#=================================================================================
##################################################################################
def train_one_epoch(loader, model, criterion, optimizer, writer, epoch, lr_scheduler, model_type, logger,is_Train=False):
    logging_mode = 'Train' if model.training else 'Val'
    end = time.time()
    time_sum, loss_sum, text_l1_sum, text_l2_sum, text_l3_sum, text_l4_sum, = 0, 0, 0, 0, 0, 0
    dice_sum, iou_sum  = 0.0, 0.0
    dices = []
    # TextLoss =  FocalLoss(gamma=2, weight=None) #nn.CrossEntropyLoss() 
    # TextLoss = ContrastiveLoss(margin=0.2,max_violation=False)
    ClsLoss = FocalLoss(gamma=2, weight=None)#nn.CrossEntropyLoss()#
    # SimLoss = nn.CosineEmbeddingLoss()
    for i, (sampled_batch, names) in enumerate(loader, 1):
        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__

        # Take variable and put them to GPU
        # images, masks, text, length = sampled_batch['image'], sampled_batch['label'], sampled_batch['text'], sampled_batch['lengths']
        images, masks, Unilateral_label, num_label,left_loc_label, right_loc_label= sampled_batch['image'], sampled_batch['label'], sampled_batch['Unilateral_label'], sampled_batch['num_label'], sampled_batch['left_loc_label'], sampled_batch['right_loc_label']
        # if text.shape[1] > 10:
        #     text = text[ :, :10, :]
        
        # images, masks = images.cuda(), masks.cuda()
        images, masks, Unilateral_label, num_label,left_loc_label, right_loc_label= images.cuda(), masks.cuda(),  Unilateral_label.cuda(), num_label.cuda(), left_loc_label.cuda(), right_loc_label.cuda()


        # ====================================================
        #             Compute loss
        # ====================================================

        preds, img_emb, Unilateral_img_emb,num_img_emb,left_loc_img_emb,right_loc_img_emb,Unilateral_emb_norm,num_emb_norm,left_loc_emb_norm,right_loc_emb_norm = model(images)
        RetrivalLoss = InfoNceLoss()#MaxHingLoss()
        Unilateral_tripletLoss=RetrivalLoss(img_emb, Unilateral_emb_norm,Unilateral_label)
        num_tripletLoss=RetrivalLoss(img_emb, num_emb_norm,num_label)
        left_loc_tripletLoss=RetrivalLoss(img_emb, left_loc_emb_norm,left_loc_label)
        right_loc_tripletLoss=RetrivalLoss(img_emb, right_loc_emb_norm,right_loc_label)

        clsloss1 = ClsLoss(Unilateral_img_emb,Unilateral_label)
        clsloss2 = ClsLoss(num_img_emb,num_label)
        clsloss3 = ClsLoss(left_loc_img_emb,left_loc_label)
        clsloss4 = ClsLoss(right_loc_img_emb,right_loc_label)
        out_loss = criterion(preds, masks.float())+Unilateral_tripletLoss+num_tripletLoss+left_loc_tripletLoss+right_loc_tripletLoss+clsloss1+clsloss2+clsloss3+clsloss4
        # print(model.training)

        if model.training:
            optimizer.zero_grad()
            out_loss.backward()
            optimizer.step()

        train_dice = criterion._show_dice(preds, masks.float())
        # train_iou = train_dice
        train_iou = iou_on_batch(masks,preds)  # MoNuSeg & Covid19
        batch_time = time.time() - end
        # if not is_Train:
        #     evaluate(logits_per_image.cpu().detach())
        dices.append(train_dice)

        time_sum += len(images) * batch_time
        loss_sum += len(images) * out_loss
        text_l1_sum += len(images) * Unilateral_tripletLoss
        text_l2_sum += len(images) * num_tripletLoss
        text_l3_sum += len(images) * left_loc_tripletLoss
        text_l4_sum += len(images) * right_loc_tripletLoss
        
        iou_sum += len(images) * train_iou
        dice_sum += len(images) * train_dice

        if i == len(loader):
            average_loss = loss_sum / (config.batch_size*(i-1) + len(images))
            average_text_l1 = text_l1_sum / (config.batch_size*(i-1) + len(images))
            average_text_l2 = text_l2_sum / (config.batch_size*(i-1) + len(images))
            average_text_l3 = text_l3_sum / (config.batch_size*(i-1) + len(images))
            average_text_l4 = text_l4_sum / (config.batch_size*(i-1) + len(images))
            average_time = time_sum / (config.batch_size*(i-1) + len(images))
            train_iou_average = iou_sum / (config.batch_size*(i-1) + len(images))
            train_dice_avg = dice_sum / (config.batch_size*(i-1) + len(images))
        else:
            average_loss = loss_sum / (i * config.batch_size)
            average_text_l1 = text_l1_sum / (i * config.batch_size)
            average_text_l2 = text_l2_sum / (i * config.batch_size)
            average_text_l3 = text_l3_sum / (i * config.batch_size)
            average_text_l4 = text_l4_sum / (i * config.batch_size)
            average_time = time_sum / (i * config.batch_size)
            train_iou_average = iou_sum / (i * config.batch_size)
            train_dice_avg = dice_sum / (i * config.batch_size)

        end = time.time()
        torch.cuda.empty_cache()
        if i % config.print_frequency == 0:
            print_summary(epoch + 1, i, len(loader), names, out_loss, loss_name, batch_time,
                          average_loss,average_text_l1,average_text_l2,average_text_l3,average_text_l4, average_time, train_iou, train_iou_average,
                          train_dice, train_dice_avg, 0, 0,  logging_mode,
                          lr=min(g["lr"] for g in optimizer.param_groups),logger=logger)

        if config.tensorboard:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_' + loss_name, out_loss.item(), step)
            writer.add_scalar(logging_mode + '_' + 'Unilateral_tripletLoss', Unilateral_tripletLoss.item(), step)
            writer.add_scalar(logging_mode + '_' + 'num_tripletLoss', num_tripletLoss.item(), step)
            writer.add_scalar(logging_mode + '_' + 'left_loc_tripletLoss', left_loc_tripletLoss.item(), step)
            writer.add_scalar(logging_mode + '_' + 'right_loc_tripletLoss', right_loc_tripletLoss.item(), step)
            # plot metrics in tensorboard
            writer.add_scalar(logging_mode + '_iou', train_iou, step)
            writer.add_scalar(logging_mode + '_dice', train_dice, step)

        torch.cuda.empty_cache()

    if lr_scheduler is not None:
        lr_scheduler.step()

    return average_loss, train_dice_avg
