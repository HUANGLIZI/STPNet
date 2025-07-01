# -*- coding: utf-8 -*-
import os
import torch
import time
import ml_collections

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#5
use_cuda = torch.cuda.is_available()
seed = 666 #666
os.environ['PYTHONHASHSEED'] = str(seed)

cosineLR = True  # Use cosineLR or not
n_channels = 3 # 2D: 3, 3D:1
n_labels = 1
epochs = 2000
img_size = 224
print_frequency = 1
save_frequency = 5000
vis_frequency = 8
early_stopping_patience = 100

pretrain = False

# task_name = 'Covid19'
# task_name = 'Kvasir-SEG'
task_name = 'COVID19_CT_new'
learning_rate = 3e-4 
#
batch_size = 24

model_name = 'STPNet'
# model_name = 'STPNet_pretrain'

train_dataset = './dataset/' + task_name + '/Train_Folder/'
val_dataset = './dataset/' + task_name + '/Val_Folder/'
test_dataset = './dataset/' + task_name + '/Test_Folder/'
task_dataset = './dataset/' + task_name + '/Train_Folder/'
session_name = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path = task_name + '/' + model_name + '/' + session_name + '/' #Focal_loss
model_path = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path = save_path + session_name + ".log"
visualize_path = save_path + 'visualize_val/'


##########################################################################
# CTrans configs
##########################################################################
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.expand_ratio = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 64  # base channel of U-Net
    config.n_classes = 1
    return config

# 1.Covid19
test_session = "Test_session_12.07_15h35"

# 2.COVID_CT_new
# test_session = "Test_session_08.03_12h43"

## pretrain
# test_session = "Test_session_07.25_16h36"
