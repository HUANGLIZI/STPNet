# -*- coding: utf-8 -*-
import numpy as np
import torch
import random
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2
from scipy import ndimage
from transformers import BertTokenizer
from transformers import AutoTokenizer
import re
# from bert_embedding import BertEmbedding


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # image, label = sample['image'], sample['label']
        image, label = image.astype(np.uint8), label.astype(np.uint8)
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        # text = torch.Tensor(text)
        # sample = {'image': image, 'label': label}
        sample = {'image': image, 'label': label, 'Unilateral_label':sample["Unilateral_label"], 'num_label':sample["num_label"], 'left_loc_label':sample["left_loc_label"], 'right_loc_label':sample["right_loc_label"]}
        return sample


class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # image, label = sample['image'], sample['label']
        image, label = image.astype(np.uint8), label.astype(np.uint8)  # OSIC
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        # text = torch.Tensor(text)
        # sample = {'image': image, 'label': label}
        sample = {'image': image, 'label': label, 'Unilateral_label':sample["Unilateral_label"], 'num_label':sample["num_label"], 'left_loc_label':sample["left_loc_label"], 'right_loc_label':sample["right_loc_label"]}  
        return sample


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images



class Poly_dataset(Dataset):

    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,is_train=False,
                 one_hot_mask: int = False,
                 image_size: int = 224) -> None:
        self.is_train=is_train
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'labelcol')
        self.images_list = os.listdir(self.input_path)
        self.mask_list = os.listdir(self.output_path)
        self.one_hot_mask = one_hot_mask
        self.text = row_text
    

        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        # self.rowtext = row_text
        self.task_name = task_name
        # self.bert_embedding = BertEmbedding()
        self.Unilateral_list = []
        self.num_list = []
        self.left_loc_list = []
        self.right_loc_list = []

        with open('./Text/BUnilateral.txt', 'r', encoding='utf-8') as file:
                for line in file:
                    self.Unilateral_list.append(line.strip())
        with open('./Text/num.txt', 'r', encoding='utf-8') as file:
                for line in file:
                    self.num_list.append(line.strip())
        with open('./Text/left_loc.txt', 'r', encoding='utf-8') as file:
                for line in file:
                    self.left_loc_list.append(line.strip())
        with open('./Text/right_loc.txt', 'r', encoding='utf-8') as file:
                for line in file:
                    self.right_loc_list.append(line.strip())
        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            # self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))
       
            # print('self.sentence_class_length[i]',self.sentence_class_length[i]))
    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):
        img_id=idx
        mask_filename = self.mask_list[idx]  # Covid19
        image_filename = mask_filename#.replace('.jpg', '')  # Covid19
        image = cv2.imread(os.path.join(self.input_path, image_filename))
        image = cv2.resize(image, (self.image_size, self.image_size))

        # read mask image
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask[mask <= 0] = 0
        mask[mask > 0] = 1

        # correct dimensions if needed
        image, mask = correct_dims(image, mask)
        if self.is_train:
            text = self.text[str(mask_filename)].replace("all","upper middle lower")
            text_clean=text.strip()
            Unilateral,num,left_loc,right_loc= re.split(r',|and', text_clean.replace(".", "").strip())

            Unilateral_label=self.Unilateral_list.index(Unilateral.strip())
            num_label=self.num_list.index(num.strip())

            left_loc_label=self.left_loc_list.index(left_loc.strip())
            right_loc_label=self.right_loc_list.index(right_loc.strip())
        
            if self.one_hot_mask:
                assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
                mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

            sample = {'image': image, 'label': mask, 'Unilateral_label':Unilateral_label, 'num_label':num_label, 'left_loc_label':left_loc_label, 'right_loc_label':right_loc_label}
            
        else:
            sample = {'image': image, 'label': mask}
        if self.joint_transform:
                sample = self.joint_transform(sample)
        return sample, image_filename
def normalize_position_description(description):
    # 使用正则表达式进行匹配和替换
    pattern = re.compile(r',\s*')  # 匹配逗号后面的空格，无论多少个空格
    description = re.sub(pattern, ', ', description)

    # 在此添加其他可能的规范化步骤

    return description
def process(cleaned_sentence):
    cleaned_sentence=cleaned_sentence.replace(" two one infected area,"," multiple infected areas,")
    cleaned_sentence=cleaned_sentence.replace(" two infected area,"," multiple infected areas,")
    cleaned_sentence=cleaned_sentence.replace("all","upper middle lower")
    cleaned_sentence=cleaned_sentence.replace(" al "," upper middle lower ")
    cleaned_sentence=cleaned_sentence.replace("two","multiple")
    cleaned_sentence=cleaned_sentence.replace("three","multiple")
    cleaned_sentence=cleaned_sentence.replace("four","multiple")
    cleaned_sentence=cleaned_sentence.replace("five","multiple")
    cleaned_sentence=cleaned_sentence.replace("six","multiple")
    cleaned_sentence=cleaned_sentence.replace("seven","multiple")
    cleaned_sentence=cleaned_sentence.replace("eight","multiple")
    cleaned_sentence=cleaned_sentence.replace("nine","multiple")
    cleaned_sentence=cleaned_sentence.replace("ten","multiple")
    if "multiple" in cleaned_sentence:
        cleaned_sentence = cleaned_sentence.replace(" area,"," areas,")
    
    
    cleaned_sentence=cleaned_sentence.replace(" lower lower "," lower ")
    cleaned_sentence=cleaned_sentence.replace(" upperlower "," upper lower ")
    cleaned_sentence=cleaned_sentence.replace(" upperl "," upper ")
    cleaned_sentence=cleaned_sentence.replace(" middlelower "," middle lower ")
    
    cleaned_sentence=cleaned_sentence.replace(" lowerl "," lower ")
    cleaned_sentence=cleaned_sentence.replace(" llower "," lower ")
    cleaned_sentence=cleaned_sentence.replace(" lower l "," lower ")
    cleaned_sentence=cleaned_sentence.replace(" ower "," lower ")
    cleaned_sentence=cleaned_sentence.replace(" lowoer "," lower ")
    
    cleaned_sentence=cleaned_sentence.replace(" lowe "," lower ")
    cleaned_sentence=cleaned_sentence.replace(" loweer "," lower ")

    cleaned_sentence=cleaned_sentence.replace(" lowerupper "," lower upper ")
    cleaned_sentence=cleaned_sentence.replace(" lupper ower "," upper lower ")
    cleaned_sentence=cleaned_sentence.replace(" uppper "," upper ")
    cleaned_sentence=cleaned_sentence.replace(" lupper "," upper ")
    cleaned_sentence=cleaned_sentence.replace(" uooer "," upper ")
    cleaned_sentence=cleaned_sentence.replace("middle  left ","middle left")
    cleaned_sentence=cleaned_sentence.replace("leff ","left")
    cleaned_sentence=cleaned_sentence.replace(" left and "," left lung and ")
    cleaned_sentence=cleaned_sentence.replace(" left and "," left lung and ")

    cleaned_sentence=cleaned_sentence.replace("one  infected","one infected")
    cleaned_sentence=cleaned_sentence.replace("higher","upper")
    cleaned_sentence=cleaned_sentence.replace("lwoer","lower")
    cleaned_sentence=cleaned_sentence.replace("leftlung","left lung")
    cleaned_sentence=cleaned_sentence.replace("aupper","upper")
    cleaned_sentence=cleaned_sentence.replace("lift","left")
    cleaned_sentence=cleaned_sentence.replace(" middlel "," middle ")
    cleaned_sentence=cleaned_sentence.replace(" iddle "," middle ")
    cleaned_sentence=cleaned_sentence.replace(" midle "," middle ")
    cleaned_sentence=cleaned_sentence.replace(" ll ","")
    cleaned_sentence=cleaned_sentence.replace("(wait","")
    cleaned_sentence=cleaned_sentence.replace(" eft "," left ")
    cleaned_sentence=cleaned_sentence.replace(" mmiddle "," middle ")
    cleaned_sentence=cleaned_sentence.replace(" midlle "," middle ")
    cleaned_sentence=cleaned_sentence.replace(" midldle "," middle ")
    cleaned_sentence=cleaned_sentence.replace(" midddle "," middle ")
    cleaned_sentence=cleaned_sentence.replace(" mddle "," middle ")
    cleaned_sentence=cleaned_sentence.replace(" middlr "," middle ")
    cleaned_sentence=cleaned_sentence.replace(" midde "," middle ")

    cleaned_sentence=cleaned_sentence.replace("lowerleft ","lower left ")
    cleaned_sentence=cleaned_sentence.replace("lowerright ","lower right ")
    cleaned_sentence=cleaned_sentence.replace("leflung","left lung")
    cleaned_sentence=cleaned_sentence.replace("rlower ight ","lower right ")
    cleaned_sentence=cleaned_sentence.replace(" ung "," lung ")
    cleaned_sentence=cleaned_sentence.replace("  "," ")
    cleaned_sentence=cleaned_sentence.replace(" rightlung"," right lung")
    cleaned_sentence=cleaned_sentence.replace(" right right "," right ")
    cleaned_sentence=cleaned_sentence.replace(" rught "," right ")

    if 'and' in cleaned_sentence and 'lung.' in cleaned_sentence:
    # 判断句子是否以 'right ' 结尾
        if not cleaned_sentence.endswith('right lung.'):
            # 如果不是，则在句子末尾加上 'right '
            cleaned_sentence=cleaned_sentence.replace(" lung."," right lung.")
    if not cleaned_sentence.endswith('.'):
            # 如果不是，则在句子末尾加上 'right '
            cleaned_sentence += '.'
    cleaned_sentence=cleaned_sentence.replace("  "," ")
    cleaned_sentence = normalize_position_description(cleaned_sentence)
    return cleaned_sentence
import pickle       
class ImageToImage2D(Dataset):

    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,is_train=False,
                 one_hot_mask: int = False,
                 image_size: int = 224) -> None:
        self.is_train=is_train
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'labelcol')
        self.images_list = os.listdir(self.input_path)
        self.mask_list = os.listdir(self.output_path)
        self.one_hot_mask = one_hot_mask
        self.text = row_text
    
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.task_name = task_name
        self.Unilateral_list = []
        self.num_list = []
        self.left_loc_list = []
        self.right_loc_list = []

        with open('./Text/BUnilateral.txt', 'r', encoding='utf-8') as file:
                for line in file:
                    self.Unilateral_list.append(line.strip())
        with open('./Text/num.txt', 'r', encoding='utf-8') as file:
                for line in file:
                    self.num_list.append(line.strip())
        with open('./Text/left_loc.txt', 'r', encoding='utf-8') as file:
                for line in file:
                    self.left_loc_list.append(line.strip())
        with open('./Text/right_loc.txt', 'r', encoding='utf-8') as file:
                for line in file:
                    self.right_loc_list.append(line.strip())
        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))
    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):
        img_id=idx
        mask_filename = self.mask_list[idx]  # Covid19
        image_filename = mask_filename.replace('mask_', '')  # Covid19
        image = cv2.imread(os.path.join(self.input_path, image_filename))
        image = cv2.resize(image, (self.image_size, self.image_size))

        # read mask image
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask[mask <= 0] = 0
        mask[mask > 0] = 1

        # correct dimensions if needed
        image, mask = correct_dims(image, mask)
        text = self.text[mask_filename].replace("all","upper middle lower")
        text_clean=process(text.strip()) 
        if "and" in text_clean:
            Unilateral,num,left_loc,right_loc= re.split(r',|and', text_clean.replace(".", "").strip())
            if left_loc.strip()=='':
                left_loc = "no lesion in left lung"
            if right_loc.strip()=='':
                right_loc = "no lesion in right lung"
        elif "left" in text_clean:
            Unilateral,num,left_loc=text_clean.replace(".","").split(",")
            right_loc = "no lesion in right lung"
        elif "right" in text_clean:
            Unilateral,num,right_loc=text_clean.replace(".","").split(",")
            left_loc = "no lesion in left lung"
        
        if self.is_train:
            Unilateral_label=self.Unilateral_list.index(Unilateral.strip())
            num_label=self.num_list.index(num.strip())
            left_loc_label=self.left_loc_list.index(left_loc.strip())
            right_loc_label=self.right_loc_list.index(right_loc.strip())
        else:
            Unilateral_label=0
            num_label=0
            left_loc_label=0
            right_loc_label=0
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=30,
        )
        
        lengths = len([t for t in tokens["input_ids"][0] if t != 0])        
        
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        sample = {'image': image, 'label': mask,'text':tokens["input_ids"],'token_type_ids':tokens["token_type_ids"],'attention_mask':tokens["attention_mask"],'lengths':lengths,'img_id':img_id, 'Unilateral_label':Unilateral_label, 'num_label':num_label, 'left_loc_label':left_loc_label, 'right_loc_label':right_loc_label}

        if self.joint_transform:
            sample = self.joint_transform(sample)
        return sample, image_filename