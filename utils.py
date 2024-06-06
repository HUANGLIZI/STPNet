import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, jaccard_score
import cv2
from torch import nn
import torch.nn.functional as F
import math
from functools import wraps
import warnings
import weakref
from PIL import Image
from numpy import average, dot, linalg
from transformers import AutoTokenizer,AutoModel
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer
import os

class WeightedBCE(nn.Module):

    def __init__(self, weights=[0.4, 0.6]):
        super(WeightedBCE, self).__init__()
        self.weights = weights

    def forward(self, logit_pixel, truth_pixel):
        # print("====",logit_pixel.size())
        logit = logit_pixel.view(-1)
        truth = truth_pixel.view(-1)
        assert (logit.shape == truth.shape)
        loss = F.binary_cross_entropy(logit, truth, reduction='none')
        pos = (truth > 0.5).float()
        neg = (truth < 0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (self.weights[0] * pos * loss / pos_weight + self.weights[1] * neg * loss / neg_weight).sum()

        return loss


class WeightedDiceLoss(nn.Module):
    def __init__(self, weights=[0.5, 0.5]):  # W_pos=0.8, W_neg=0.2
        super(WeightedDiceLoss, self).__init__()
        self.weights = weights

    def forward(self, logit, truth, smooth=1e-5):
        batch_size = len(logit)
        logit = logit.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert (logit.shape == truth.shape)
        p = logit.view(batch_size, -1)
        t = truth.view(batch_size, -1)
        w = truth.detach()
        w = w * (self.weights[1] - self.weights[0]) + self.weights[0]
        p = w * (p)
        t = w * (t)
        intersection = (p * t).sum(-1)
        union = (p * p).sum(-1) + (t * t).sum(-1)
        dice = 1 - (2 * intersection + smooth) / (union + smooth)

        loss = dice.mean()
        return loss


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, inputs, targets):
        N = targets.size()[0]
        smooth = 1
        input_flat = inputs.view(N, -1)
        targets_flat = targets.view(N, -1)
        intersection = input_flat + targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        loss = 1 - N_dice_eff.sum() / N
        return loss


class MultiClassDiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None):
        super(MultiClassDiceLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.dice_loss = WeightedDiceLoss()

    def forward(self, inputs, targets):
        assert inputs.shape == targets.shape, "predict & target shape do not match"
        total_loss = 0
        for i in range(5):
            dice_loss = self.dice_loss(inputs[:, i], targets[:, i])
            total_loss += dice_loss
            total_loss = total_loss / 5
        return total_loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        dice1 = self._dice_loss(inputs[:, 1], target[:, 1]) * weight[1]
        dice2 = self._dice_loss(inputs[:, 2], target[:, 2]) * weight[2]
        dice3 = self._dice_loss(inputs[:, 3], target[:, 3]) * weight[3]
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes, dice1, dice2, dice3


class WeightedDiceCE(nn.Module):
    def __init__(self, dice_weight=0.5, CE_weight=0.5):
        super(WeightedDiceCE, self).__init__()
        self.CE_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(4)  # OSIC: 4, RITE: 5
        self.CE_weight = CE_weight
        self.dice_weight = dice_weight

    def _show_dice(self, inputs, targets):
        # inputs = inputs.argmax(dim=1)
        dice, dice1, dice2, dice3 = self.dice_loss(inputs, targets)
        hard_dice_coeff = 1 - dice
        dice01 = 1 - dice1
        dice02 = 1 - dice2
        dice03 = 1 - dice3
        torch.cuda.empty_cache()
        return hard_dice_coeff, dice01, dice02, dice03

    def forward(self, inputs, targets):
        targets = targets.long()
        dice_CE_loss = self.dice_loss(inputs, targets)
        torch.cuda.empty_cache()
        return dice_CE_loss


class WeightedDiceBCE_unsup(nn.Module):
    def __init__(self, dice_weight=1, BCE_weight=1):
        super(WeightedDiceBCE_unsup, self).__init__()
        self.BCE_loss = WeightedBCE(weights=[0.5, 0.5])
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5])
        self.BCE_weight = BCE_weight
        self.dice_weight = dice_weight

    def _show_dice(self, inputs, targets):
        inputs[inputs >= 0.5] = 1
        inputs[inputs < 0.5] = 0
        targets[targets > 0] = 1
        targets[targets <= 0] = 0
        hard_dice_coeff = 1.0 - self.dice_loss(inputs, targets)
        return hard_dice_coeff

    def forward(self, inputs, targets, LV_loss):
        dice = self.dice_loss(inputs, targets)
        BCE = self.BCE_loss(inputs, targets)
        dice_BCE_loss = self.dice_weight * dice + self.BCE_weight * BCE + 0.1 * LV_loss

        return dice_BCE_loss


class WeightedDiceBCE(nn.Module):
    def __init__(self, dice_weight=1, BCE_weight=1):
        super(WeightedDiceBCE, self).__init__()
        self.BCE_loss = WeightedBCE(weights=[0.5, 0.5])
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5])
        self.BCE_weight = BCE_weight
        self.dice_weight = dice_weight

    def _show_dice(self, inputs, targets):
        inputs[inputs >= 0.5] = 1
        inputs[inputs < 0.5] = 0
        targets[targets > 0] = 1
        targets[targets <= 0] = 0
        hard_dice_coeff = 1.0 - self.dice_loss(inputs, targets)
        return hard_dice_coeff

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        BCE = self.BCE_loss(inputs, targets)
        dice_BCE_loss = self.dice_weight * dice + self.BCE_weight * BCE

        return dice_BCE_loss
        
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
        return focal_loss

def auc_on_batch(masks, pred):
    '''Computes the mean Area Under ROC Curve over a batch during training'''
    aucs = []
    for i in range(pred.shape[1]):
        prediction = pred[i][0].cpu().detach().numpy()
        # print("www",np.max(prediction), np.min(prediction))
        mask = masks[i].cpu().detach().numpy()
        # print("rrr",np.max(mask), np.min(mask))
        aucs.append(roc_auc_score(mask.reshape(-1), prediction.reshape(-1)))
    return np.mean(aucs)


def iou_on_batch(masks, pred):
    '''Computes the mean Area Under ROC Curve over a batch during training'''
    ious = []

    for i in range(pred.shape[0]):
        pred_tmp = pred[i][0].cpu().detach().numpy()
        mask_tmp = masks[i].cpu().detach().numpy()
        pred_tmp[pred_tmp >= 0.5] = 1
        pred_tmp[pred_tmp < 0.5] = 0
        mask_tmp[mask_tmp > 0] = 1
        mask_tmp[mask_tmp <= 0] = 0
        ious.append(jaccard_score(mask_tmp.reshape(-1), pred_tmp.reshape(-1)))
    return np.mean(ious)


def dice_coef(y_true, y_pred):
    smooth = 1e-5
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_on_batch(masks, pred):
    '''Computes the mean Area Under ROC Curve over a batch during training'''
    dices = []

    for i in range(pred.shape[0]):
        pred_tmp = pred[i][0].cpu().detach().numpy()
        mask_tmp = masks[i].cpu().detach().numpy()
        pred_tmp[pred_tmp >= 0.5] = 1
        pred_tmp[pred_tmp < 0.5] = 0
        mask_tmp[mask_tmp > 0] = 1
        mask_tmp[mask_tmp <= 0] = 0
        dices.append(dice_coef(mask_tmp, pred_tmp))
    return np.mean(dices)


def save_on_batch(images1, masks, pred, names, vis_path):
    '''Computes the mean Area Under ROC Curve over a batch during training'''
    for i in range(pred.shape[0]):
        pred_tmp = pred[i][0].cpu().detach().numpy()
        mask_tmp = masks[i].cpu().detach().numpy()
        pred_tmp[pred_tmp >= 0.5] = 255
        pred_tmp[pred_tmp < 0.5] = 0
        mask_tmp[mask_tmp > 0] = 255
        mask_tmp[mask_tmp <= 0] = 0

        cv2.imwrite(vis_path + names[i][:-4] + "_pred.jpg", pred_tmp)
        cv2.imwrite(vis_path + names[i][:-4] + "_gt.jpg", mask_tmp)


class _LRScheduler(object):

    def __init__(self, optimizer, last_epoch=-1):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0

        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr

    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()

        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class CosineAnnealingWarmRestarts(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)
    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0` after restart, set :math:`\eta_t=\eta_{max}`.
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min

        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)

        self.T_cur = self.last_epoch

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", DeprecationWarning)

        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """Step could be called after every batch update
        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> iters = len(dataloader)
            >>> for epoch in range(20):
            >>>     for i, sample in enumerate(dataloader):
            >>>         inputs, labels = sample['inputs'], sample['labels']
            >>>         scheduler.step(epoch + i / iters)
            >>>         optimizer.zero_grad()
            >>>         outputs = net(inputs)
            >>>         loss = criterion(outputs, labels)
            >>>         loss.backward()
            >>>         optimizer.step()
        This function can be called in an interleaved way.
        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> for epoch in range(20):
            >>>     scheduler.step()
            >>> scheduler.step(26)
            >>> scheduler.step() # scheduler.step(27), instead of scheduler(20)
        """

        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

def read_attr(filename):
    df = pd.read_excel(filename, engine='openpyxl')
    text = {}
    for i in df.index.values:  # Gets the index of the row number and traverses it
        text[df.Image[i]] = df.Description[i]
    if not text:
        text = 0
    return text  # return dict (key: values)
        
def read_text(filename):
    df = pd.read_excel(filename)
    text = {}
    for i in df.index.values:  # Gets the index of the row number and traverses it
        count = len(df.Description[i].split())
        if count < 9:
            df.Description[i] = df.Description[i] + ' EOF XXX' * (9 - count)
        text[df.Image[i]] = df.Description[i]
    return text  # return dict (key: values)


def read_text_LV(filename):
    df = pd.read_excel(filename)
    text = {}
    for i in df.index.values:  # Gets the index of the row number and traverses it
        count = len(df.Description[i].split())
        if count < 30:
            df.Description[i] = df.Description[i] + ' EOF XXX' * (20 - count)  # LV_loss: 24
        text[df.Image[i]] = df.Description[i]
    return text  # return dict (key: values)


# Unification images processing
def get_thum(image, size=(224, 224), greyscale=False):
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        image = image.convert('L')
    return image


# Calculate the cosine distance between pictures
def img_similarity_vectors_via_numpy(image1, image2):
    image1 = get_thum(image1)
    image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_turple in image.getdata():
            vector.append(average(pixel_turple))
        vectors.append(vector)
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    res = dot(a / a_norm, b / b_norm)
    return res


def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities

class InfoNceLoss(nn.Module):

    def __init__(self, temperature=0.07, ):
        super().__init__()

        self.temperature = temperature
        logit_scale_init_value=0.07
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/logit_scale_init_value)))
        self.lossfunc = FocalLoss()
    def forward(self, im, s, text_label):
        sim_mat = get_sim(im, s)
        text_label = text_label.cuda()
        logits = sim_mat *self.logit_scale
        loss = self.lossfunc(logits, text_label)
        return loss        


class ImageTextContrastiveLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        img_labels=None,
        text_labels=None,
        aug_input_ids=None,
        aug_attention_mask=None,
        **kwargs,
        ):
        '''args:
        labels: the image corresponds to which classes of diagnoses
        text_labels: the text corresponds to which classes of diagnoses
        '''
        if img_labels is None or text_labels is None:
            '''use hard clip loss as the original clip
            '''
            outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    return_loss=True,
                    )
        else:
            '''use soft clip loss
            '''
            outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    return_loss=False,
                    )

            # get logits
            logits = outputs['logits']

            # compute soft-labels, -1: negative, 0: uncertain, 1: positive
            # in the original data: 1: positive, 0: negative, -1: uncertain, NA: not mentioned
            label_sim = torch.matmul(img_labels, text_labels.T)
            label_sim = label_sim.to(logits.device)

            if aug_input_ids is not None:
                aug_text_embeds = self.model.encode_text(aug_input_ids, aug_attention_mask)
                img_embeds = outputs['img_embeds']
                logits_aug = self.model.compute_logits(img_embeds, aug_text_embeds)
                aug_loss_value = self._soft_clip_loss(logits_aug, label_sim)
                loss_value = self._soft_clip_loss(logits, label_sim)
                outputs['loss_value'] = (aug_loss_value + loss_value) / 2
            else:
                outputs['loss_value'] = self._soft_clip_loss(logits, label_sim)

        return_res = {
            'loss_value': outputs['loss_value'],
        }
        return return_res

    def _soft_clip_loss(self, logits_per_img, soft_label):
        '''take labels of images and sentences as a softlabel
        e.g., image_label = [1, 0, 1, -1], sentence_label = [0, 0, 1, -1]
        this pair has similarity as: 1 * 0 + 0 * 0 + 1 * 1 + -1 * -1 = 2.
        We will clamp the similarity into [-1,1], and take softmax as a soft-label.
        '''
        # when using InfoNCE-like loss
        image_loss = self._soft_xent_loss(logits_per_img, F.softmax(soft_label,1))
        caption_loss = self._soft_xent_loss(logits_per_img.T, F.softmax(soft_label.T,1))
        return (image_loss + caption_loss) / 2

        # when using multilabel bce loss
        # image_loss = self._soft_bce_loss(logits_per_img, soft_label)
        # return image_loss

    def _soft_xent_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax(input, dim = 1)
        return  -(target * logprobs).sum() / input.shape[0]

    def _soft_bce_loss(self, input, target):
        return nn.functional.binary_cross_entropy_with_logits(input, target)
    
def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())
    
def create_dir(path):
    """ Create a directory. """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: creating directory with name {path}")