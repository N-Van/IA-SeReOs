import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import utils.dataprocess as dp
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.losses import DomainEnrichLoss, dice_loss, DiceOverlap, Accuracy



bce_losses = nn.BCEWithLogitsLoss()
accuracy = Accuracy()

def rdn_train(net, optimizer, data_loader, epoch=None, total_epoch=None, use_gpu = False):
    if use_gpu:
        net.cuda()
    else:
        net.cpu()

    # set data_loader
    data_loader1 = data_loader[0]
    data_loader2 = data_loader[1]

    it = iter(enumerate(data_loader2))
    max_batches2 = len(data_loader2.dataset) // data_loader2.batch_size + (1 if (len(data_loader2.dataset) % data_loader2.batch_size) != 0 else 0)

    # the epoch message for printing
    epoch_print = 'Epoch:'
    if epoch is not None:
        epoch_print += f'{epoch + 1}'
    if total_epoch is not None:
        epoch_print += f'/{total_epoch}'
    last_batches = 0.0
    loss1_sum = 0.0
    loss2_sum = 0.0
    with tqdm(total=len(data_loader1.dataset), desc=epoch_print, unit=' batches') as pbar:
        for i_batches, sample_batched in enumerate(data_loader1):
            last_batches = i_batches
            i_batches2, sample_batched2 = next(it)
            if i_batches2 + 1 >= max_batches2:
                it = iter(enumerate(data_loader2))

            mask = sample_batched['mask']
            image = sample_batched['image']

            image2 = sample_batched2['image']
            index = sample_batched2['index']

            # convert to gpu
            if use_gpu:
                mask = mask.cuda().long()
                image = image.cuda()
                image2 = image2.cuda()

            # # prediction
            net(image2)
            
            loss1 = DomainEnrichLoss()(net, index)

            pred = F.sigmoid(net(image))
            mask = dp.create_one_hot(mask)
            loss2 = 0.25 * bce_losses(pred, mask) + (1 - 0.25) * dice_loss(pred, mask)

            loss = loss2 + 0.0001*loss1
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print results
            pbar.update(mask.shape[0])
            pbar.set_postfix(loss=loss.cpu().data.numpy(),loss1=loss1.cpu().data.numpy(),loss2=loss2.cpu().data.numpy())
            loss1_sum = loss1_sum + loss1.cpu().data.numpy()
            loss2_sum = loss2_sum + loss2.cpu().data.numpy()

        print(f'\nAverage, loss1: {(loss1_sum / (last_batches + 1)):.6f}, loss2: {(loss2_sum/ (last_batches + 1)):.6f}.')

    ...

def rdn_val(net, data_set, use_gpu = False, i_epoch = None, class_num = 3):

    dice_overlap = DiceOverlap(class_num)
    if use_gpu:
        net.cuda()
    else:
        net.cpu()

    # check whether net is in train mode or not
    origin_is_train_mode = net.training

    # change the net to eval mode
    if origin_is_train_mode:
        net.eval()

    # check whether data set is in train mode
    data_set.val()

    criterion_value_sum = 0.0
    data_loader = DataLoader(data_set, batch_size=1, num_workers=0)
    dice_overlap_results = 0.0

    for i_batches, sample_batched in enumerate(data_loader):
        mask = sample_batched['mask']
        image = sample_batched['image']

        if use_gpu:
            mask = mask.cuda()
            image = image.cuda()

        # prediction
        with torch.no_grad():
            pred = net(image)
            criterion_value_sum += accuracy(pred, mask.long()).cpu().data.numpy()

            if dice_overlap is not None:
                dice_overlap_results += dice_overlap(pred, mask.long())

    criterion_value = criterion_value_sum / len(data_loader.dataset)
    dice_overlap_results = dice_overlap_results / len(data_loader.dataset)
    for i in range(dice_overlap_results.shape[0]):
        print(f'Class: {i:.0f}, Dice Overlap: {dice_overlap_results[i]:.6f}')

    if origin_is_train_mode:
        net.train()

    # print message
    if i_epoch is not None:
        print(f"Epoch: {i_epoch + 1}, Accuracy Value: {criterion_value:.6f}")

    return criterion_value, dice_overlap_results