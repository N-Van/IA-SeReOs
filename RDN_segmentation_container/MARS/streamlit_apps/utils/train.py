import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import utils.dataprocess as dp
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.losses import DomainEnrichLoss, dice_loss, DiceOverlap, Accuracy
import torchvision

from torch.utils.tensorboard import SummaryWriter

bce_losses = nn.BCEWithLogitsLoss()
accuracy = Accuracy()

def rdn_train(net, optimizer, data_loader, epoch=None, total_epoch=None, use_gpu = False, tensorboard_plot=False, nb_ite=0):
    if use_gpu:
        net.cuda()
    else:
        net.cpu()

    # set data_loader
    data_loader1 = data_loader[0] #random
    data_loader2 = data_loader[1] #sorted

    it = iter(enumerate(data_loader2))
    max_batches2 = len(data_loader2.dataset) // data_loader2.batch_size + (1 if (len(data_loader2.dataset) % data_loader2.batch_size) != 0 else 0)
    max_batches1 = len(data_loader1.dataset) // data_loader1.batch_size + (1 if (len(data_loader1.dataset) % data_loader1.batch_size) != 0 else 0)

    # the epoch message for printing
    epoch_print = 'Epoch:'
    if epoch is not None:
        epoch_print += f'{epoch + 1}'
    if total_epoch is not None:
        epoch_print += f'/{total_epoch}'
    last_batches = 0.0
    loss1_sum = 0.0
    loss2_sum = 0.0
    ite = 0
    writer = SummaryWriter("runs")
    with tqdm(total=len(data_loader1.dataset), desc=epoch_print, unit=' batches') as pbar:
        for i_batches, sample_batched in enumerate(data_loader1):
            last_batches = i_batches
            i_batches2, sample_batched2 = next(it)
            if i_batches2 + 1 >= max_batches2:
                it = iter(enumerate(data_loader2))

            mask = sample_batched['mask']
            
            image = sample_batched['image']

            mask2 = sample_batched2['mask']

            image2 = sample_batched2['image']

            index = sample_batched['index']
            
            index2 = sample_batched2['index']

            # convert to gpu
            if use_gpu:
                mask = mask.cuda().long()
                image = image.cuda()
                image2 = image2.cuda()

            # # prediction
            pred = net(image)
            
            loss1 = DomainEnrichLoss()(net, index, mask)
            
            
            mask = dp.create_one_hot(mask)
            if tensorboard_plot and nb_ite+ite == 0:
                writer.add_graph(net, image)
            if tensorboard_plot and (ite % (max_batches1 // 3) == 0):
                if epoch is not None:
                    with torch.no_grad():
                        pred2 = net(image)
                    m2 = mask.argmax(1)
                    m2= m2.cpu().squeeze().data.numpy()
                    pred2 = pred2.argmax(1)
                    pred2 = pred2.cpu().squeeze().data.numpy()
                    color_dict = [[0.0], [128.0/255.0], [1]]
                    pred_img = torch.empty_like(image).copy_(image)
                    mask_img = torch.empty_like(image).copy_(image)
                    for i in range(len(pred_img)):
                        for j in range(len(pred_img[i][0])):
                            for k in range(len(pred_img[i][0][j])):
                                pred_img[i][0][j][k] = color_dict[pred2[i][j][k]][0]
                                mask_img[i][0][j][k] = color_dict[m2[i][j][k]][0]

                    # print("image",image)
                    # print("mask",mask_img)
                    # print("pred",pred_img)
                    writer.add_image('input_image', torchvision.utils.make_grid(image),nb_ite + last_batches)
                    writer.add_image('prediction_image', torchvision.utils.make_grid(pred_img),nb_ite + last_batches)
                    writer.add_image('mask_image', torchvision.utils.make_grid(mask_img),nb_ite + last_batches)
                    output1 = np.concatenate(net.x_rdn1[0].cpu().data.numpy())
                    for i in range(1,len(net.x_rdn1)):
                        output1 = np.concatenate((output1,np.concatenate(net.x_rdn1[i].cpu().data.numpy())),axis=1)
                    output2 = np.concatenate(net.x_rdn2[0].cpu().data.numpy())
                    for i in range(1,len(net.x_rdn2)):
                        output2 = np.concatenate((output2,np.concatenate(net.x_rdn2[i].cpu().data.numpy())),axis=1)

                    writer.add_image('DEB_Bone_image', torchvision.utils.make_grid(torch.from_numpy(output1)), nb_ite + last_batches)
                    writer.add_image('DEB_Dirt_image', torchvision.utils.make_grid(torch.from_numpy(output2)), nb_ite + last_batches)

            #loss2 = 0.25 * bce_losses(pred, mask) + (1 - 0.25) * dice_loss(pred, mask)
            CE_loss = nn.CrossEntropyLoss()
            loss2 = CE_loss(pred, mask) 

            loss = loss2 + loss1
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print results
            pbar.update(mask.shape[0])
            pbar.set_postfix(loss=loss.cpu().data.numpy(),loss1=loss1.cpu().data.numpy(),loss2=loss2.cpu().data.numpy())
            loss1_sum = loss1_sum + loss1.cpu().data.numpy()
            loss2_sum = loss2_sum + loss2.cpu().data.numpy()
            writer.add_scalars('Losses',{'loss':loss.cpu().data.numpy(),'loss1':loss1.cpu().data.numpy(),'loss2':loss2.cpu().data.numpy()}, nb_ite + last_batches)
            writer.add_scalars('Average_Losses',{'loss':(loss2_sum / (last_batches + 1)) + (loss1_sum / (last_batches + 1)),'loss1':(loss1_sum / (last_batches + 1)),'loss2':(loss2_sum / (last_batches + 1))}, nb_ite + last_batches)
            
            ite += 1
        print(f'\nAverage, loss1: {(loss1_sum / (last_batches + 1)):.6f}, loss2: {(loss2_sum/ (last_batches + 1)):.6f}.')
    writer.close()
    return nb_ite + last_batches
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
        writer = SummaryWriter("runs")
        writer.add_scalars('Dice Overlap',{'Air':dice_overlap_results[0],'Dirt':dice_overlap_results[1],'Bone':dice_overlap_results[2]}, i_epoch)
        writer.close()
    return criterion_value, dice_overlap_results