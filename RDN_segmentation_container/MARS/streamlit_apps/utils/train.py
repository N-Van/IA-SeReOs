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
from codecarbon import EmissionsTracker
import datetime
import mlflow
import mlflow.pytorch
import os

def generate_experiment_name():
    return f"streamlit_training-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

def create_or_set_experiment():
    experiment_name = generate_experiment_name()
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    return experiment.experiment_id

def rdn_train(net, optimizer, data_loader, epoch=None, total_epoch=None, use_gpu=False, tensorboard_plot=False, nb_ite=0, writer=None):
    mlflow.end_run()
    experiment_id = create_or_set_experiment()
    run_name = f"epoch_{epoch + 1}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mlflow.log_params({
            "epoch": epoch,
            "total_epochs": total_epoch,
            # Add other parameters as needed
        })

        tracker = EmissionsTracker(project_name="rdn_train")
        tracker.start()

        if use_gpu:
            net.cuda()
        else:
            net.cpu()

        data_loader1, data_loader2 = data_loader
        max_batches2 = len(data_loader2.dataset) // data_loader2.batch_size + (1 if len(data_loader2.dataset) % data_loader2.batch_size != 0 else 0)
        it = iter(enumerate(data_loader2))

        last_batches = 0.0
        loss1_sum = 0.0
        ite = 0

        with tqdm(total=len(data_loader1.dataset), desc='Training', unit=' batches') as pbar:
            for i_batches, sample_batched in enumerate(data_loader1):
                if sample_batched is None:
                    continue

                last_batches = i_batches
                i_batches2, sample_batched2 = next(it)
                if i_batches2 + 1 >= max_batches2:
                    it = iter(enumerate(data_loader2))

                mask = sample_batched['mask']
                image = sample_batched['image']
                mask2 = sample_batched2['mask']
                image2 = sample_batched2['image']

                if use_gpu:
                    mask = mask.cuda().long()
                    image = image.cuda()
                    image2 = image2.cuda()

                pred = net(image)
                CE_loss = nn.CrossEntropyLoss()
                loss2 = CE_loss(pred, mask)
                loss = loss2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(mask.shape[0])
                pbar.set_postfix(loss=loss.cpu().data.numpy(), loss2=loss2.cpu().data.numpy())
                loss1_sum += loss2.cpu().data.numpy()
                ite += 1

            avg_loss2 = loss1_sum / (last_batches + 1)
            mlflow.log_metric("Average Loss", avg_loss2)

            if writer:
                writer.add_scalar('Loss/Train', avg_loss2, epoch)

        tracker.stop()
        mlflow.pytorch.log_model(net, "model")
        return nb_ite + last_batches

def visualize_images(writer, net, image, mask, device, nb_ite, last_batches, epoch):
    with torch.no_grad():
        pred2 = net(image)

    # Convert predictions and masks to 1-channel grayscale
    m2 = mask.argmax(1).cpu().squeeze().unsqueeze(1).float().numpy()
    pred2 = pred2.argmax(1).cpu().squeeze().unsqueeze(1).float().numpy()

    # Normalize the images for visualization
    input_img = image.cpu().numpy()

    if input_img.shape[1] > 3:  # More than 3 channels
        mid_slice = input_img.shape[1] // 2  # Middle slice
        input_img = input_img[:, mid_slice-1:mid_slice + 2, :, :]

    writer.add_image('input_image', torchvision.utils.make_grid(torch.tensor(input_img), normalize=True, scale_each=True), nb_ite + last_batches)
    writer.add_image('mask_image', torchvision.utils.make_grid(torch.tensor(m2), normalize=True, scale_each=True), nb_ite + last_batches)
    writer.add_image('prediction_image', torchvision.utils.make_grid(torch.tensor(pred2), normalize=True, scale_each=True), nb_ite + last_batches)

def rdn_val(net, data_set, use_gpu=False, i_epoch=None, class_num=None, writer=None):
    tracker = EmissionsTracker(project_name="rdn_val")
    tracker.start()

    if use_gpu:
        net.cuda()
    else:
        net.cpu()

    net.eval()
    val_loader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=0)
    total_loss = 0.0
    dice_overlap_results = np.zeros(class_num)

    dice_overlap = DiceOverlap(class_num)  # Assuming DiceOverlap is defined elsewhere

    with torch.no_grad():
        for sample_batched in val_loader:
            mask = sample_batched['mask']
            image = sample_batched['image']

            if use_gpu:
                mask = mask.cuda().long()
                image = image.cuda()

            pred = net(image)
            CE_loss = nn.CrossEntropyLoss()
            loss = CE_loss(pred, mask)
            total_loss += loss.item()

            dice_overlap_results += dice_overlap(pred, mask.long())

    avg_val_loss = total_loss / len(val_loader)
    avg_dice_overlap = dice_overlap_results / len(val_loader.dataset)

    mlflow.log_metric("Validation Loss", avg_val_loss)
    mlflow.log_metrics({
        "Dice Overlap_Air": avg_dice_overlap[0],
        "Dice Overlap_Dirt": avg_dice_overlap[1],
        "Dice Overlap_Bone": avg_dice_overlap[2],
    })

    if writer:
        writer.add_scalar('Loss/Validation', avg_val_loss, i_epoch)
        writer.add_scalars('Dice Overlap', {
            'Air': avg_dice_overlap[0],
            'Dirt': avg_dice_overlap[1],
            'Bone': avg_dice_overlap[2]
        }, i_epoch)

    tracker.stop()
    return avg_val_loss, avg_dice_overlap
