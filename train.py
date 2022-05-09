
import os
import random
import time
import datetime
import numpy as np
import albumentations as A
import cv2
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from text2embed import Text2Embed
from utils import seeding, create_dir, print_and_save, shuffling, epoch_time, calculate_metrics, label_dictionary, mask_to_bbox
from model import TGAPolypSeg
from metrics import DiceLoss, DiceBCELoss, MultiClassBCE

def load_names(path, file_path):
    f = open(file_path, "r")
    data = f.read().split("\n")[:-1]
    images = [os.path.join(path,"images", name) + ".jpg" for name in data]
    masks = [os.path.join(path,"masks", name) + ".jpg" for name in data]
    return images, masks

def load_data(path):
    train_names_path = f"{path}/train.txt"
    valid_names_path = f"{path}/val.txt"

    train_x, train_y = load_names(path, train_names_path)
    valid_x, valid_y = load_names(path, valid_names_path)

    label_dict = label_dictionary()
    train_label = len(train_x) * [label_dict["polyp"]]
    valid_label = len(valid_x) * [label_dict["polyp"]]

    return (train_x, train_y, train_label), (valid_x, valid_y, valid_label)

class DATASET(Dataset):
    def __init__(self, images_path, labels_path, masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.labels_path = labels_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)

        self.embed = Text2Embed()

    def mask_to_text(self, mask):
        bboxes = mask_to_bbox(mask)
        num_polyps = 0 if len(bboxes) == 1 else 1

        polyp_sizes = None
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            h = (y2 - y1)
            w = (x2 - x1)
            area = (h * w) / (mask.shape[0] * mask.shape[1])

            if area < 0.10:
                polyp_sizes = 0
            elif area >= 0.30:
                polyp_sizes = 2
            elif area >= 0.10 and area < 0.30:
                polyp_sizes = 1

        return np.array(num_polyps), np.array(polyp_sizes)

    def __getitem__(self, index):
        """ Reading Image & Mask """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

        """ Applying Data Augmentation """
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        """ Image """
        image = cv2.resize(image, size)
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0

        """ Mask """
        mask = cv2.resize(mask, size)
        mask_copy = mask
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255.0

        """ Mask to Textual information """
        num_polyps, polyp_sizes = self.mask_to_text(mask_copy)

        """ Label """
        label = []
        words = self.labels_path[index]
        for word in words:
            word_embed = self.embed.to_embed(word)[0]
            label.append(word_embed)
        label = np.array(label)

        return (image, label), (mask, num_polyps, polyp_sizes)

    def __len__(self):
        return self.n_samples

def train(model, loader, optimizer, loss_fn, device):
    model.train()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    for i, ((x, l), (y1, y2, y3)) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        l = l.to(device, dtype=torch.float32)
        y1 = y1.to(device, dtype=torch.float32)
        y2 = y2.to(device, dtype=torch.long)
        y3 = y3.to(device, dtype=torch.long)

        optimizer.zero_grad()
        p1, p2, p3 = model(x, l)
        p2 = torch.softmax(p2, dim=1)
        p3 = torch.softmax(p3, dim=1)

        loss1 = loss_fn[0](p1, y1)
        loss2 = loss_fn[1](p2, y2)
        loss3 = loss_fn[2](p3, y3)
        loss = loss1 + loss2 + loss3
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        """ Calculate the metrics """
        batch_jac = []
        batch_f1 = []
        batch_recall = []
        batch_precision = []

        for yt, yp in zip(y1, p1):
            score = calculate_metrics(yt, yp)
            batch_jac.append(score[0])
            batch_f1.append(score[1])
            batch_recall.append(score[2])
            batch_precision.append(score[3])

        epoch_jac += np.mean(batch_jac)
        epoch_f1 += np.mean(batch_f1)
        epoch_recall += np.mean(batch_recall)
        epoch_precision += np.mean(batch_precision)

    epoch_loss = epoch_loss/len(loader)
    epoch_jac = epoch_jac/len(loader)
    epoch_f1 = epoch_f1/len(loader)
    epoch_recall = epoch_recall/len(loader)
    epoch_precision = epoch_precision/len(loader)

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]

def evaluate(model, loader, loss_fn, device):
    model.eval()

    epoch_loss = 0
    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    with torch.no_grad():
        for i, ((x, l), (y1, y2, y3)) in enumerate(loader):
            x = x.to(device, dtype=torch.float32)
            l = l.to(device, dtype=torch.float32)
            y1 = y1.to(device, dtype=torch.float32)
            y2 = y2.to(device, dtype=torch.long)
            y3 = y3.to(device, dtype=torch.long)

            p1, p2, p3 = model(x, l)
            p2 = torch.softmax(p2, dim=1)
            p3 = torch.softmax(p3, dim=1)

            loss1 = loss_fn[0](p1, y1)
            loss2 = loss_fn[1](p2, y2)
            loss3 = loss_fn[2](p3, y3)
            loss = loss1 + loss2 + loss3
            epoch_loss += loss.item()

            """ Calculate the metrics """
            batch_jac = []
            batch_f1 = []
            batch_recall = []
            batch_precision = []

            for yt, yp in zip(y1, p1):
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])

            epoch_jac += np.mean(batch_jac)
            epoch_f1 += np.mean(batch_f1)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)

        epoch_loss = epoch_loss/len(loader)
        epoch_jac = epoch_jac/len(loader)
        epoch_f1 = epoch_f1/len(loader)
        epoch_recall = epoch_recall/len(loader)
        epoch_precision = epoch_precision/len(loader)

        return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("files")

    """ Training logfile """
    train_log_path = "files/train_log.txt"
    if os.path.exists(train_log_path):
        print("Log file exists")
    else:
        train_log = open("files/train_log.txt", "w")
        train_log.write("\n")
        train_log.close()

    """ Record Date & Time """
    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    """ Hyperparameters """
    image_size = 256
    size = (image_size, image_size)
    batch_size = 16
    num_epochs = 500
    lr = 1e-4
    early_stopping_patience = 50
    checkpoint_path = "files/checkpoint.pth"
    path = "/media/nikhil/Seagate Backup Plus Drive/ML_DATASET/Kvasir-SEG"

    data_str = f"Image Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    data_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    print_and_save(train_log_path, data_str)

    """ Data augmentation: Transforms """
    transform =  A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ])

    """ Dataset """
    (train_x, train_y, train_label), (valid_x, valid_y, valid_label) = load_data(path)
    train_x, train_y, train_label = shuffling(train_x, train_y, train_label)
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print_and_save(train_log_path, data_str)

    """ Dataset and loader """
    train_dataset = DATASET(train_x, train_label, train_y, (image_size, image_size), transform=transform)
    valid_dataset = DATASET(valid_x, valid_label, valid_y, (image_size, image_size), transform=None)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    """ Model """
    device = torch.device('cuda')
    model = TGAPolypSeg()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = [DiceBCELoss(), nn.CrossEntropyLoss(), nn.CrossEntropyLoss()]
    loss_name = "BCE Dice Loss"
    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    """ Training the model """
    best_valid_metrics = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_metrics = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)

        if valid_metrics[1] > best_valid_metrics:
            data_str = f"Valid F1 improved from {best_valid_metrics:2.4f} to {valid_metrics[1]:2.4f}. Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)

            best_valid_metrics = valid_metrics[1]
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0

        elif valid_metrics[1] < best_valid_metrics:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.4f} - Jaccard: {train_metrics[0]:.4f} - F1: {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n"
        data_str += f"\t Val. Loss: {valid_loss:.4f} - Jaccard: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
        print_and_save(train_log_path, data_str)

        if early_stopping_count == early_stopping_patience:
            data_str = f"Early stopping: validation loss stops improving from last {early_stopping_patience} continously.\n"
            print_and_save(train_log_path, data_str)
            break
