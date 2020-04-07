import random
import datetime
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import (
    MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR
)
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.models as models
import albumentations as albm
from tqdm import tqdm
from .data import NORMALIZE_PARAMS, CustomDataSet, Transformed
from .transforms import Partial


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_model(name: str, pretrained: bool) -> nn.Module:
    build_fn = getattr(models, name)
    model = build_fn(num_classes=1, pretrained=False)
    if pretrained:
        pretrained_model = build_fn(pretrained=True)
        state_dict = pretrained_model.state_dict()
        del state_dict["fc.weight"], state_dict["fc.bias"]
        model.load_state_dict(state_dict)
    return model


def get_loaders(
        image_dir, annot_file, cv_splits, cv_fold,
        aug_rot_limit, aug_gamma_limit,
        aug_brightness_limit, aug_contrast_limit,
        input_size, batch_size, seed, cv_split_seed):

    def worker_init_fn(worker_id):
        init_seed(seed + worker_id)

    dataset = CustomDataSet(image_dir, annot_file)

    cv = list(
        StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=cv_split_seed)
        .split(np.arange(len(dataset)), dataset.get_all_labels())
    )
    train_indices, val_indices = cv[cv_fold]

    train_transforms, val_transforms = get_transforms(
        input_size=input_size,
        aug_rot_limit=aug_rot_limit,
        aug_gamma_limit=aug_gamma_limit,
        aug_brightness_limit=aug_brightness_limit,
        aug_contrast_limit=aug_contrast_limit)

    train_dataset = Transformed(
        Subset(dataset, train_indices),
        transforms=Compose([
            lambda data: train_transforms(**data),
            Partial(ToTensor(), "image"),
            Partial(Normalize(**NORMALIZE_PARAMS), "image"),
        ]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, drop_last=True, shuffle=False,
        worker_init_fn=worker_init_fn,
        num_workers=10, pin_memory=True)

    val_dataset = Transformed(
        Subset(dataset, val_indices),
        transforms=Compose([
            lambda data: val_transforms(**data),
            Partial(ToTensor(), "image"),
            Partial(Normalize(**NORMALIZE_PARAMS), "image"),
        ]))
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size, drop_last=False, shuffle=False,
        worker_init_fn=worker_init_fn,
        num_workers=10, pin_memory=True)

    return train_loader, val_loader


def get_transforms(
        input_size, aug_rot_limit,
        aug_gamma_limit, aug_brightness_limit, aug_contrast_limit):
    train_transforms = albm.Compose([
        albm.RandomCrop(input_size, input_size),
        albm.RandomGamma(
            gamma_limit=aug_gamma_limit, p=1.0),
        albm.RandomBrightnessContrast(
            brightness_limit=aug_brightness_limit,
            contrast_limit=aug_contrast_limit, p=1.0),
        albm.HorizontalFlip()
    ])
    val_transforms = albm.Compose([
        albm.CenterCrop(input_size, input_size)
    ])
    return train_transforms, val_transforms


def get_optimizer(name: str, config: dict, parameters):
    optimizers = {
        "sgd": SGD,
        "adam": Adam,
    }
    if name in optimizers:
        return optimizers[name](parameters, **config)
    else:
        raise NotImplementedError


def get_scheduler(name: str, config: dict, optimizer):
    schedulers = {
        "step": MultiStepLR,
        "plateau": ReduceLROnPlateau,
        "cosine": CosineAnnealingLR
    }
    if name in schedulers:
        return schedulers[name](optimizer, **config)
    else:
        raise NotImplementedError


def get_criterion(name: str, config: dict):
    criterions = {
        "bce": nn.BCEWithLogitsLoss
    }
    if name in criterions:
        return criterions[name](**config)
    else:
        raise NotImplementedError


def run(name, seed, device_id,
        model_name, pretrained, weights, num_epochs,
        image_dir, annot_file, cv_splits, cv_fold,
        aug_rot_limit, aug_gamma_limit, aug_brightness_limit, aug_contrast_limit,
        input_size, batch_size, cv_split_seed,
        criterion_name, criterion_config,
        optimizer_name, optimizer_config,
        scheduler_name, scheduler_config,
        checkpoint_dir, tensorboard_dir):

    trial_id = f"{name}-fold{cv_fold}-{datetime.datetime.now():%Y%m%d%H%M%S}"

    init_seed(seed)

    device = torch.device(f"cuda:{device_id}")

    model = get_model(model_name, pretrained)
    if weights is not None:
        state_dict = torch.load(weights, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)

    dp_model = nn.DataParallel(model)

    train_loader, val_loader = get_loaders(
        image_dir, annot_file, cv_splits, cv_fold,
        aug_rot_limit, aug_gamma_limit,
        aug_brightness_limit, aug_contrast_limit,
        input_size, batch_size,
        seed=seed, cv_split_seed=cv_split_seed)

    optimizer = get_optimizer(optimizer_name, optimizer_config, model.parameters())
    lr_scheduler = get_scheduler(scheduler_name, scheduler_config, optimizer)

    criterion = get_criterion(criterion_name, criterion_config)

    writer = SummaryWriter(
        log_dir=str(Path(tensorboard_dir) / trial_id))

    def log_scalar(key, value, step):
        writer.add_scalar(key, value, step)

    dp_model.to(device)

    best_score = -float("inf")
    for epoch in range(num_epochs):

        print(f"Epoch={epoch}")

        loss = train(
            dp_model, train_loader, device, criterion, optimizer)
        log_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        log_scalar("loss", loss, epoch)

        val_loss, roc_auc = validate(
            dp_model, val_loader, device, criterion)
        log_scalar("val_loss", val_loss, epoch)
        log_scalar("roc_auc_", 1.0 - roc_auc, epoch)
        print(f"    ROC-AUC = {roc_auc:.4f}")

        if best_score < roc_auc:
            best_score = roc_auc
            checkpoint_file = Path(checkpoint_dir) / f"{trial_id}-{epoch:04d}.pth"
            torch.save(model.state_dict(), checkpoint_file)

        if isinstance(lr_scheduler, MultiStepLR):
            lr_scheduler.step()
        elif isinstance(lr_scheduler, ReduceLROnPlateau):
            lr_scheduler.step(val_loss)

    return best_score


def train(model, data_loader, device, criterion, optimizer, lr_scheduler):
    print("  Training")
    model.train()
    all_samples = 0
    loss_sum = 0
    progress = tqdm(data_loader)
    for i_batch, sample in enumerate(progress):
        x = sample["image"]
        y_true = sample["label"]
        num_samples = len(x)
        y_pred = model(x.to(device))
        loss = criterion(y_pred, y_true.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if isinstance(lr_scheduler, CosineAnnealingLR):
            lr_scheduler.step()
        all_samples += num_samples
        loss_sum += loss.cpu().item() * num_samples
        if i_batch == (len(data_loader) - 1):
            progress.set_description(f"    loss={loss_sum / all_samples:.6f}")
        else:
            progress.set_description(f"    loss={loss.cpu().item():.6f}")
    return loss_sum / all_samples


def validate(model, data_loader, device, criterion):
    print("  Validation")
    model.eval()
    all_samples = 0
    loss_sum = 0
    progress = tqdm(data_loader)
    y_true_all = []
    y_pred_all = []
    for i_batch, sample in enumerate(progress):
        x = sample["image"]
        y_true = sample["label"]
        num_samples = len(x)
        with torch.no_grad():
            y_pred = model(x.to(device))
            loss = criterion(y_pred, y_true.to(device))
        y_true_all.append(y_true.flatten())
        y_pred_all.append(y_pred.sigmoid().flatten().cpu())
        all_samples += num_samples
        loss_sum += loss.cpu().item() * num_samples
        if i_batch == (len(data_loader) - 1):
            progress.set_description(f"    loss={loss_sum / all_samples:.6f}")
        else:
            progress.set_description(f"    loss={loss.cpu().item():.6f}")
    roc_auc = roc_auc_score(
        torch.cat(y_true_all).numpy(),
        torch.cat(y_pred_all).numpy())
    return loss_sum / all_samples, roc_auc


if __name__ == "__main__":

    run(name="basic-binary-classification",
        seed=626, device_id=0,
        model_name="resnet18", pretrained=True, weights=None,
        image_dir="data/train", annot_file="data/train.csv",
        cv_splits=5, cv_fold=0, cv_split_seed=0,
        input_size=224, batch_size=32, num_epochs=64,
        aug_rot_limit=1.0,
        aug_gamma_limit=(90, 110),
        aug_brightness_limit=0.1,
        aug_contrast_limit=0.1,
        criterion_name="bce",
        criterion_config=dict(),
        optimizer_name="sgd",
        optimizer_config=dict(lr=1e-2, momentum=0.9, weight_decay=1e-4, nesterov=True),
        scheduler_name="plateau",
        scheduler_config=dict(),
        checkpoint_dir="checkpoint", tensorboard_dir="tensorboard")
