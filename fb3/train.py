from config import CFG
import gc
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from utils import AverageMeter, timeSince, get_score
from transformers import AutoTokenizer
from dataset import collate, TrainDataset
from model import CustomModel
import numpy as np
from torch.optim import AdamW
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
import time

# ====================================================
# train loop
# ====================================================
def train_loop(folds, fold):

    CFG.logger.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds[folds["fold"] != fold].reset_index(drop=True)
    valid_folds = folds[folds["fold"] == fold].reset_index(drop=True)
    valid_labels = valid_folds[CFG.target_cols].values
    # ====================================================
    # tokenizer
    # ====================================================
    tokenizer = AutoTokenizer.from_pretrained(CFG.model)
    tokenizer.save_pretrained(CFG.output_dir + "tokenizer/")

    train_dataset = TrainDataset(CFG, train_folds, tokenizer)
    valid_dataset = TrainDataset(CFG, valid_folds, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size * 2,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG, config_path=None, pretrained=True)
    torch.save(model.config, CFG.output_dir + "config.pth")
    model.to(CFG.device)

    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p
                    for n, p in model.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "lr": encoder_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "lr": encoder_lr,
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if "model" not in n],
                "lr": decoder_lr,
                "weight_decay": 0.0,
            },
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(
        model,
        encoder_lr=CFG.encoder_lr,
        decoder_lr=CFG.decoder_lr,
        weight_decay=CFG.weight_decay,
    )
    optimizer = AdamW(
        optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas
    )

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=cfg.num_warmup_steps,
                num_training_steps=num_train_steps,
            )
        elif cfg.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=cfg.num_warmup_steps,
                num_training_steps=num_train_steps,
                num_cycles=cfg.num_cycles,
            )
        return scheduler

    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.SmoothL1Loss(reduction="mean")  # RMSELoss(reduction="mean")

    best_score = np.inf

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(
            fold, train_loader, model, criterion, optimizer, epoch, scheduler
        )

        # eval
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion)

        # scoring
        score, scores = get_score(valid_labels, predictions)

        elapsed = time.time() - start_time

        CFG.logger.info(
            f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
        )
        CFG.logger.info(f"Epoch {epoch+1} - Score: {score:.4f}  Scores: {scores}")
        if CFG.wandb:
            wandb.log(
                {
                    f"[fold{fold}] epoch": epoch + 1,
                    f"[fold{fold}] avg_train_loss": avg_loss,
                    f"[fold{fold}] avg_val_loss": avg_val_loss,
                    f"[fold{fold}] score": score,
                }
            )

        if best_score > score:
            best_score = score
            CFG.logger.info(
                f"Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model"
            )
            torch.save(
                {"model": model.state_dict(), "predictions": predictions},
                CFG.output_dir + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
            )

    predictions = torch.load(
        CFG.output_dir + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
        map_location=torch.device("cpu"),
    )["predictions"]
    valid_folds[[f"pred_{c}" for c in CFG.target_cols]] = predictions

    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds


def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, (inputs, labels) in enumerate(train_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(CFG.device)
        labels = labels.to(CFG.device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CFG.max_grad_norm
        )
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                "Epoch: [{0}][{1}/{2}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                "Grad: {grad_norm:.4f}  "
                "LR: {lr:.8f}  ".format(
                    epoch + 1,
                    step,
                    len(train_loader),
                    remain=timeSince(start, float(step + 1) / len(train_loader)),
                    loss=losses,
                    grad_norm=grad_norm,
                    lr=scheduler.get_lr()[0],
                )
            )
        if CFG.wandb:
            wandb.log(
                {
                    f"[fold{fold}] loss": losses.val,
                    f"[fold{fold}] lr": scheduler.get_lr()[0],
                }
            )
    return losses.avg


def valid_fn(valid_loader, model, criterion):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (inputs, labels) in enumerate(valid_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(CFG.device)
        labels = labels.to(CFG.device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.to("cpu").numpy())
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
            print(
                "EVAL: [{0}/{1}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) ".format(
                    step,
                    len(valid_loader),
                    loss=losses,
                    remain=timeSince(start, float(step + 1) / len(valid_loader)),
                )
            )
    predictions = np.concatenate(preds)
    return losses.avg, predictions
