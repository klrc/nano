
from _utils.loggers import Loggers
from _utils.metrics import fitness
from _utils.loggers.wandb.wandb_utils import check_wandb_resume
from _utils.torch_utils import select_device, de_parallel
from _utils.plots import plot_labels
from _utils.general import labels_to_class_weights, increment_path, init_seeds, \
    get_latest_run, check_dataset, check_file, check_img_size, set_logging, one_cycle, colorstr
from _utils.datasets import create_dataloader
from _utils.autoanchor import check_anchors
from _utils.melt4 import shrink

import torch
from pathlib import Path


def project_path(options):
    # Directories
    save_dir = Path(options.save_dir)
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    return w


def train(model, options, device):
    w = project_path(options)
    model, start_epoch = try_load_ckpt(model, options, device)
    optimizer = create_optimizer(model, options)
    train_loader, val_loader = create_dataloader(options)
    max_epoch = 0  # get max_epoch from options
    for epoch in range(start_epoch, max_epoch):
        model.train()
        optimizer.zero_grad()
        pbar = enumerate(train_loader)
        for i, (imgs, targets, paths, _) in pbar:
           loss_items = train_batch(model, optimizer, imgs, targets, paths) 
        results = val.run(model, val_loader)
        on_train_val_end()
    torch.cuda.empty_cache()
    return
