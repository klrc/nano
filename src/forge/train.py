import torch

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

def main(opt):
    device = select_device(opt)
    model = create_model(opt, device)
    return train(model, opt, device)


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    return main(opt)