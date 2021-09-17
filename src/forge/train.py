import torch

def train(model, options):
    w = project_path(options)
    model, start_epoch = try_load_ckpt(model, options)
    optimizer = create_optimizer(model, options)
    train_loader, val_loader = create_dataloader(options)
    max_epoch = 0  # get max_epoch from options
    for epoch in range(start_epoch, max_epoch):
        model.train()
        optimizer.zero_grad()
        pbar = enumerate(train_loader)
        for i, (imgs, targets, paths, _) in pbar:
           loss_items = train_batch(model, optimizer, imgs, targets, paths) 
        results = val(model, val_loader)
        on_train_val_end()
    torch.cuda.empty_cache()
    return