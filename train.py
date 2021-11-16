from os import stat
import nano
import torch
import wandb

wandb.init(project="nano", dir="./runs")

model = nano.models.yolox_esmk_shrink(num_classes=3)
# state_dict = torch.load("runs/train/exp127/weights/best.pt", map_location="cpu")["state_dict"]
# state_dict = {k:v for k, v in state_dict.items() if 'branch' not in k}
# model.load_state_dict(state_dict, strict=False)

trainer = nano.detection.trainer

trainer.run(
    model=model,
    data="configs/coc-s.yaml",
    hyp="configs/hyp.finetune.yaml",
    label_smoothing=0.1,
    batch_size=32,
    eval_batch_size=32,
    epochs=1000,
    imgsz=416,
    adam=True,
    logger=wandb,
)
