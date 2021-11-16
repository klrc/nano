import nano
import torch
import wandb

wandb.init(project="nano", dir="./runs")

model = nano.models.yolox_esmk_shrink_misc(num_classes=4)
state_dict = torch.load("runs/train/exp128/weights/last.pt", map_location="cpu")["state_dict"]
model.load_state_dict(state_dict, strict=False)
model.head.misc_bias = torch.nn.Parameter(torch.ones(1, 1, 1, 1) * 4, requires_grad=True)

trainer = nano.detection.trainer

trainer.run(
    model=model,
    data="configs/coc-misc-s.yaml",
    hyp="configs/hyp.finetune.yaml",
    label_smoothing=0.2,
    batch_size=32,
    eval_batch_size=32,
    epochs=1000,
    imgsz=416,
    logger=wandb,
)
