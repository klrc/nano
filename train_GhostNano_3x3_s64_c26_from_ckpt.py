import torch
from nano.datasets.dataset_info import voc_classes, c26_classes, voc_to_c26
from nano.datasets.live_data import *
import nano.datasets.transforms as T
from nano.training_core import training_layer, validation_layer, log_layer, load_device
from nano.models.model_zoo import GhostNano_3x3_s64
from nano.models.assigners.simota import SimOTA

if __name__ == "__main__":
    model = GhostNano_3x3_s64(num_classes=26)
    model.load_state_dict(torch.load("runs/train/exp9/last.pt", map_location="cpu")["state_dict"])
    criteria = SimOTA(num_classes=len(c26_classes))
    device = load_device("cpu")

    dataset1 = detection_data_layer("../datasets/coco/images/train2017", "../datasets/coco/labels/train2017")
    dataset2 = detection_data_layer("../datasets/VOC/images/val2012", "../datasets/VOC/labels/val2012")
    layer = infinite_sampler(dataset1, dataset2, shuffle=True)
    layer = load_data_as_numpy(layer)
    layer = class_mapping_layer(layer, voc_classes, c26_classes, voc_to_c26)
    layer = transform_layer(layer, T.horizontal_flip, p=0.5)
    layer = transform_layer(layer, T.resize, min_size=416)
    layer = transform_layer(layer, T.random_scale, min_scale=0.5, max_scale=1)
    layer = transform_layer(layer, T.random_affine, p=0.2, min_scale=1, max_scale=1.5)
    layer = transform_layer(layer, T.hsv_transform)
    layer = albumentations_transform_layer(layer)
    layer = transform_layer(layer, T.mosaic4, feed_samples=4, mosaic_size=416)
    layer = to_tensor(layer)
    trainloader = collate_dataloader(layer, 16, letterbox_collate_fn)
    dataset = detection_data_layer("../datasets/coco/images/val2017", "../datasets/coco/labels/val2017")
    layer = load_data_as_numpy(dataset)
    layer = class_mapping_layer(layer, voc_classes, c26_classes, voc_to_c26)
    layer = transform_layer(layer, T.resize, max_size=416)
    layer = to_tensor(layer)
    valloader = collate_dataloader(layer, 16, letterbox_collate_fn)
    trainer = training_layer(trainloader, model, criteria, device, batch_size=128)
    validator = validation_layer(trainer, valloader, c26_classes, device)
    logger = log_layer(validator, patience=16)

    # Ah shit, here we go again
    for _ in logger:
        pass
