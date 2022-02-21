# import torch
# from tests import test_inference, test_front_camera, test_yuv, test_model_zoo, test_yuv_activation
# from nano.models.model_zoo import GhostNano_3x4_m96
# from nano.data.dataset_info import drive3_names
# from scripts.test_dataloader import test_dataloader
from nano.data.dataset import CAVIARSeed
from nano.data.visualize import RenderLabels
import cv2

if __name__ == "__main__":
    # test_model_zoo()
    # test_dataloader()
    seed = CAVIARSeed("/Volumes/ASM236X/CAVIAR", pick_rate=0.05)
    print(len(seed))

# if __name__ == "__main__":
#     model = GhostNano_3x4_m96(3)
#     model.load_state_dict(torch.load("release/GhostNano_3x4_m96/GhostNano_3x4_m96.pt", map_location="cpu"))

#     # test_front_camera(model, 0.1, 0.45, drive3_names, "cpu")
#     test_yuv_activation(model, 0.2, 0.45, drive3_names, yuv_file="../datasets/1280x720_3.yuv", device="cpu")


# if __name__ == "__main__":
#     from nano.data.dataset import detection_data_layer, Assembly
#     import nano.data.transforms as T
#     from nano.data.dataset_info import coco_to_drive3, voc_to_drive3
#     import numpy as np
#     from tqdm import tqdm

#     trainset1 = detection_data_layer("/Volumes/ASM236X/coco/train2017", "/Volumes/ASM236X/coco/labels/train2017")
#     trainset2 = detection_data_layer("/Volumes/ASM236X/coco/val2017", "/Volumes/ASM236X/coco/labels/val2017")
#     trainset3 = detection_data_layer("../datasets/VOC/images/train2012", "../datasets/VOC/labels/train2012")
#     valset = detection_data_layer("../datasets/VOC/images/val2012", "../datasets/VOC/labels/val2012")
#     factory = Assembly()
#     factory.append_data(trainset1, trainset2, trainset3, valset)
#     factory.compose(
#         T.ToNumpy(mark_source=True),
#         T.IndexMapping(coco_to_drive3, pattern="/coco"),
#         T.IndexMapping(voc_to_drive3, pattern="/VOC"),
#         T.Resize(max_size=512),
#         T.ToTensor(),
#     )

#     valloader = factory.as_dataloader(batch_size=1, num_workers=4, collate_fn=T.letterbox_collate_fn)
#     sizes = {'min': [], 'max': []}
#     bar = tqdm(valloader)
#     for _, labels in bar:
#         for x in labels:
#             if x[1] == 0:
#                 sizes['min'].append(min(x[4]-x[2], x[5]-x[3]))
#                 sizes['max'].append(max(x[4]-x[2], x[5]-x[3]))
#         desc = ""
#         for k, v in sizes.items():
#             desc += f'{k}: {np.mean(v):.4f}'
#         bar.set_description(desc=desc)
