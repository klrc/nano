import torch
from nano.tests import test_inference, test_front_camera, test_yuv, test_model_zoo
from nano.models.model_zoo import GhostNano_3x4_m96
from nano.data.dataset_info import drive3_names

if __name__ == "__main__":
    model = GhostNano_3x4_m96(3)
    model.load_state_dict(torch.load("runs/train/exp17/best.pt", map_location="cpu")["state_dict"])

    # test_front_camera(model, 0.25, 0.45, drive3_names, "cpu")
    test_yuv(model, 0.25, 0.45, drive3_names, yuv_file="../datasets/1280x720_5.yuv", device="cpu")
