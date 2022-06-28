import torch

if __name__ == "__main__":
    from arch.control_unit.detection_pipeline import full_dataset_test, coco_class_names
    from arch.model.object_detection.yolov5 import yolov5n

    model = yolov5n(80, weights=".yolov5_checkpoints/yolov5n_sd.pt", activation=torch.nn.SiLU).eval()
    # model = yolov5n(80, weights="runs/yolov5n.1/best.pt").eval()

    # Full dataset test
    full_dataset_test(model, "../datasets/FH_test_walking_day_night/test_frames/images", class_names=coco_class_names)