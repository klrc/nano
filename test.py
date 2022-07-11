import torch


if __name__ == "__main__":
    from arch.control_unit.detection_pipeline import full_dataset_test, coco_class_names
    from arch.control_unit.detection_heatmap_visualizer.core import Probe
    from arch.model.object_detection.yolov5 import yolov5n

    # model = yolov5n(80, weights=".yolov5_checkpoints/yolov5n_sd.pt", activation=torch.nn.SiLU).eval()
    model = yolov5n(6, weights="runs/yolov5n.12/best.pt", activation=torch.nn.SiLU).eval()

    # model = yolov5n(80, weights=".yolov5_checkpoints/yolov5n_sd.pt").eval()
    # torch.onnx.export(model.fuse(), torch.rand(1, 3, 384, 640), "tmp.onnx", verbose=False, opset_version=12)

    probe = Probe(model, forward_type='yolov5')

    # Full dataset test
    full_dataset_test(probe, "../datasets/办公室椅子测试", class_names=coco_class_names)
