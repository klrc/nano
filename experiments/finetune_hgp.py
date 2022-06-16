from object_detection.training_utils.default_settings import DefaultSettings
from object_detection.training_utils.trainer import train
from object_detection.yolov5_ultralytics import yolov5n, yolov5s

if __name__ == "__main__":
    # process dataset
    import os

    label_mapper = {
        0: DefaultSettings.names.index("cell phone"),  # phone
        1: 80,  # gun
        2: 81,  # hand
    }
    output_root = "../datasets/HGP/labels"
    labels_root = "../datasets/HGP/labels_raw"
    if not os.path.exists(labels_root):
        os.system(f"mv {output_root} {labels_root}")
    else:
        os.system(f"rm -r {output_root}")
    for dataset in ("train2017", "val2017"):
        image_dir = f"{labels_root}/{dataset}"
        output_dir = f"{output_root}/{dataset}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # traverse through images
        print(image_dir)
        assert os.path.exists(image_dir)
        for file_name in os.listdir(image_dir):
            if not file_name.startswith("._") and file_name.endswith(".txt"):
                labels = []
                with open(f"{image_dir}/{file_name}", "r") as f:
                    for line in f.readlines():
                        c, x, y, w, h = [float(m) for m in line.split(" ")]
                        c = label_mapper[int(c)]
                        line = f"{c} {x} {y} {w} {h}"
                        labels.append(line)

                with open(f"{output_dir}/{file_name}", "w") as f:
                    for line in labels:
                        f.write(f"{line}\n")

    # fine-tune (more data & strong augmentation)
    settings = DefaultSettings()
    settings.trainset_path = [
        "../datasets/coco/train2017.txt",
        "../datasets/HGP/images/train2017",
    ]
    settings.valset_path = ["../datasets/coco/val2017.txt"]
    settings.lr0 = 0.003
    settings.batch_size = 2
    settings.momentum = 0.75
    settings.weight_decay = 0.00025
    settings.lrf = 0.15
    settings.names.append("gun")
    settings.names.append("hand")
    settings.nc = 82
    settings.save_dir = "runs/yolov5n"
    model = yolov5n(num_classes=82, weights="runs/yolov5n.1/best.pt")
    train(model, settings, device="0")

    settings = DefaultSettings()
    settings.trainset_path = [
        "../datasets/coco/train2017.txt",
        "../datasets/HGP/images/train2017",
    ]
    settings.valset_path = ["../datasets/coco/val2017.txt"]
    settings.lr0 = 0.003
    settings.momentum = 0.75
    settings.weight_decay = 0.00025
    settings.lrf = 0.15
    settings.names.append("gun")
    settings.names.append("hand")
    settings.nc = 82
    settings.save_dir = "runs/yolov5s"
    model = yolov5s(num_classes=82, weights="runs/yolov5s.2/best.pt")
    train(model, settings, device="0")
