coco_names = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)  # class names


voc_names = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)  # class names


drive3_names = (
    "person",
    "bike",
    "car",
)

any_to_drive3 = {
    "person": "person",
    "bicycle": "bike",
    "motorcycle": "bike",
    "motorbike": "bike",
    "car": "car",
    "bus": "car",
    "truck": "car",
}


def __index_mapping(src_names, target_names, src_to_target_names):
    index_map = {}
    for k, v in src_to_target_names.items():
        if k in src_names:
            src_cid = src_names.index(k)
            target_cid = target_names.index(v)
            index_map[src_cid] = target_cid
    return index_map


voc_to_drive3 = __index_mapping(voc_names, drive3_names, any_to_drive3)
coco_to_drive3 = __index_mapping(coco_names, drive3_names, any_to_drive3)
