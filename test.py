
if __name__ == "__main__":
    from arch.control_unit.detection_dataset_traveler import travel_dataset
    from arch.control_unit.detection_pipeline import coco_class_names

    coco_class_names.append("gun")
    coco_class_names.append("human hand")
    coco_class_names.append("toy")

    travel_dataset("/Volumes/ASM236X/IndoorCVPR_09/images", coco_class_names)
