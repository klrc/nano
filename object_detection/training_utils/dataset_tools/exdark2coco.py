import os
import cv2


def convert_ExDark(root_images):
    for cls in os.listdir(root_images):
        if os.path.isdir(f"{root_images}/{cls}") and not cls.startswith("._"):
            for file_name in os.listdir(f"{root_images}/{cls}"):
                if not file_name.startswith("._"):
                    image_path = f"{root_images}/{cls}/{file_name}"
                    os.system(f'mv {image_path} {root_images}/{file_name.replace(".JPEG",".jpg").replace(".JPG",".jpg").replace(".png",".jpg")}')
                    print(f"processing {image_path}")
            os.system(f"rm -d {root_images}/{cls}")


def convert_ExDark_Anno(root_annotations, class_names):
    for cls in os.listdir(root_annotations):
        if os.path.isdir(f"{root_annotations}/{cls}") and not cls.startswith("._"):
            for file_name in os.listdir(f"{root_annotations}/{cls}"):
                if file_name.startswith("._"):
                    continue
                anno_path = f"{root_annotations}/{cls}/{file_name}"
                # get image size
                image_fname = f'{file_name.replace(".txt", "").replace(".JPEG",".jpg").replace(".JPG",".jpg").replace(".png",".jpg")}'
                image_path = f"{root_images}/{image_fname}"
                if os.path.exists(image_path):
                    image = cv2.imread(image_path)
                    height, width, _ = image.shape
                    with open(anno_path, "r") as f:
                        lines = []
                        for line in f.readlines():
                            if "%" in line:
                                continue
                            c, tx, ly, w, h = line.split(" ")[:5]
                            c = class_names.index(c.lower())
                            tx, ly, w, h = [float(x) for x in (tx, ly, w, h)]
                            tx = (tx + 0.5 * w) / width
                            ly = (ly + 0.5 * h) / height
                            w = (w) / width
                            h = (h) / height
                            line = [c, tx, ly, w, h]
                            lines.append(" ".join([str(x) for x in line]) + "\n")
                    with open(f'{root_annotations}/{image_fname.replace(".jpg", ".txt")}', "w") as f:
                        f.writelines(lines)
                    print(f"processing {file_name} ({len(lines)} objects)")
                else:
                    print(f"missing {image_path}")
            os.system(f"rm -rf {root_annotations}/{cls}")


if __name__ == "__main__":
    # paths
    root = "/Volumes/ASM236X/ExDark"
    root_images = root + "/images"
    root_annotations = root + "/ExDark_Anno"
    coco_class_names = "people|bicycle|car|motorbike|airplane|bus|train|truck|boat|traffic light|fire hydrant|stop sign|parking meter|bench|bird|cat|dog|horse|sheep|cow|elephant|bear|zebra|giraffe|backpack|umbrella|handbag|tie|suitcase|frisbee|skis|snowboard|sports ball|kite|baseball bat|baseball glove|skateboard|surfboard|tennis racket|bottle|wine glass|cup|fork|knife|spoon|bowl|banana|apple|sandwich|orange|broccoli|carrot|hot dog|pizza|donut|cake|chair|couch|potted plant|bed|table|toilet|tv|laptop|mouse|remote|keyboard|cell phone|microwave|oven|toaster|sink|refrigerator|book|clock|vase|scissors|teddy bear|hair drier|toothbrush".split(  # noqa:E501
        "|"
    )  # class names

    # convert_ExDark(root_images)
    convert_ExDark_Anno(root_annotations, coco_class_names)
