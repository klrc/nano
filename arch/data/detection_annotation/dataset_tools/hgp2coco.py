import os


label_root = "/Volumes/ASM236X/HGP/labels"
dataset_dirs = ("train2017", "val2017")
backup_suffix = "_raw"
supply_suffix = "_supply"
backup_root = label_root + backup_suffix
supply_root = label_root + supply_suffix

# class_names=("person", "car", "phone", "hand")
# raw = phone, gun, hand
raw_mapper = {
    0: 2,  # mscoco 'phone' class index
    2: 3,  # additional index for 'hand'
}
# supply = person, car
supply_mapper = {
    0: 0,  # mscoco 'person' class index
    1: 1,  # mscoco 'car' class index
}

# raw backup
if os.path.exists(backup_root):
    os.system(f"rm -r {label_root}")
else:
    os.system(f"mv {label_root} {backup_root}")

# map label classses
for dataset in dataset_dirs:
    source_dir = f"{backup_root}/{dataset}"
    supply_dir = f"{supply_root}/{dataset}"
    output_dir = f"{label_root}/{dataset}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # traverse through images
    print(source_dir)
    assert os.path.exists(source_dir)
    image_dir = source_dir.replace("/labels/", "/images/")
    for file_name in os.listdir(image_dir):
        file_name = file_name.replace(".png", ".txt")
        if not file_name.startswith("._") and file_name.endswith(".txt"):
            labels = []
            # read raw labels
            if os.path.exists(f"{source_dir}/{file_name}"):
                with open(f"{source_dir}/{file_name}", "r") as f:
                    for line in f.readlines():
                        c, x, y, w, h = [float(m) for m in line.split(" ")]
                        c = int(c)
                        if c in raw_mapper:
                            c = raw_mapper[c]
                            line = f"{c} {x} {y} {w} {h}"
                            labels.append(line)
            # read supply labels
            if os.path.exists(f"{supply_dir}/{file_name}"):
                with open(f"{supply_dir}/{file_name}", "r") as f:
                    for line in f.readlines():
                        c, x, y, w, h = [float(m) for m in line.split(" ")]
                        c = int(c)
                        if c in supply_mapper:
                            c = supply_mapper[c]
                            line = f"{c} {x} {y} {w} {h}"
                            labels.append(line)
            with open(f"{output_dir}/{file_name}", "w") as f:
                for line in labels:
                    f.write(f"{line}\n")
