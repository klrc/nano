import os
import shutil

root = '../datasets/VOC'
labels = os.path.join(root, 'labels/test2007')
images = os.path.join(root, 'images/test2007')

converted_root = '../datasets/VOC_val'
try:
    shutil.rmtree(converted_root)
    os.makedirs(os.path.join(converted_root, 'labels/voc_val'))
    os.makedirs(os.path.join(converted_root, 'images/voc_val'))
except FileExistsError:
    pass

raw_voc_labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                  'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

target_labels = ['person', 'bicycle', 'car', 'motorbike', 'bus']

label_map = {i: None if name not in target_labels else target_labels.index(name) for i, name in enumerate(raw_voc_labels)}
for label_file in os.listdir(labels):
    fid = label_file.split('.')[0]
    lines = []
    with open(os.path.join(labels, label_file), 'r') as f:
        for line in f.readlines():
            line = line.split(' ')
            converted_label = label_map[int(line[0])]
            if converted_label is not None:
                line[0] = str(converted_label)
                lines.append(' '.join(line))
    if len(lines) > 0:
        print(fid)
        with open(os.path.join(converted_root, 'labels/voc_val', label_file), 'w') as f:
            f.writelines(lines)
        with open(os.path.join(converted_root, 'voc_val.txt'), 'a') as f:
            f.write(f'./images/voc_val/{fid}.jpg\n')
        shutil.copy(f'{images}/{fid}.jpg', f'{converted_root}/images/voc_val/{fid}.jpg')
