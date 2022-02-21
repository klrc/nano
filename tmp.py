import os
import xml.etree.cElementTree as ET

root = "/Volumes/ASM236X/CAVIAR"
os.system(f"rm {root}/._*.xml")

xml_dict = {}
dir_dict = {}

for x in os.listdir(root):
    if x.endswith("xml"):
        try:
            tree = ET.parse(f"{root}/{x}")
            for frame in tree.getroot():
                frame_n = int(frame.attrib["number"]) + 1
            xml_dict[x] = frame_n
            print(x, frame_n)
        except Exception as e:
            print("parse failed", x)
            raise e
    elif os.path.isdir(f"{root}/{x}"):
        count = 0
        for y in os.listdir(f"{root}/{x}"):
            if y.endswith("jpg"):
                if ".ppm" in y:
                    os.system(f'mv {root}/{x}/{y} {root}/{x}/{y.replace(".ppm", "")}')
                count += 1
            if y.endswith("xml"):
                # os.system(f"mv {root}/{x}/{y} {root}/")
                print(x, y, "!!!!!!!!!!!")
                count = None
                break
        if count is not None:
            dir_dict[x] = count
            print(x, count)

print(xml_dict)
print(dir_dict)

for f_xml, c_xml in xml_dict.items():
    hit_times = 0
    hit_f_dir = None

    for f_dir, c_dir in dir_dict.items():
        if c_xml == c_dir:
            # if (f_xml.startswith('c') and 'cor' in f_dir) or (f_xml.startswith('f') and 'front' in f_dir):
            print(f_xml, f_dir, c_xml)
            hit_times += 1
            hit_f_dir = f_dir
    if hit_times == 1:
        os.system(f"mv {root}/{f_xml} {root}/{hit_f_dir}/")
    else:
        print(f'{f_xml}:{hit_times}')
