import os
import cv2


def dfs(root):
    for file in os.listdir(root):
        fp = f"{root}/{file}"
        if os.path.isdir(fp):
            for x in dfs(fp):
                yield x
        else:
            yield fp


def screenshot(avi_path, size=None):
    cap = cv2.VideoCapture(avi_path)
    interval = 32
    count = 0
    i = 0
    while True:
        success, data = cap.read()
        if not success:
            break
        if size is not None:
            data = cv2.resize(data, size)
        count += 1
        if count >= interval:
            cv2.imshow("test", data)
            # cv2.imwrite(f"{output_root}/{seq_name}_{i}.png", data)
            i += 1
            count = 0
        # Press ESC to break
        if cv2.waitKey(10) == 27:
            break
    cap.release()


dataset_root = "../datasets/FH_test_walking_day_night"
output_root = "../datasets/FH_test_walking_day_night/test_frames"

if not os.path.exists(output_root):
    os.makedirs(output_root)

for fp in dfs(dataset_root):
    fname = fp.split("/")[-1]
    if fname.endswith(".avi") and not fname.startswith("._"):
        seq_name = fname.split("_avi")[0]
        screenshot(fp, size=(960, 640))
        # for t, img in screenshot(fp):
        #     print(t, img.shape)
        # cv2.imwrite(f"{output_root}/{seq_name}_{t}.png", img)
