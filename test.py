from object_detection.training_utils.default_settings import DefaultSettings
from object_detection.yolov5_ultralytics import yolov5n, yolov5s
from test_utils.detection_test import camera_test, travel_dataset, video_test
from test_utils.video_loader import H264_LOADER


names = DefaultSettings.names
names.append('gun')
names.append('hand')
names.append('toy')


travel_dataset('/Volumes/ASM236X/IndoorCVPR_09/images', class_names=names)


# model = yolov5s(82, weights='runs/yolov5s.3/best.pt').eval()
# # model = yolov5n(82, weights="runs/yolov5n.7/best.pt").eval()
# # # model = yolov5n(80, weights="runs/yolov5n.7/fuse.pt").eval()

# # camera_test(model, class_names=names, inf_size=480)

# if __name__ == "__main__":
#     # video_source = '../datasets/v1.5.6.5人形测试/12M左右不报.avi'
#     for video_source in (
#         "../datasets/6630-V1.5.7.0误报&漏报视频2000613/误报/hand.h264",
#         "../datasets/6630-V1.5.7.0误报&漏报视频2000613/误报/hand1.h264",
#         "../datasets/6630-V1.5.7.0误报&漏报视频2000613/误报/red.h264",
#         "../datasets/6630-V1.5.7.0误报&漏报视频2000613/误报/umbrella.h264",
#         "../datasets/6630-V1.5.7.0误报&漏报视频2000613/误报/UVC.h264",
#     ):
#         video_test(model, H264_LOADER(video_source, (480, 384)), names, fps=8, inf_size=480)
