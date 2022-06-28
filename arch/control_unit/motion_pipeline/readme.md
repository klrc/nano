## Get Started
```python
if __name__ == "__main__":
    from arch.control_unit.motion_pipeline import video_test
    from arch.io.video_loader import YUV400_VID_LOADER, H264_LOADER
    from arch.model.motion_detection.low_cost_md.demo_simple_diff import MDASimpleDiff

    # camera_test(func=MotionDetection((180, 320)))

    for test_sequence in (
        YUV400_VID_LOADER("../datasets/filtered.yuv", (1080, 1920)),
        H264_LOADER("../datasets/FH_test_walking_day_night/day/1_01_R_20220304201712_avi.avi"),
        H264_LOADER("../datasets/FH_test_walking_day_night/night/1_01_A_20220310171644_avi.avi"),
        H264_LOADER("../datasets/mda_test/test_sequence_motion_1.mp4"),
        H264_LOADER("../datasets/mda_test/test_sequence_motion_2.mp4"),
    ):
        video_test(test_sequence, dsize=(320, 180), func=MDASimpleDiff((180, 320)))
```