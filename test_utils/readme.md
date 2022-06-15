## Object Detection Test Tools
### full_dataset_test
    full_dataset_test(model, path, class_names, inf_size)
- model: `torch.nn.Module` detection model
- path: dataset images dir (auto detect labels dir)
- class_names: dataset class names
- inf_size: max image size

![](docs/Jun-15-2022%2013-59-38.gif)

### video_test
    video_test(model, loader: VideoLoader, class_names, fps, inf_size)
- model: `torch.nn.Module` detection model
- loader: `video_loader.VideoLoader` implementation (fake video stream with fixed fps)
    
    e.g.:

        # loader = YUV420_VID_LOADER("/Users/sh/Downloads/1280x720_2.yuv", (720, 1280))

- class_names: dataset class names
- fps: frame rate
- inf_size: max image size

![](docs/Jun-15-2022%2014-03-11.gif)

### camera_test
    camera_test(model, class_names, inf_size)
- model: `torch.nn.Module` detection model
- class_names: dataset class names
- inf_size: max image size

![](docs/Jun-15-2022%2014-06-03.gif)