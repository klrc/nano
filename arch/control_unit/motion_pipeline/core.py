import cv2


def camera_test(functions, size=320, titles=None):
    cam = cv2.VideoCapture(0)
    cam_width, cam_height = cam.get(cv2.CAP_PROP_FRAME_WIDTH), cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    scale = max(cam_width, cam_height) / size
    target_size = (int(cam_width / scale), int(cam_height / scale))
    while True:
        _, frame = cam.read()
        frame = cv2.resize(frame, target_size)
        cv2.imshow("camera", frame)
        cv2.setWindowProperty("camera", cv2.WND_PROP_TOPMOST, 1)
        for func in functions:
            cv2.imshow("demo", cv2.resize(func(frame), target_size, interpolation=cv2.INTER_NEAREST))
            cv2.setWindowProperty("demo", cv2.WND_PROP_TOPMOST, 1)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break


def video_test(loader, dsize=(320, 180), fps=24, func=None):
    from multiprocessing import Queue

    pipe = Queue(maxsize=2)
    loader.play(pipe, fps=fps, dsize=dsize)

    frame = pipe.get()
    while frame is not None:
        cv2.imshow("camera", frame)
        cv2.setWindowProperty("camera", cv2.WND_PROP_TOPMOST, 1)
        if func:
            cv2.imshow("demo", cv2.resize(func(frame), dsize, interpolation=cv2.INTER_NEAREST))
            cv2.setWindowProperty("demo", cv2.WND_PROP_TOPMOST, 1)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break
        frame = pipe.get()
