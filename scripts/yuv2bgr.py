import cv2
import numpy as np


def yuv2bgr(file_name, height, width, start_frame):
    """
    :param file_name: 待处理 YUV 视频的名字
    :param height: YUV 视频中图像的高
    :param width: YUV 视频中图像的宽
    :param start_frame: 起始帧
    :return: None
    """

    fp = open(file_name, 'rb')
    fp.seek(0, 2)  # 设置文件指针到文件流的尾部 + 偏移 0
    fp_end = fp.tell()  # 获取文件尾指针位置

    frame_size = height * width * 3 // 2  # 一帧图像所含的像素个数
    num_frame = fp_end // frame_size  # 计算 YUV 文件包含图像数
    print("This yuv file has {} frame imgs!".format(num_frame))
    fp.seek(frame_size * start_frame, 0)  # 设置文件指针到文件流的起始位置 + 偏移 frame_size * startframe
    print("Extract imgs start frame is {}!".format(start_frame + 1))

    for i in range(num_frame - start_frame):
        yyyy_uv = np.zeros(shape=frame_size, dtype='uint8', order='C')
        for j in range(frame_size):
            yyyy_uv[j] = ord(fp.read(1))  # 读取 YUV 数据，并转换为 unicode

        img = yyyy_uv.reshape((height * 3 // 2, width)).astype('uint8')  # NV12 的存储格式为：YYYY UV 分布在两个平面（其在内存中为 1 维）
        bgr_img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_NV12)  # 由于 opencv 不能直接读取 YUV 格式的文件, 所以要转换一下格式，支持的转换格式可参考资料 5
        cv2.imwrite('yuv2bgr/{}.jpg'.format(i + 1), bgr_img)  # 改变后缀即可实现不同格式图片的保存(jpg/bmp/png...)
        print("Extract frame {}".format(i + 1))

    fp.close()
    print("job done!")
    return None


# # BGR2NV21
# def bgr2nv21(image_path, image_height, image_width):
#     bgr = cv2.imread(image_path)
#     B = []
#     G = []
#     R = []

#     YUV = []
#     for i in range(image_height):
#         for j in range(image_width):
#             B.append(bgr[i * image_width * 3 + j * 3])
#             G.append(bgr[i * image_width * 3 + j * 3 + 1])
#             R.append(bgr[i * image_width * 3 + j * 3 + 2])

#             y = ((74 * R + 150 * G + 29 * B) / 256)
#             u = ((-43 * R - 84 * G + 112 * B) / 255) + 128
#             v = ((128 * R - 107 * G - 21 * B) / 255) + 128
#             YUV[i * image_width + j] = min(max(y, 0), 255)
#             YUV[(i // 2 + image_height) * image_width + (j // 2) * 2] = min(max(v, 0), 255)
#             YUV[(i // 2 + image_height) * image_width + (j // 2) * 2 + 1] = min(max(u, 0), 255)
#     yuv_name = image_path.replace("png", "yuv")
#     np.tofile(yuv_name)