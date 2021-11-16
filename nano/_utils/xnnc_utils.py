import sys
import docker


class DockerShell:
    def __init__(self, image, root) -> None:
        self.image = image
        self.root = root

    def __enter__(self):
        client = docker.from_env()
        self.container = client.containers.run(
            image=self.image,
            command="/bin/sh",
            volumes={
                "/home/sh/Projects/klrc/nano/nano/_utils/xnnc": {
                    "bind": "/xnnc",
                    "mode": "rw",
                }
            },
            working_dir=self.root,
            auto_remove=False,
            stdin_open=True,
            detach=True,
        )
        return self

    def exec_run(self, cmd, stream=False):
        print('>', cmd)
        exit_code, output = self.container.exec_run(cmd=cmd, stream=stream)
        if stream is True:
            for data in output:
                print(data.decode(), end="")
            print()
        else:
            print(output.decode())
        if not stream:
            print(f"exit_code={exit_code}\n")

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("=========== stop and removing container ..", end="")
        self.container.stop()
        self.container.remove()
        print(".done")


def docker_shell(image="xnnc-docker", root="xnnc/Example/cadenceNet"):
    return DockerShell(image, root)


def demo_cadencenet_test():
    with docker_shell("xnnc-docker", "/xnnc/Example/cadenceNet") as s:
        s.exec_run("python3 ../../Scripts/xnnc.py --keep --config_file cadenceNet.cfg", stream=True)


def demo_resnet50_test():
    with docker_shell("xnnc-docker", "/xnnc/Example/Resnet50") as s:
        s.exec_run("python3 ../../Scripts/xnnc.py --config_file CF_ResNet_V1_50.cfg", stream=True)


def demo_fasterrcnn_test():
    pass
    # the following codes fail, considered as corrupted source.
    # with docker_shell("xnnc-docker", "/xnnc/Example/faster-R-CNN") as s:
    #     s.exec_run("make", stream=True)


def yolov5_test():
    ''' XNNC custom layer ----------
    layer {
        name: "detection_out"
        type: "CppCustom"
        bottom: "conv_blob72"
        bottom: "conv_blob73"
        bottom: "conv_blob74"
        top: "detection_out"
        cpp_custom_param {
            module: "XnncMobiYoloOutputLayer"
            param_map_str: "num_classes:81 share_location:1 background_label_id:0 nms_threshold:0.35 top_k:400 keep_top_k:200 confidence_threshold:0.5"
        }
    }
    layer {
        name: "_Resize_455"
        type: "CppCustom"
        bottom: "1084"
        top: "1089"
        cpp_custom_param {
            module: "mResize"
            param_map_str: "scaleX:2 scaleY:2 align_corners:1"
        }
    }
    '''
    with docker_shell("xnnc-docker:1.1", "/xnnc/Example/yolov5shufflenet") as s:
        s.exec_run("rm layers/MobYolo_output/CMakeCache.txt", stream=True)
        s.exec_run("rm -r layers/MobYolo_output/CMakeFiles", stream=True)
        s.exec_run("rm layers/MobYolo_output/Makefile", stream=True)
        s.exec_run("rm ./libXnncMobiYoloOutputLayer.so", stream=True)
        s.exec_run("rm layers/MobYolo_output/libXnncMobiYoloOutputLayer.so", stream=True)
        s.exec_run("rm layers/MobYolo_output/cmake_install.cmake", stream=True)
        s.exec_run("cmake layers/MobYolo_output/CMakeLists.txt", stream=True)
        s.exec_run("make -C layers/MobYolo_output", stream=True)
        s.exec_run('cp layers/MobYolo_output/libXnncMobiYoloOutputLayer.so ./', stream=True)
        s.exec_run("make install -C layers/MobYolo_output", stream=True)
        s.exec_run("python3 ../../Scripts/xnnc.py --config_file cfg/MobileYolo.cfg", stream=True)


def custom_layer_test():
    raise NotImplementedError
    # with docker_shell("xnnc-docker:1.1", "/xnnc/Example/yolov5shufflenet") as s:
        # s.exec_run("rm layers/MobYolo_output/CMakeCache.txt", stream=True)
        # s.exec_run("rm -r layers/MobYolo_output/CMakeFiles", stream=True)
        # s.exec_run("rm layers/MobYolo_output/Makefile", stream=True)
        # s.exec_run("rm ./libXnncMobiYoloOutputLayer.so", stream=True)
        # s.exec_run("rm layers/MobYolo_output/libXnncMobiYoloOutputLayer.so", stream=True)
        # s.exec_run("rm layers/MobYolo_output/cmake_install.cmake", stream=True)
        # s.exec_run("cmake layers/MobYolo_output/CMakeLists.txt", stream=True)
        # s.exec_run("make -C layers/MobYolo_output", stream=True)
        # s.exec_run('cp layers/MobYolo_output/libXnncMobiYoloOutputLayer.so ./', stream=True)
        # s.exec_run("make install -C layers/MobYolo_output", stream=True)
        # s.exec_run("python3 ../../Scripts/xnnc.py --config_file cfg/MobileYolo.cfg", stream=True)


if __name__ == "__main__":
    demo_cadencenet_test()
    # demo_resnet50_test()
    # yolov5_test()
    # custom_layer_test()
