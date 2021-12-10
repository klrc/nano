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
                "/home/sh/Projects/klrc/nano/xnnc/src": {
                    "bind": "/xnnc",
                    "mode": "rw",
                },
                "/home/sh/Projects/tensilica/xtensa/XtDevTools": {
                    "bind": "/home/sh/Projects/tensilica/xtensa/XtDevTools",
                    "mode": "rw",
                },
            },
            working_dir=self.root,
            auto_remove=False,
            stdin_open=True,
            detach=True,
        )
        return self

    def exec_run(self, cmd, stream=False):
        print(">", cmd)
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


def docker_shell(image="xnnc-docker:1.2", root="xnnc/Example/cadenceNet"):
    return DockerShell(image, root)


def demo_cadencenet_test():
    with docker_shell(root="/xnnc/Example/cadenceNet") as s:
        s.exec_run("python3 ../../Scripts/xnnc.py --keep --config_file cadenceNet.cfg", stream=True)


def demo_resnet50_test():
    with docker_shell(root="/xnnc/Example/Resnet50") as s:
        s.exec_run("python3 ../../Scripts/xnnc.py --config_file CF_ResNet_V1_50.cfg", stream=True)


def demo_fasterrcnn_test():
    raise NotImplementedError
    # the following codes fail, considered as corrupted source.
    with docker_shell(root="/xnnc/Example/faster-R-CNN") as s:
        s.exec_run("make", stream=True)


def yolov5_test():
    with docker_shell(root="/xnnc/Example/yolov5shufflenet") as s:
        s.exec_run("rm layers/MobYolo_output/CMakeCache.txt", stream=True)
        s.exec_run("rm -r layers/MobYolo_output/CMakeFiles", stream=True)
        s.exec_run("rm layers/MobYolo_output/Makefile", stream=True)
        s.exec_run("rm ./libXnncMobiYoloOutputLayer.so", stream=True)
        s.exec_run("rm layers/MobYolo_output/libXnncMobiYoloOutputLayer.so", stream=True)
        s.exec_run("rm layers/MobYolo_output/cmake_install.cmake", stream=True)
        s.exec_run("cmake layers/MobYolo_output/CMakeLists.txt", stream=True)
        s.exec_run("make -C layers/MobYolo_output", stream=True)
        s.exec_run("cp layers/MobYolo_output/libXnncMobiYoloOutputLayer.so ./", stream=True)
        s.exec_run("make install -C layers/MobYolo_output", stream=True)
        s.exec_run("python3 ../../Scripts/xnnc.py --config_file cfg/MobileYolo.cfg", stream=True)


def custom_layer_test():
    with docker_shell(root="/xnnc/Example/custom-cadenceNet") as s:
        for custom_layer in [
            "mish",
        ]:
            s.exec_run(f"rm layers/{custom_layer}/CMakeCache.txt", stream=True)
            s.exec_run(f"rm -r layers/{custom_layer}/CMakeFiles", stream=True)
            s.exec_run(f"rm layers/{custom_layer}/Makefile", stream=True)
            s.exec_run(f"rm layers/{custom_layer}/lib{custom_layer}.so", stream=True)
            s.exec_run(f"rm layers/{custom_layer}/cmake_install.cmake", stream=True)
            s.exec_run(f"rm layers/{custom_layer}/install_manifest.txt", stream=True)
            s.exec_run(f"rm ./lib{custom_layer}.so", stream=True)
            s.exec_run(f"cmake layers/{custom_layer}/CMakeLists.txt", stream=True)
            s.exec_run(f"make -C layers/{custom_layer}", stream=True)
            s.exec_run(f"cp layers/{custom_layer}/lib{custom_layer}.so ./", stream=True)
            s.exec_run(f"make install -C layers/{custom_layer}", stream=True)
        s.exec_run("python3 ../../Scripts/xnnc.py --keep --config_file cadenceNet.cfg", stream=True)


def yolox_test():
    with docker_shell(root="/xnnc/Example/yolox-series") as s:
        # for custom_layer in ["yoloxpp", "slice"]:
        #     s.exec_run(f"rm layers/{custom_layer}/CMakeCache.txt", stream=True)
        #     s.exec_run(f"rm -r layers/{custom_layer}/CMakeFiles", stream=True)
        #     s.exec_run(f"rm layers/{custom_layer}/Makefile", stream=True)
        #     s.exec_run(f"rm layers/{custom_layer}/lib{custom_layer}.so", stream=True)
        #     s.exec_run(f"rm layers/{custom_layer}/cmake_install.cmake", stream=True)
        #     s.exec_run(f"rm layers/{custom_layer}/install_manifest.txt", stream=True)
        #     s.exec_run(f"rm ./lib{custom_layer}.so", stream=True)
        #     s.exec_run(f"cmake layers/{custom_layer}/CMakeLists.txt", stream=True)
        #     s.exec_run(f"make -C layers/{custom_layer}", stream=True)
        #     s.exec_run(f"cp layers/{custom_layer}/lib{custom_layer}.so ./", stream=True)
        #     s.exec_run(f"make install -C layers/{custom_layer}", stream=True)
        # s.exec_run("python3 ../../Scripts/xnnc.py --keep --stage gen_quant_profile --config_file yolox.cfg", stream=True)
        # s.exec_run("python3 ../../Scripts/xnnc.py --keep --stage use_quant_profile --config_file yolox.cfg", stream=True)
        s.exec_run("python3 ../../Scripts/xnnc.py --keep --stage gen_code --config_file yolox.cfg", stream=True)
        # s.exec_run("python3 ../../Scripts/xnnc.py --keep --stage gen_graph_bin --config_file yolox.cfg", stream=True)


if __name__ == "__main__":
    # demo_cadencenet_test()
    # demo_resnet50_test()
    # yolov5_test()
    # custom_layer_test()
    yolox_test()
