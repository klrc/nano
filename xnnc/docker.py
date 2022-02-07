import docker


class DockerShell:
    def __init__(self, image, root, volumes={}) -> None:
        self.image = image
        self.root = root
        self.volumes = volumes

    def __enter__(self):
        client = docker.from_env()
        self.container = client.containers.run(
            image=self.image,
            command="/bin/sh",
            volumes=self.volumes,
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


def caffe_builder(
    working_dir,
    build_path,
    script_path,
    image="caffe-python39-cpu",
):
    return DockerShell(
        image,
        working_dir,
        volumes={
            build_path: {
                "bind": build_path,
                "mode": "rw",
            },
            script_path: {
                "bind": '/script',
                "mode": "rw",
            },
        },
    )


def xnnc_builder(
    working_dir,
    xnnc_path,
    xtdev_path,
    image="xnnc-docker:1.2",
):
    """
    Xtensa NN Compiler docker builder,
    Usage:
        with xnnc_builder(root="/xnnc/Example/cadenceNet") as s:
            s.exec_run("python3 ../../Scripts/xnnc.py --keep --config_file cadenceNet.cfg", stream=True)
    """
    return DockerShell(
        image,
        working_dir,
        volumes={
            xnnc_path: {
                "bind": "/xnnc",
                "mode": "rw",
            },
            xtdev_path: {
                "bind": "/home/sh/Projects/tensilica/xtensa/XtDevTools",
                "mode": "rw",
            },
        },
    )
