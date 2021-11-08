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
        exit_code, output = self.container.exec_run(cmd=cmd, stream=stream)
        if stream is True:
            for data in output:
                print(data.decode(), end='')
            print()
        else:
            print(output.decode())
        print(f"-------- exit_code=<{exit_code}>")

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("=========== stop and removing container ..", end="")
        self.container.stop()
        self.container.remove()
        print(".done")


def docker_shell(image="xnnc-docker", root="xnnc/Example/cadenceNet"):
    return DockerShell(image, root)


def demo_cadencenet_test():
    with docker_shell("xnnc-docker", "/xnnc/Example/cadenceNet") as s:
        s.exec_run("python3 ../../Scripts/xnnc.py --config_file cadenceNet.cfg", stream=True)


if __name__ == "__main__":
    demo_cadencenet_test()
