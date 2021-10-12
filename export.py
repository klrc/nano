import torch
import os


def export_onnx(model, dummy_input, f):
    # ONNX model export
    prefix = 'ONNX:'
    try:
        import onnx
        print(f'\n{prefix} starting export with onnx {onnx.__version__}...')

        torch.onnx.export(
            model,
            dummy_input,
            f,
            verbose=True,
            keep_initializers_as_inputs=False,
            opset_version=11,
            input_names=['images'],
            output_names=['output'],
        )
        # torch.onnx.export(model, img, f, verbose=False, opset_version=opset,
        #                   training=torch.onnx.TrainingMode.EVAL,
        #                   do_constant_folding=True,
        #                   input_names=['images'],
        #                   output_names=['output'])

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # print(onnx.helper.printable_graph(model_onnx.graph))  # print

        print(f'{prefix} export success, saved as {f}')
        print(f"{prefix} run --dynamic ONNX model inference with detect.py: 'python detect.py --weights {f}'")
    except Exception as e:
        print(f'{prefix} export failure: {e}')


def run(model, output_path):
    model = model.to('cpu')
    img = torch.zeros(1, 3, 224, 416).to('cpu')

    for _ in range(2):
        model(img)  # dry runs

    export_onnx(model, img, output_path)


if __name__ == '__main__':

    device = 'cpu'

    # Model
    from nano.models.yolov5_mobilenetv3_s import yolov5_mobilenetv3_s
    model = yolov5_mobilenetv3_s(num_classes=6)
    state_dict = torch.load('release/yolov5_shufflenet_1_5x@coco-s+animal/best.pt', map_location='cpu')['state_dict']
    model.load_state_dict(state_dict)

    model.eval()
    target_path = 'release/yolov5_shufflenet_1_5x@coco-s+animal/yolov5_shufflenet_1_5x.onnx'
    sim_target_path = target_path.replace(".onnx", "-sim.onnx")
    run(model, target_path)
    os.system(f'python -m onnxsim {target_path} {sim_target_path}')

    proto_path = target_path.replace(".onnx", ".prototxt")
    caffemodel_path = target_path.replace(".onnx", ".caffemodel")
    os.system(f'cd nano/onnx2caffe; python convertCaffe.py ../../{sim_target_path} ../../{proto_path} ../../{caffemodel_path}')

    from nano.xnnc import convert_to_xnnc
    convert_to_xnnc(proto_path, [['856'], ['836'], ['output']])
