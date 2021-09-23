from nano.torch_model import yolov5s
import torch

model = yolov5s()
print(model)
x = torch.rand(4,3,224,224)
y = model(x)
for m in y:
    print(m.shape)