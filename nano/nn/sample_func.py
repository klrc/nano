import torch.nn as nn

class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        print('this is a test module.')
        return x