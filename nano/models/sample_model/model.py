import nano.nn as nn


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.TestModule()

    def forward(self, x):
        x = self.layer(x)
        return x

def _test_sample_model():
    return TestModel()