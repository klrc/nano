import torch
from torch.nn.modules import conv
from nano.models.model_zoo.nano_ghost import GhostNano_3x3_m96

model = GhostNano_3x3_m96(num_classes=3)
legacy = torch.load("release/GhostNano_3x3_m96/GhostNano_3x3_m96.pt", map_location="cpu")
legacy = {k: v for k, v in legacy.items()}
target = model.state_dict()

replacement = {
    "neck.pw_fs1": "neck.encoder.0",
    "neck.pw_fs2": "neck.encoder.1",
    "neck.pw_fs3": "neck.encoder.2",
    "neck.in_csp_fs1": "neck.compressor_topdown.0",
    "neck.in_csp_fs2": "neck.compressor_topdown.1",
    "neck.out_csp_fs2": "neck.compressor_bottomup.0",
    "neck.out_csp_fs3": "neck.compressor_bottomup.1",
    "neck.dp_fs1": "neck.downsample.0",
    "neck.dp_fs2": "neck.downsample.1",
}

converted = {}
for k, r in replacement.items():
    for name, v in legacy.items():
        if k in name:
            converted[name.replace(k, r)] = v
            print("replacing", name, name.replace(k, r))

for name, v in legacy.items():
    if name in target:
        converted[name] = v
        print('directly adding', name)

# check
for k, v in target.items():
    if k not in converted:
        raise NotImplementedError(k)
for k, v in converted.items():
    if k not in target:
        raise NotImplementedError(k)

model.load_state_dict(converted)
torch.save(model.state_dict(), "release/GhostNano_3x3_m96/GhostNano_3x3_m96_r.pt")
