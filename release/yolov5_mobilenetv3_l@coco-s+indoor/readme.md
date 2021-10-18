     Epoch   gpu_mem       box       obj       cls    labels  img_size
  0%|                                                                                                                              | 0/3024 [00:00<?, ?it/s]/home/sh/.local/lib/python3.9/site-packages/torch/nn/functional.py:3609: UserWarning: Default upsampling behavior when mode=bilinear is changed to align_corners=False since 0.4.0. Please specify align_corners=True if the old behavior is desired. See the documentation of nn.Upsample for details.
  warnings.warn(
/home/sh/.local/lib/python3.9/site-packages/torch/nn/functional.py:3657: UserWarning: The default behavior for interpolate/upsample with float scale_factor changed in 1.6.0 to align with other frameworks/libraries, and now uses scale_factor directly, instead of relying on the computed output size. If you wish to restore the old behavior, please set recompute_scale_factor=True. See the documentation of nn.Upsample for details. 
  warnings.warn(
     0/999     1.35G   0.04207   0.00565   0.01017        78       416:   0%|                                                      | 0/3024 [00:01<?, ?it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|███████████████████████████████| 86/86 [00:30<00:00,  2.83it/s]
                 all       2734       6604      0.651       0.56       0.61      0.313
              person       2734       4528      0.593      0.577      0.596       0.24
             bicycle       2734        337      0.699      0.378      0.512      0.221
                 car       2734       1201      0.515      0.621      0.599      0.333
          motorcycle       2734        325      0.743      0.578      0.655      0.331
                 bus       2734        213      0.704      0.646      0.687      0.438
