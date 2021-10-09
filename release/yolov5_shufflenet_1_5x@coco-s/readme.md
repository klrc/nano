# autoanchor: Evolving anchors with Genetic Algorithm: fitness = 0.6985: 100%|████████████████████████████████████████████| 1000/1000 [00:21<00:00, 45.90it/s]
# autoanchor: thr=0.34: 0.9903 best possible recall, 3.12 anchors past thr
# autoanchor: n=9, img_size=416, metric_all=0.292/0.701-mean/best, past_thr=0.537-mean: 6,7,  8,19,  18,14,  16,36,  39,29,  31,77,  83,61,  75,159,  223,179
# autoanchor: New anchors saved to model. Update model *.yaml to use these anchors in the future.


layer {
name: "shuffle@"
type: "CppCustom"
bottom: "?"
top: "?"
cpp_custom_param {
module: "XnncShuffleChannel"
param_map_str: "group:2 "
}
}
