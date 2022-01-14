def overlap_split(img, model):
    """
    split the canvas into tl, tr, bl, br for high-resolution detection
    """
    h, w = img.size(-2), img.size(-1)
    sh, sw = h + 64, w + 64
    results = []
    for offset_h, offset_w in ((0, 0), (0, w - sw), (h - sh, 0), (h - sw, w - sw)):
        _sliced = img[..., offset_h : offset_h + sh, offset_w : offset_w + sw]
        _sr = model(_sliced)  # inference and training outputs
        _sr[..., 0] += offset_w
        _sr[..., 1] += offset_h
        _sr[..., 2] += offset_w
        _sr[..., 3] += offset_h
        results.append(_sr)
    return torch.cat(results, 1)