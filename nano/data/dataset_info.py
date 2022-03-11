_coco_names = "person|bicycle|car|motorcycle|airplane|bus|train|truck|boat|traffic light|fire hydrant|stop sign|\
parking meter|bench|bird|cat|dog|horse|sheep|cow|elephant|bear|zebra|giraffe|backpack|umbrella|handbag|tie|\
suitcase|frisbee|skis|snowboard|sports ball|kite|baseball bat|baseball glove|skateboard|surfboard|\
tennis racket|bottle|wine glass|cup|fork|knife|spoon|bowl|banana|apple|sandwich|orange|broccoli|\
carrot|hot dog|pizza|donut|cake|chair|couch|potted plant|bed|dining table|toilet|tv|laptop|mouse|\
remote|keyboard|cell phone|microwave|oven|toaster|sink|refrigerator|book|clock|vase|scissors|teddy bear|\
hair drier|toothbrush"
_voc_names = "aeroplane|bicycle|bird|boat|bottle|bus|car|cat|chair|cow|diningtable|dog|horse|motorbike|person|\
pottedplant|sheep|sofa|train|tvmonitor"
_virat_names = "--|person|car|construction vehicle|carried object|bike"
_default_mapper = {
    "person": "person",
    "bike": "bike",
    "bicycle": "bike",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "construction vehicle": "truck",
    "else": "OOD",
}


class ClassHub:
    """
    Hub module for quick dataset class mapping
    """

    def __init__(self, names: str, mapper=None):
        super().__init__()
        supported_datasets = {
            "coco": _coco_names,
            "voc": _voc_names,
            "virat": _virat_names,
        }
        if names in supported_datasets:
            names = supported_datasets[names]

        self.names = names.split("|")
        if mapper is None:
            mapper = _default_mapper
        self.mapper = mapper

    def to(self, target_names="person|bike|car|OOD") -> None:
        if isinstance(target_names, str):
            assert "|" in target_names, 'please seperate with "|" or pass a list to the hub'
            target_names = target_names.split("|")

        index_map = {}
        for src_cid, k in enumerate(self.names):
            # disambiguation
            if k in self.mapper:
                k = self.mapper[k]
            elif "else" in self.mapper:  # out of distribution
                k = self.mapper["else"]
            # match
            if k in target_names:
                index_map[src_cid] = target_names.index(k)
        return index_map


if __name__ == "__main__":
    # use like this
    print(ClassHub("voc").to("person|bike|motorcycle|car|bus|truck|OOD"))

    for k, v in ClassHub("coco").to("person|bike|motorcycle|car|bus|truck|OOD").items():
        print('{:15s}'.format(_coco_names.split("|")[k]), "person|bike|motorcycle|car|bus|truck|OOD".split("|")[v])
    
    print()
    for k, v in ClassHub("voc").to("person|bike|motorcycle|car|bus|truck|OOD").items():
        print('{:15s}'.format(_voc_names.split("|")[k]), "person|bike|motorcycle|car|bus|truck|OOD".split("|")[v])

