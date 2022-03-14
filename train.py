from nano.models.assigners.simota import SimOTA
from nano.models.model_zoo.nano_ghost import GhostNano_3x4_l128, GhostNano_3x4_m96, NanoDCT_3x3_m96
from nano.trainer.core import Trainer, Validator, Controller
from nano.data.dataset import person_vehicle_detection_preset, preson_vehicle_detection_preset_mscoco_test
import nano.data.transforms as T


if __name__ == "__main__":

    for model_template, target_resolution, batch_size in (
        (NanoDCT_3x3_m96, (256, 448), 256),
        # (GhostNano_3x4_m96, (256, 448), 128),
        # (GhostNano_3x4_l128, (256, 448), 128),
    ):
        device = "cuda:0"
        dataset_root = "../datasets"
        class_names = "person|bike|motorcycle|car|bus|truck|OOD".split("|")
        trainset = person_vehicle_detection_preset(target_resolution, class_names, dataset_root)
        trainloader = trainset.as_dataloader(batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=T.letterbox_collate_fn)
        valset = preson_vehicle_detection_preset_mscoco_test(target_resolution, class_names, dataset_root)
        valloader = valset.as_dataloader(batch_size=batch_size // 2, num_workers=4, collate_fn=T.letterbox_collate_fn)

        model = model_template(7)
        criteria = SimOTA(class_balance=(0.31, 1, 1, 1, 1, 1, 1))

        trainer = Trainer(trainloader, model, criteria, device, lr0=0.003, optimizer="AdamW", batch_size=batch_size)
        validator = Validator(valloader, class_names, device)
        controller = Controller(trainer, validator, patience=100)

        controller.run()
