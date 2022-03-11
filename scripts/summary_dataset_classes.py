import sys
import cv2
from loguru import logger

sys.path.append(".")


if __name__ == "__main__":
    try:
        logger.debug("Dataloader Test ------------------")
        from nano.data.dataset import person_vehicle_detection_preset
        import nano.data.transforms as T

        class_names = "person|bike|motorcycle|car|bus|truck|OOD".split("|")
        box_counts = [0 for _ in range(len(class_names))]
        dataset = person_vehicle_detection_preset((256, 448), class_names, "/Volumes/ASM236X")
        dataloader = dataset.as_dataloader(batch_size=16, num_workers=4, shuffle=True, collate_fn=T.letterbox_collate_fn)

        for images, target in dataloader:
            for _, cid, _, _, _, _ in target:
                box_counts[int(cid)] += 1
            total = sum(box_counts)
            line = [f"{k}: {v/total:.2%}" for k, v in zip(class_names, box_counts)]
            # line.append(f'sum: {sum(box_counts)}')
            line = [f'{x}\t' for x in line]
            print(" ".join(line))

    except Exception as e:
        logger.error(e)
        raise e
