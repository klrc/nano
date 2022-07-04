if __name__ == "__main__":
    from arch.control_unit.detection_dataset_traveler import travel_dataset, travel_files
    from arch.control_unit.detection_pipeline import coco_class_names

    coco_class_names.append("gun")
    coco_class_names.append("human hand")

    # travel_dataset("/Volumes/ASM236X/IndoorCVPR_09/images", coco_class_names)

    queue = [
        "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/airport_inside/airport_inside_0134.jpg",
        "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/airport_inside/airport_inside_0274.jpg",
        "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/airport_inside/airport_inside_0344.jpg",
        "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/auditorium/auditorium_560_43.jpg",
        #     "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/bar/bar_0116.jpg",
        #     "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/bathroom/indoor_0124.jpg",
        #     "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/bathroom/indoor_0279.jpg",
        #     "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/bedroom/bedroom3.jpg",
        #     "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/bedroom/c_26.jpg",
        #     "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/bedroom/IMG_1556.jpg",
        #     "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/bedroom/IMG_2127.jpg",
        #     "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/bedroom/indoor_0584.jpg",
        #     "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/bookstore/bookstore_23_21_flickr.jpg",
        #     "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/bookstore/bookstore_24_24_flickr.jpg",
        #     "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/bookstore/Librairie_07_07_altavista.jpg",
        #     "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/casino/casino_0001.jpg",
        #     "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/casino/casino_0016.jpg",
        #     "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/casino/casino_0034.jpg",
        #     "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/casino/casino_0039.jpg",
        #     "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/casino/casino_0040.jpg",
        #     "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/computerroom/computadores.jpg",
        #     "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/computerroom/Salle_Informatique_Bellavista_060831_2_.jpg",
        #     "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/concert_hall/allenroom.jpg",
        #     "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/corridor/corridora5.jpg",
        #     "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/corridor/hallway1_c.jpg",
        #     "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/corridor/pasillo_edificio_escuela__480x640_c.jpg",
        #     "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/dining_room/dining031.jpg",
        #     "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/dining_room/dining057.jpg",
        #     "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/garage/garajep0.jpg",
        #     "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/grocerystore/DSCN0258.jpg",
        #     "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/hairsalon/DVP4957863_P.jpg",
        #     "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/inside_subway/inside_subway_0291.jpg",
        #     "/Volumes/ASM236X/IndoorCVPR_09/ImagesRaw/mall/Buenos_Aires_shopping_center_2_.jpg",
    ]
    travel_files(queue, coco_class_names, label_suffix=".xml", img_dir="ImagesRaw", label_dir="Annotations")
