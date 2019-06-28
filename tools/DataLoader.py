def loadMate():
    img_path = []
    img_label = []
    classes = {}

    with open("data-100k-27class.txt", "r") as img_data:
        for img in img_data.readlines():
            img = img.strip("\n")
            path, label = img.split(",")[0], img.split(",")[1:]
            img_path.append(path)
            img_label.append(label)

    with open("label.names", "r") as img_class:
        for index, class_name in enumerate(img_class.readlines()):
            classes[class_name.strip("\n")] = index

    return classes, img_path, img_label

