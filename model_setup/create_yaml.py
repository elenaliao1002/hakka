from convert_annotations import *


def create_yaml(json_path, output_path):
    df = extract_info_from_json(json_path)

    cat_dict = dict()
    new_cat_dict = dict()

    for i, row in df.iterrows():
        if row["category_id"] not in cat_dict:
            cat_dict[row["category_id"]] = row["name_readable"]

    for i, keys in enumerate(cat_dict.keys()):
        new_cat_dict[i] = cat_dict[keys]

    # number of classes
    nc = len(new_cat_dict)

    # class names
    names = list(new_cat_dict.values())

    with open(output_path, "w") as f:
        f.write("train: ../../data/images/train \n")
        f.write("val: ../../data/images/val \n")
        f.write("test: ../../data/images/test \n")
        f.write(f"nc: {nc} \n")
        f.write(f"names: {names} \n")


if __name__ == "__main__":
    json_path = "data/labels/train/annotations.json"
    output_path = "yolov5/data/food.yaml"
    create_yaml(json_path, output_path)
