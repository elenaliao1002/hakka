import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image, ImageDraw

from convert_annotations import *

# Function to get the annotations from json
def extract_info_from_json(file_path):
    # Opening JSON file
    f = open(file_path)
    # returns JSON object as a dictionary
    data = json.load(f)

    # create categories dataframe
    categories = {
        "category_id": [],
        "name": [],
        "name_readable": [],
        "supercategory": [],
    }
    for i in range(len(data["categories"])):
        categories["category_id"].append(data["categories"][i]["id"])
        categories["name"].append(data["categories"][i]["name"])
        categories["name_readable"].append(data["categories"][i]["name_readable"])
        categories["supercategory"].append(data["categories"][i]["supercategory"])
    df_categories = pd.DataFrame(data=categories)

    # create images dataframe
    images = {"image_id": [], "file_name": [], "width": [], "height": []}
    for i in range(len(data["images"])):
        images["image_id"].append(data["images"][i]["id"])
        images["file_name"].append(data["images"][i]["file_name"])
        images["width"].append(data["images"][i]["width"])
        images["height"].append(data["images"][i]["height"])
    df_images = pd.DataFrame(data=images)

    # create annotations dataframe
    annotations = {
        "id": [],
        "image_id": [],
        "category_id": [],
        "bbox": [],
        "segmentation": [],
    }
    for i in range(len(data["annotations"])):
        annotations["id"].append(data["annotations"][i]["id"])
        annotations["image_id"].append(data["annotations"][i]["image_id"])
        annotations["category_id"].append(data["annotations"][i]["category_id"])
        annotations["bbox"].append(data["annotations"][i]["bbox"])
        annotations["segmentation"].append(data["annotations"][i]["segmentation"][0])
    df_annotations = pd.DataFrame(data=annotations)

    # merge all the dataframes together
    df = df_annotations.merge(df_images, on="image_id", how="left")
    df = df.merge(df_categories, on="category_id", how="left")

    # relabel categories starting from 0
    cat_dict = dict()
    for i, row in df.iterrows():
        if row["category_id"] not in cat_dict:
            cat_dict[row["category_id"]] = len(cat_dict)
    df["new_category_id"] = list(df["category_id"].map(cat_dict))

    return df


# Convert annotations dataframe to the required yolo format and write it to disk
def convert_to_yolov5(df, out_dir, split="train", mode="segmentation"):
    df_new = df

    # For each bounding box
    for index, row in df_new.iterrows():
        if mode == "bbox":
            bbox_buffer = []
            [x_min, y_min, width, height] = row["bbox"]

            # Transform the bbox coordinates as per the format required by YOLO v5
            b_center_x = (x_min + x_min + width) / 2
            b_center_y = (y_min + y_min + height) / 2
            b_width = width
            b_height = height

            # Normalize the coordinates by the dimensions of the image
            image_w, image_h = row["width"], row["height"]
            b_center_x /= image_w
            b_center_y /= image_h
            b_width /= image_w
            b_height /= image_h

            bbox_buffer.append(
                "{} {:.5f} {:.5f} {:.5f} {:.5f}".format(
                    row["new_category_id"], b_center_x, b_center_y, b_width, b_height
                )
            )

            # Name of the file which we have to save
            save_file_name = os.path.join(
                f"{out_dir}/labels/{split}", row["file_name"].replace("jpg", "txt")
            )

            if row["file_name"] not in filenames:
                filenames.append(row["file_name"])
                # Save the annotation to disk
                print("\n".join(bbox_buffer), file=open(save_file_name, "w"))
            else:
                print("\n".join(bbox_buffer), file=open(save_file_name, "a"))

        elif mode == "segmentation":
            seg_buffer = []
            segmentation = row["segmentation"]

#             Normalize the coordinates by the dimensions of the image
            image_w, image_h = row["width"], row["height"]
            for i in range(len(segmentation)):
                if i % 2 == 0:
                    segmentation[i] /= image_w
                else:
                    segmentation[i] /= image_h

            seg = f"{row['new_category_id']}"
            for i in range(len(segmentation)):
                seg += f" {round(segmentation[i], 5)}"
            seg_buffer.append(seg)

            # Name of the file which we have to save
            save_file_name = os.path.join(
                f"{out_dir}/labels/{split}", row["file_name"].replace("jpg", "txt")
            )

            # Save the annotation to disk
            print("\n".join(seg_buffer), file=open(save_file_name, "a"))

        else:
            raise ValueError("For mode please choose bbox or segmentation")


# check annotations
def plot_annotations(anno_path, mode="segmentation"):
    # Get the corresponding image file
    img_path = anno_path.replace("labels", "images").replace("txt", "jpg")
    assert os.path.exists(img_path)

    # Load the image
    image = Image.open(img_path)
    w, h = image.size

    cat_dict = dict()
    df = extract_info_from_json(
        os.path.join(os.path.dirname(anno_path), "annotations.json")
    )
    for i, row in df.iterrows():
        if row["new_category_id"] not in cat_dict:
            cat_dict[row["new_category_id"]] = row["name_readable"]

    with open(anno_path, "r") as file:
        annotation_list = file.read().split("\n")
        annotation_list = [ele for ele in annotation_list if ele != ""]
        annotation_list = [x.split(" ") for x in annotation_list]
        annotation_list = [[float(y) for y in x] for x in annotation_list]

    annotations = np.array(annotation_list)

    if mode == "bbox":
        plotted_image = ImageDraw.Draw(image)
        transformed_annotations = np.copy(annotations)

        transformed_annotations[:, [1, 3]] = annotations[:, [1, 3]] * w
        transformed_annotations[:, [2, 4]] = annotations[:, [2, 4]] * h

        transformed_annotations[:, 1] = transformed_annotations[:, 1] - (
            transformed_annotations[:, 3] / 2
        )
        transformed_annotations[:, 2] = transformed_annotations[:, 2] - (
            transformed_annotations[:, 4] / 2
        )
        transformed_annotations[:, 3] = (
            transformed_annotations[:, 1] + transformed_annotations[:, 3]
        )
        transformed_annotations[:, 4] = (
            transformed_annotations[:, 2] + transformed_annotations[:, 4]
        )

        for ann in transformed_annotations:
            obj_cls, x0, y0, x1, y1 = ann
            plotted_image.rectangle(((x0, y0), (x1, y1)))
            plotted_image.text((x0, y0 - 10), cat_dict[(int(obj_cls))])

        plt.figure(figsize=(15, 12))
        plt.imshow(np.array(image))
        plt.show()

    elif mode == "segmentation":
        img_ann = image.copy()
        plotted_image = ImageDraw.Draw(img_ann)

        for ann in annotations:
            xy = []
            for i in range(1, len(ann), 2):
                xy.append((int(ann[i] * w), int(ann[i + 1] * h)))
            plotted_image.polygon(xy, fill="wheat")
            plotted_image.text(
                (xy[0][0], xy[0][1] - 10), cat_dict[(int(ann[0]))], fill="#FF0000"
            )

        img = Image.blend(image, img_ann, 0.8)

        plt.figure(figsize=(15, 12))
        plt.imshow(np.array(img))
        plt.show()

    else:
        raise ValueError("For mode please choose bbox or segmentation")


if __name__ == "__main__":
    df_train = extract_info_from_json("data/labels/train/annotations.json")
    df_val = extract_info_from_json("data/labels/val/annotations.json")

    # for mode please choose bbox or segmentation
    convert_to_yolov5(df_train, "data", split="train", mode="segmentation")
    convert_to_yolov5(df_val, "data", split="val", mode="segmentation")
