import os
from glob import glob
import subprocess as sp


def fetch_data(dataset, out_dir):
    if not os.path.isdir(out_dir):
        # download dataset
        download = ["aicrowd", "dataset", "download", "-c", f"{dataset}"]
        # create directories
        mkdirs = [
            "mkdir",
            "-p",
            f"{out_dir}/",
            f"{out_dir}/images",
            f"{out_dir}/labels",
            f"{out_dir}/labels/train",
            f"{out_dir}/labels/val",
        ]
        # unzip tar.gz files
        unzip_train = ["tar", "-xvf", "public_training_set_release_2.0.tar.gz"]
        mv_train = ["mv", "images/", f"{out_dir}/images/train/"]
        mv_ann_train = [
            "mv",
            "annotations.json",
            f"{out_dir}/labels/train/annotations.json",
        ]
        unzip_val = ["tar", "-xvf", "public_validation_set_2.0.tar.gz"]
        mv_val = ["mv", "images/", f"{out_dir}/images/val/"]
        mv_ann_val = [
            "mv",
            "annotations.json",
            f"{out_dir}/labels/val/annotations.json",
        ]
        unzip_test = ["tar", "-xvf", "public_test_release_2.0.tar.gz"]
        mv_test = ["mv", "images/", f"{out_dir}/images/test/"]
        # remove tar.gz
        rm_tar = [
            "rm",
            "public_test_release_2.1.tar.gz",
            "public_training_set_release_2.1.tar.gz",
            "public_validation_set_release_2.1.tar.gz",
            "public_training_set_release_2.0.tar.gz",
            "public_validation_set_2.0.tar.gz",
            "public_test_release_2.0.tar.gz",
            "0c5704db-adeb-438e-9efc-66e269a19c3c_public_training_annotations_only_2.1.tar.gz",
            "cfad74b4-d34b-4f13-96ee-da7ccd3c184a_predictions.json",
        ]

        steps = [
            download,
            mkdirs,
            unzip_train,
            mv_train,
            mv_ann_train,
            unzip_val,
            mv_val,
            mv_ann_val,
            unzip_test,
            mv_test,
            rm_tar,
        ]

        for step in steps:
            process = sp.Popen(step)
            process.wait()


if __name__ == "__main__":
    # downloading this dataset will take 7-8 mins
    fetch_data("food-recognition-benchmark-2022", "data")
