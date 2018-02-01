import os
import shutil
import numpy as np


def split_files(num_classes=2, split_ratio=0.2):
    base = os.getcwd() + "/101_ObjectCategories"
    train = os.getcwd() + "/train"
    validate = os.getcwd() + "/validate"

    # remove old directories
    shutil.rmtree(train, ignore_errors=True)
    shutil.rmtree(validate, ignore_errors=True)

    # make directories for train and validate
    os.makedirs(train)
    os.makedirs(validate)

    folders = [x[0] for x in os.walk(base)] # find all directories
    folders.pop(0) # remove base dir

    # select directories to use
    if 1 < num_classes < len(folders):
        folders = [folders[i] for i in range(num_classes)]
    num_train = 0
    num_validate = 0

    for folder in folders:
        # Create directories in train and validate
        dir_train = folder.replace(base, "train")
        dir_val = folder.replace(base, "validate")
        os.makedirs(dir_train)
        os.makedirs(dir_val)

        for file in os.listdir(folder):
            # split files between train and validate
            if np.random.rand(1) < split_ratio:
                shutil.copyfile("{}/{}".format(folder, file), "{}/{}".format(dir_val, file))
                num_validate += 1
            else:
                shutil.copyfile("{}/{}".format(folder, file), "{}/{}".format(dir_train, file))
                num_train += 1

    print("Train images: {}, Validation images: {}".format(num_train, num_validate))
    return num_train, num_validate


if __name__ == "__main__":
    split_files(num_classes=2, split_ratio=0.2)
