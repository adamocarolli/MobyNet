import csv
import os
import shutil

# Load whale fluke training images into associated Class ID directories
# for model building.
#
# Output structure as below:
# data/
#   class_id_1
#       img1
#       img2 ..
#   class_id_2 ..
#       img1 ..
#
# Note: train.csv must be in current working directory (cwd) and train.zip
#       must be extracted in cwd for script to run.

with open('train.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')

    # Read training csv to map whale fluke images to Class ID's.
    dict = {}
    for row in readCSV:
        image_filename = row[0]
        image_id = row[1]

        if image_id == 'Id':
            pass
        elif image_id not in dict:
            dict[image_id] = [image_filename]
        else:
            dict[image_id] += [image_filename]


    # Ensure directory structure exists, if none exists
    # create directory structure.
    if not os.path.exists("./data"):
        os.makedirs("./data")
    if not os.path.exists("./data/train"):
        os.makedirs("./data/train")
    if not os.path.exists("./data/validation"):
        os.makedirs("./data/validation")
    if not os.path.exists("./data/test"):
        os.makedirs("./data/test")

    # Move whale fluke images into associated directory.
    for image_id in dict:
        if not os.path.exists("./data/train/" + image_id):
            os.makedirs("./data/train/" + image_id)
        for image_filename in dict[image_id]:
            shutil.move('./train/{}'.format(image_filename),
                        './data/train/{}/{}'.format(image_id, image_filename))
