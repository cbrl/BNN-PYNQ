import os
import csv
import scipy.io as sio
from PIL import Image
from copy import deepcopy
from json import dumps
from yapf.yapflib.yapf_api import FormatCode


# Calculate the accuracy of a networks results using the test labels
def calculate_accuracy(results, labels):
    countRight = 0
    for idx in range(len(labels)):
        if labels[idx] == results[idx]:
            countRight += 1
    return countRight*100/len(labels)

# Merge dictionaries that contain sub-dictionaries
def dict_of_dicts_merge(*dicts):
    out = {}
    for item in dicts:
        overlapping_keys = out.keys() & item.keys()
        for key in overlapping_keys:
            if (isinstance(out[key], dict) and isinstance(item[key], dict)):
                out[key] = dict_of_dicts_merge(out[key], item[key])
        for key in item.keys() - overlapping_keys:
            out[key] = deepcopy(item[key])
    return out

def dict_to_str(dict):
    dict_string = dumps(dict)
    formatted_code, _ = FormatCode(dict_string)
    return formatted_code

def write_dict_to_file(file_name, output_dict):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    f = open(file_name, "w+")
    f.write(dict_to_str(output_dict))
    f.close()


# Load the CIFAR-10 test set, given the root folder of the CIFAR-10 dataset
def load_cifar10_testset(folder, num_images=10000):
    num_images = min(num_images, 10000)

    input_file = ''
    if num_images == 10000:
        input_file = folder + "/test_batch.bin"
    else:
        input_file = folder + "/test_batch_{}".format(num_images) + ".bin"

    if not os.path.exists(input_file):
        with open(folder + "/test_batch.bin", "rb") as infile:
            with open(input_file, "wb+") as outfile:
                for i in range(num_images):
                    outfile.write(infile.read(3073))

    labels = []
    with open(input_file, "rb") as file:
        for i in range(num_images):
            #read first byte -> label
            labels.append(int.from_bytes(file.read(1), byteorder="big"))
            #read image (3072 bytes) and do nothing with it
            file.read(3072)
        file.close()

    return (input_file, labels)


# Load the GTSRB test set, given the Ground Truth file and Images folder
def load_gtsrb_testset(gt_file, images_folder, num_images=12630):
    num_images = min(num_images, 12630)

    image_files = []
    images = []
    labels = []

    with open(gt_file) as gtfile:
        gtreader = csv.reader(gtfile, delimiter=';')
        next(gtreader)
        for row in gtreader:
            image_files.append(images_folder + '/' + row[0])
            labels.append(int(row[7]))

    labels = labels[0:num_images]
    for i in range(num_images):
        images.append(Image.open(image_files[i]))
        images[i].load()

    return (images, labels)


# Load the SVHN test set, given the SVHN test file
def load_svhn_testset(file, num_images=26032):
    num_images = min(num_images, 26032)

    images = []
    labels = []

    data = sio.loadmat(file)

    # Subtact 1 to match classifier output. E.g. The classifier will return class 1 for an image of the number 2.
    label_mat = data['y'] - 1
    image_mat = data['X']

    labels = label_mat.transpose().tolist()[0]
    labels = labels[0:num_images]

    #for i in range(image_mat.shape[3]):
    for i in range(num_images):
        images.append(Image.fromarray(image_mat[:,:,:,i]))
        images[i].load()

    return (images, labels)


# Load the MNIST test set, given the root folder of the MNIST test set
def load_mnist_testset(folder, num_images=10000):
    num_images = min(num_images, 10000)

    idx3_path = folder + "/t10k-images-idx3-ubyte"
    idx1_path = folder + "/t10k-labels-idx1-ubyte"
    labels = []
    
    with open(idx1_path, "rb") as lbl_file:
        #read magic number and number of labels (MSB first) -> MNIST header
        magicNum = int.from_bytes(lbl_file.read(4), byteorder="big")
        countLbl = int.from_bytes(lbl_file.read(4), byteorder="big")
        #now the labels are following byte-wise
        for idx in range(num_images):
            labels.append(int.from_bytes(lbl_file.read(1), byteorder="big"))
        lbl_file.close()

    return (idx3_path, labels)
