import copy
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

ground_truth_file = 'qset3_internal_and_local.gt.txt'
# Original data set with listed images
SOURCE_DIR = Path("images")
# Destination folder for the class sorted images
DST_DIR = Path("./data")
# Folder for the train, val and test split images
OUTPUT_DIR = Path("output")
# batch size
batch = 32
# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# The number of Logo brands on the dataset
num_of_classes = 37


# 1. Data Preparation Functions
def copy_data_to_folders(source, destination):
    """
    Copy images form the source image folder to a correspondng subfolder in the data.
    if such folder exists, copy image there.
    else create the folder and copy
        # TODO: add txt file for the bounding boxes
        # with repeating images, maybe one file?
    Arguments:
        source: the location of the file
        destination: whete it will be moved
    """
    if os.path.exists(destination) or os.path.isdir(destination):
        # print(f" File copied in existing {destination}")
        pass
    else:
        os.makedirs(destination)
        # print(f"File copied to {destination}")

    shutil.copy2(source, destination)


def sort_data_into_class_folders():
    """
    Function to read the ground truth file and sort the images in "images folder"
    to their respective class subfolder within data (DST_DIR)

    """
    with open(ground_truth_file, "r") as file:
        # Extract lines from groundtruth file and get column values
        # eg file line: ['Veolia_0011', 'Veolia', '07599203.jpg', 'logo', '0', '434', '121', '451', '132']
        for line in file.readlines():
            input = line.split()
            label = input[1]
            path = input[2]
            has_logo = bool(int(input[4]))
            # bounding_box = input[5:]
            destination = ""

            # Define destination folders based  on class
            if has_logo:
                source = SOURCE_DIR / path
                destination = os.path.join(DST_DIR, label)  # DST_DIR/label

            else:
                source = SOURCE_DIR / path
                no_logo_dst = "no_logos"
                destination = os.path.join(DST_DIR, no_logo_dst, label)

            # Function to copy images in the above defined destination
            copy_data_to_folders(source, destination)


# Fill train validation and test sets with symlink:
def create_symlinks(source_folder, destination_folder, file_list):
    """
    Iterate over the given list and copile the path of the image.
    if th epath already exist(i.e. image is already added to the folder)
    continue, otherwise a symlink from the source to the destination folder is created for the image at the specified path

    Arguments:
        source_folder: where the image is stored
        destination_folder: where the symlink will be
        file_list: list of images to be moved through symlinks
    """
    for file_name in file_list:
        print(source_folder)
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)

        # Check if image is already linked
        if os.path.exists(destination_path):
            print(f"Warning: Destination already exists - {destination_path}")
        else:
            # print(f"Source path: {source_path}")
            # print(f"Destination path: {destination_path}")
            os.makedirs(destination_folder, exist_ok=True)
            # os.symlink(source_path, destination_folder) # Error opening image: [Errno 2] No such file or directory: 'output/train/Nike/07705525.jpg'
            shutil.copy2(source_path, destination_folder)


def create_train_val_test_folders():
    """
    Create folder paths for the train test and val sets
    Return:
        traion_folder, validation_folder, test_folder: folder for the data split later
    """

    train_folder = os.path.join(OUTPUT_DIR, "train")
    validation_folder = os.path.join(OUTPUT_DIR, "val")
    test_folder = os.path.join(OUTPUT_DIR, "test")

    # Create train, val and test folder to store symlinks
    for folder in [train_folder, validation_folder, test_folder]:
        os.makedirs(folder, exist_ok=True)

    return train_folder, validation_folder, test_folder


def train_val_test_split(train_folder, validation_folder, test_folder):
    """
    Function to split the input data into train, validation and test sets into "output" folder
    Args:
        train_folder: destination folder for the train data set
        validation_folder: destination folder for the validation data set
        test_folder: destination folder for the test data set
    """
    # Iterate through the class directories in the data folder and fill the datasets
    for class_folder in os.listdir(DST_DIR):
        # Remove the no_logo class, as it contains negatives
        if class_folder == "no_logos":
            print(f"Directory {class_folder} skipped")
            continue

        # Class_path to find the class directory withing the source data folder
        class_path = os.path.join(DST_DIR, class_folder)

        # List the contents of the class folder and separate 80% train, 10% validation, 10% test
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            random.shuffle(images)

            number_of_images = len(images)
            train_split = int(0.8 * number_of_images)
            val_split = int(0.1 * number_of_images)
            # test_split = int(0.1 * number_of_images)

            train_set = images[:train_split]
            val_set = images[train_split : train_split + val_split]
            test_set = images[val_split:]

            # Make the validation set and test set contain unseen images
            val_set = list(set(val_set) - set(train_set))
            test_set = list(set(test_set) - set(train_set) - set(val_set))

            # Create the links to the class folders in the data parent folder
            create_symlinks(
                class_path, os.path.join(train_folder, class_folder), train_set
            )
            create_symlinks(
                class_path, os.path.join(validation_folder, class_folder), val_set
            )
            create_symlinks(
                class_path, os.path.join(test_folder, class_folder), test_set
            )


def transform_datasets(train_mean, train_std):
    """
    Transform data for train, val and test sets to conform to resnet requirements and collect into ImageFolder objects
    Args:
        train_mean:  mean to normalise the data
        train_std: standart deviation to normalise data
    Return:
        train_dataset, val_dataset, test_dataset: the ImageFolder object with the respective dataset
    """
    # resize the image 256, convert ToTensor, normalise to a range 0-1 using the mean and standart deviation
    data_transforms = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ]
    )
    # normalise all three datasets
    train_dataset = ImageFolder(
        root=os.path.join(OUTPUT_DIR, "train"), transform=data_transforms
    )
    val_dataset = ImageFolder(
        root=os.path.join(OUTPUT_DIR, "val"), transform=data_transforms
    )
    test_dataset = ImageFolder(
        root=os.path.join(OUTPUT_DIR, "test"), transform=data_transforms
    )

    return train_dataset, val_dataset, test_dataset


def load_data_into_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    """
    Load data into respective dataloaders from respective ImageFolder objects
    Args:
        train_dataset: ImageFolder of train data
        val_dataset: ImageFolder of validation data
        test_dataset: ImageFolder of test data

    Return:
        train_dataloader, val_dataloader, test_dataloader: Dataloader from respective ImageFolder objects of the data sets
    """
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True
    )

    return train_dataloader, val_dataloader, test_dataloader


# 2. Initialise the Model Function:
def init_model(resnet_model, lr):
    """
    Initialise the ResNet model for fixed feature extraction
    Args:
        resnet_model: the torchvision ResNet model to be initialised
        lr: learning rate for the model optimiser
    Return:
        model: the resnet model with frozen layers
        optimiser: the SGD optimiser with the given model parameters
    """
    model = resnet_model
    # Freeze network layers
    for param in model.parameters():
        param.requires_grad = False
    # Get number of input units to the last layer
    numb_units = model.fc.in_features
    # Change the last layer
    model.fc = nn.Linear(numb_units, num_of_classes)
    # Move model to device
    model = model.to(device)
    # Initialise the SGD optimiser with the model parameters and learning rate argument
    optimiser = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    return model, optimiser


# 3. Train and Validation Function
def train_val(model, criterion, optimiser, dataloaders, datasets, num_epochs=5):
    """
    Train and validate model function on train and val dataset

    Args:
        model: the model to be trained
        dataloaders: the dataloader dictionary to access train and val dataloaders
        datasets: the dictionary of train, test and val dtasets to get the number
                of examples in the train and val datasets
    Return:
        model: trained model
    """
    # From pytorch transfer learning tutorial
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print("Epoc {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimiser.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    # Forward pass
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    # Compute loss
                    loss = criterion(outputs, labels)

                    # Compute gradients and update parameters if train
                    if phase == "train":
                        loss.backward()
                        optimiser.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels).item()  # labels.data

            epoch_loss = running_loss / len(datasets[phase])
            epoch_acc = running_corrects / len(datasets[phase])

            print(f"Epoch loss: {epoch_loss}")
            print(f"Epoch acc: {epoch_acc}")
            print()
            print(
                f"{phase.title()} Loss: {epoch_loss:.4f} Acc.: {epoch_acc * 100:.2f} %"
            )
            print()
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60 :.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Accuracy: {100 * best_acc :.2f} %")

    model.load_state_dict(best_model_wts)

    return model


# 4. Test Function
def test(model, dataloaders, datasets):
    """
    Test model function on test dataset
    Args:
        model: the model to be tested
        dataloaders: the dataloader dictionary to access test_dataloaders( could be only test_dataloader but used for consistency)
        datasets: the dictionary of train, test and val datasets to extract the number of examples in the set
    """
    model.eval()
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels).item()

    test_acc = running_corrects / len(datasets["test"])
    print(f"Test acc: {test_acc}")
    print(f"Test Acc.: {100 * test_acc :.2f} %")


# 5. Main Function
def main():
    # 1. Sort images from source folder to subfolders by class name
    if not os.path.exists(DST_DIR):
        print("File directory data doesn't exists")
        print(DST_DIR)
        sort_data_into_class_folders()
    # else: Data is already sorted into classes
    print("Data is sorted into classes!")

    # 2. Data Augmentation, loading and test, val, train split
    train_mean = np.array([0.485, 0.456, 0.406])
    train_std = np.array([0.229, 0.224, 0.225])

    # Check if data has been split into train, test and validation sets
    if not os.path.exists(OUTPUT_DIR):
        print("No train/val/test split done yet!")
        # Create container folders and split the data
        train_folder, validation_folder, test_folder = create_train_val_test_folders()
        # split data
        train_val_test_split(train_folder, validation_folder, test_folder)
    else:
        print("Train/val/test split is done!")

    # Data is split -> Load data into ImageFolders
    train_dataset, val_dataset, test_dataset = transform_datasets(train_mean, train_std)
    # Datasets collector
    datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    # Debuging
    print(f"Number of examples in train: {len(train_dataset)}")
    print(f"Number of examples in val: {len(val_dataset)}")
    print(f"Number of examples in test: {len(test_dataset)}")

    # Load data into dataloaders
    train_dataloader, val_dataloader, test_dataloader = load_data_into_dataloaders(
        train_dataset, val_dataset, test_dataset, batch
    )
    # Dataloaders collector
    dataloaders = {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader,
    }

    # 3.Initialise criterion and learning rate
    criterion = nn.CrossEntropyLoss()
    lr = 0.001

    # 4. Initialise the models
    # ResNet 18
    # Visual Separators
    print()
    print("Train model ResNet 18")
    print("-" * 100)
    # Initialise ResNet18 model and SGD optimiser
    model_18, optimiser = init_model(torchvision.models.resnet18(pretrained=True), lr)
    # Train and validate model
    model_18 = train_val(
        model=model_18,
        criterion=criterion,
        optimiser=optimiser,
        dataloaders=dataloaders,
        datasets=datasets,
    )
    # Test model
    test(model=model_18, dataloaders=dataloaders, datasets=datasets)

    # ResNet152
    # Visual Separators
    print()
    print("Train model ResNet 152")
    print("-" * 100)
    # Initialise ResNet152 model and SGD optimiser
    model_152, optimiser = init_model(torchvision.models.resnet152(pretrained=True), lr)
    # Train and test the model
    model_152 = train_val(model_152, criterion, optimiser, dataloaders, datasets)
    test(model=model_152, dataloaders=dataloaders, datasets=datasets)

    ##ResNet34
    # print()
    # print(f"Train model ResNet 34")
    # print("-" * 100)
    # # Initilise Resnet34 model and SGD optimiser
    # model_34, optimiser = init_model(torchvision.models.resnet34(pretrained=True), lr)
    ## Train and test the model
    # model_34 = train_val(model_34, criterion, optimiser, dataloaders, datasets)
    # test(model=model_34, dataloaders=dataloaders, datasets=datasets)

    ##ResNet50
    # print()
    # print(f"Train model ResNet 50")
    # print("-" * 100)
    ## Initilise Resnet50 model and SGD optimiser
    # model_50,, optimiser = init_model(torchvision.models.resnet50(pretrained=True), lr)
    # #Train and test the model
    # model_50 = train_val(model_50, criterion, optimiser, dataloaders, datasets)
    # test(model=model_50, dataloaders=dataloaders, datasets=datasets)

    ## Resnet 101
    # print()
    # print(f"Train model ResNet 101")
    # print("-" * 100)
    ## Initilise Resnet101 model and SGD optimiser
    # model_101, optimiser = init_model(torchvision.models.resnet101(pretrained=True), lr)
    ##Train and test the model
    # model_101 = train_val(model_101, criterion, optimiser, dataloaders, datasets)
    # test(model=model_101, dataloaders=dataloaders, datasets=datasets)


if __name__ == "__main__":
    main()
