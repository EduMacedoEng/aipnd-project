# Imports here
# Data handling and computation
from datetime import datetime
import numpy as np
import pandas as pd
import os

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch for Neural Networks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from torchvision.models import vgg16, VGG16_Weights
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision.models._utils')


# Image processing
from PIL import Image

# Additional tools
from sklearn.model_selection import train_test_split
import json

def load_data(data_path):
    
    data_dir = os.path.join(os.getcwd(), data_path)

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_dir = os.path.join(data_dir, 'train')
    train_dir = os.path.normpath(train_dir)  # Normalize the path

    valid_dir = os.path.join(data_dir, 'valid')
    valid_dir = os.path.normpath(valid_dir)  # Normalize the path

    test_dir = os.path.join(data_dir, 'test')
    test_dir = os.path.normpath(test_dir)  # Normalize the path

    # Check if these directories exist
    #print("Does Train Directory Exist?", os.path.exists(train_dir))
    #print("Does Validation Directory Exist?", os.path.exists(valid_dir))
    #print("Does Test Directory Exist?", os.path.exists(test_dir))
        
    # Defining transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_val_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Loading the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=test_val_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_val_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    validationloader = DataLoader(validation_data, batch_size=32)
    testloader = DataLoader(test_data, batch_size=32)

    # Combining these into a dictionary
    dataloaders = {
        'train': trainloader,
        'validation': validationloader,
        'test': testloader
    }
    
    return train_data, dataloaders

def label_mapping(path):
    with open(path, 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name

def build_model(arch, hidden_units, learning_rate, gpu):
    if arch == "vgg16":
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    else:
        # Add other architectures here
        pass
    
    for param in model.parameters():
        param.requires_grad = False
    
    # Define a new classifier
    model.classifier = nn.Sequential(
                            nn.Linear(hidden_units, 4096),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(4096, 1024),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(1024, 102), # Assuming 102 flower categories
                            nn.LogSoftmax(dim=1))
    
    # Define the criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, criterion, optimizer, device

def save_checkpoint(model, save_dir, arch, train_dataset, optimizer, name_classes, epochs):
    model.class_to_idx = train_dataset.class_to_idx
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    class_to_name = {idx_to_class[k]: name_classes[idx_to_class[k]] for k in idx_to_class}
    
    checkpoint = {
        'arch': arch,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict(),
        'classifier': model.classifier,
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs,
        'class_to_name': class_to_name
    }
    
    torch.save(checkpoint, save_dir + 'model_checkpoint_P2.pth')

def train_model(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    print("Initializing Training...")
    print("-----------------------------------------")
    print("Loading Data...")
    train_data, dataloaders = load_data(data_dir)
    print("Label Mapping...")
    name_classes = label_mapping('../cat_to_name.json')
    print("Bulding Model...")
    model, criterion, optimizer, device = build_model(arch, hidden_units, learning_rate, gpu)
    
    print("\nTraining Model:")
    print("-----------------------------------------")
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation pass
        model.eval()
        with torch.no_grad():
            validation_loss = 0
            accuracy = 0
            for inputs, labels in dataloaders['validation']:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)
                
                validation_loss += batch_loss.item()
                
                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Epoch {epoch+1}/{epochs}") 
        print(f"Train Loss: \t\t{running_loss/len(dataloaders['train']):.3f}")
        print(f"Validation Loss: \t{validation_loss/len(dataloaders['validation']):.3f}")
        print(f"Validation Acc: \t{(accuracy/len(dataloaders['validation'])*100):.1f}%\n")
        
    print("\nTraining Complete!")
    
    print("\nSaving Model:")
    print("-----------------------------------------")
    save_checkpoint(model, save_dir, arch, train_data, optimizer, name_classes, epochs)
    print("Saving Complete!")
    print("=========================================")
    
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Processing a PIL image for use in a PyTorch model
    # Open the image
    img = Image.open(image_path)

    # Resize the image where the shortest side is 256 pixels, keeping the aspect ratio
    img.thumbnail((256, 256))

    # Crop out the center 224x224 portion of the image
    width, height = img.size
    new_width, new_height = 224, 224
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    img = img.crop((left, top, right, bottom))

    # Convert to numpy array and normalize
    np_image = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Transpose the color channel to the first dimension
    np_image = np_image.transpose((2, 0, 1))

    return np_image

def load_checkpoint(filepath):
    # Load the saved file
    checkpoint = torch.load(filepath)

    # Rebuild the model: Assuming it's a vgg16 model for this example
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    
    # Remember to replace the classifier with the same architecture you used before
    classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                               nn.ReLU(),
                               nn.Dropout(0.5),
                               nn.Linear(4096, 1024),
                               nn.ReLU(),
                               nn.Dropout(0.5),
                               nn.Linear(1024, 102)) # Assuming 102 flower categories
    
    model.classifier = classifier

    # Load the state dict back into the model
    model.load_state_dict(checkpoint['state_dict'])

    # If you also need to load the optimizer state
    # optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load the class_to_idx
    model.class_to_idx = checkpoint['class_to_idx']
    model.class_to_name = checkpoint['class_to_name']

    return model

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    if title:
        ax.set_title(title) # Set the title on the ax object
    
    return ax

def predict_image(image_path, checkpoint, top_k, category_filepath, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    print("Initializing Prediction...")
    print("-----------------------------------------")
    print("Processing Image...")
    # Process the image
    img = process_image(image_path)
    print("Converting to Tensor...")
    # Convert to tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)

    # Add batch of size 1 to image
    image_tensor = image_tensor.unsqueeze(0)
    print("Loading Trained Model...")
    # Get model trained
    model = load_checkpoint(checkpoint)
    
    # Move the model and image to the current device
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    image_tensor = image_tensor.to(device)

    # Ensure no gradient is being calculated
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        output = model.forward(image_tensor)
    print("Converting probabilities...")
    # Convert output probabilities to probabilities
    probabilities = torch.exp(output)

    # Get the top k probabilities and indices
    top_probs, top_indices = probabilities.topk(top_k)

    # Convert to lists
    top_probs = top_probs.cpu().numpy().flatten()
    top_indices = top_indices.cpu().numpy().flatten()

    # Convert index to class
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[index] for index in top_indices]
    
    print("Saving prediciton...")
    display_prediction(img, category_filepath, top_probs, top_classes)
    
    print("Saving Complete!")
    print("=========================================")
    
    
def display_prediction(img, category_filepath, top_probs, top_classes):
    
    # Normalize the probabilities if they don't sum to 1
    probs_normalized = top_probs / np.sum(top_probs)
    
    # Convert indices to classes
    cat_to_name = label_mapping(category_filepath)
    class_names = [cat_to_name[cls] for cls in top_classes]

    # Plotting the image
    plt.figure(figsize = (15,10))
    ax = plt.subplot(2,1,1)

    # Display the image
    imshow(torch.from_numpy(img), ax, title=class_names[0])

    # Plotting the bar chart
    plt.subplot(2,1,2)
    sns.barplot(x=probs_normalized, y=class_names, color=sns.color_palette()[0])
    filename = f"{class_names[0]}_prediction_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.jpg"
    plt.savefig(os.path.join('../predictions/', filename))