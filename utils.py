
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision
import random
import numpy as np
import cv2
import torch
import torchvision
import random
import cv2
import requests
import zipfile
import os
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import seaborn as sns

def apply_filter(dataset, randomize=False):
    # Create a list to hold the filtered images
    filtered_dataset = []
    convert_tensor = transforms.ToTensor()

    # Generate a random 3x3 kernel if randomization is enabled
    kernel = None
    if randomize:
        kernel = np.random.rand(3, 3)

    for image, label in dataset:
        # Convert the image to grayscale
        fimage = convert_tensor(image).permute(1, 2, 0).numpy()  # Convert the image tensor to a NumPy array
        fimage = cv2.cvtColor(fimage, cv2.COLOR_RGB2GRAY)  # Grayscale image
        
        # Apply the filter
        if randomize:
            fimage = cv2.filter2D(fimage, -1, kernel)
        else:
            fimage = cv2.GaussianBlur(fimage,(3,3), 0, 0) 
            fimage = cv2.Sobel(fimage, cv2.CV_64F, dx=1, dy=1, ksize=3)  # Sobel filter
        
        # Convert the filtered image back to a tensor
        fimage = torch.tensor(fimage, dtype=torch.float).unsqueeze(0)
        filtered_dataset.append((convert_tensor(image), fimage))
    
    return filtered_dataset

def preview_filtered_dataset(train_dataset, num_images=5, save_plot=False):
    # Select a few random images from the training dataset
    indices = random.sample(range(len(train_dataset)), num_images)
    images = [train_dataset[i][0] for i in indices]
    filtered_images = [train_dataset[i][1] for i in indices]

    # Plot the images
    fig, ax = plt.subplots(num_images, 2, figsize=(5, 15))
    for i, (image, filtered_image) in enumerate(zip(images, filtered_images)):
        ax[i][0].imshow(image.permute(1, 2, 0).squeeze())
        ax[i][0].axis('off')
        ax[i][1].imshow(filtered_image.squeeze(), cmap='gray')
        ax[i][1].axis('off')
    
    if save_plot:
        plt.savefig('preview_plot.png')
    else:
        plt.show()

def preview_predictions(model, val_dataset, num_images=4, save_plot=False, plot_save_path=None):
    """
    Shows a preview of the model's predictions on the validation dataset. The input, output, and target images
    are plotted for a given number of images. The images are taken from the beginning of the validation dataset.
    
    Parameters:
        model (torch.nn.Module): The model to use for prediction.
        val_dataset (torch.utils.data.Dataset): The validation dataset to use for prediction.
        num_images (int, optional): The number of images to show in the preview. Default is 4.
        save_plot (bool, optional): Flag to indicate whether to save the plot. Default is False.
        plot_save_path (str, optional): The file path to save the plot to. Only used if save_plot is True.
            Default is None.
    """
    # Create a data iterator for the validation dataset
    val_iterator = torch.utils.data.DataLoader(val_dataset, batch_size=num_images, shuffle=False)
    
    # Get a batch of images from the val dataset
    input_images, target_images = next(iter(val_iterator))

    # Use the model to predict the output images
    output_images = model(input_images)

    # Detach the images from the computational graph to prevent a ValueError
    input_images = input_images.detach().numpy()
    output_images = output_images.detach().numpy()
    target_images = target_images.detach().numpy()

    # Create the figure and subplots
    fig, axs = plt.subplots(nrows=num_images, ncols=3, figsize=(9, num_images*4))

    # Plot the input, output, and target images for each image in the batch
    for i in range(num_images):
        axs[i, 0].imshow(input_images[i, 0, :, :], cmap="gray", vmin=0, vmax=1)
        axs[i, 0].set_title("Input")
        axs[i, 1].imshow(output_images[i, 0, :, :], cmap="gray")
        axs[i, 1].set_title("Model Output")
        axs[i, 2].imshow(target_images[i, 0, :, :], cmap="gray")
        axs[i, 2].set_title("Target")

    # Remove the axes labels and tick marks
    for ax in axs.flat:
        ax.set_axis_off()

    # Show the figure
    plt.show()
    
    # Save the plot if specified
    if save_plot:
        fig.savefig(plot_save_path)


def load_filtered_dataset(randomize=False):
    # Check if the data has already been downloaded
    if not os.path.exists("./data/tiny-imagenet-200"):
        # Download the TinyImageNet dataset
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        response = requests.get(url, stream=True)

        with open("tiny-imagenet-200.zip", "wb") as f:
            f.write(response.content)

        # Extract the contents of the zip file
        with zipfile.ZipFile("tiny-imagenet-200.zip", "r") as zip_ref:
            zip_ref.extractall("./data")

    # Load the TinyImageNet dataset
    dataset = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=None)

    # Apply the filter to the dataset
    filtered_dataset = apply_filter(dataset, randomize=randomize)

    # Split the filtered dataset into a training set and a validation set
    random.shuffle(filtered_dataset)
    split_index = int(0.8 * len(filtered_dataset))
    train_dataset = filtered_dataset[:split_index]
    val_dataset = filtered_dataset[split_index:]
    
    return train_dataset, val_dataset

def plot_losses(training_losses, validation_losses, save=False,trunc=0):
    # Set the seaborn style
    sns.set()

    # Create a high resolution figure
    plt.figure(figsize=(12, 8))


    # Plot the average of the loss
    plt.plot(training_losses[trunc:], label="Train Loss", linewidth=3, color='red')


    # # Plot the average of the loss
    plt.plot(validation_losses[trunc:], label="Validation Loss", linewidth=3, color='blue')

    # Add a legend
    plt.legend(loc="upper right", fontsize=20)

    # Set the x-axis and y-axis labels
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Loss", fontsize=20)

    # Set the title
    plt.title("Loss as training progresses", fontsize=24)

    # Set the font size of the tick labels
    plt.tick_params(labelsize=20)

    if save:
        # Save the figure to a file
        plt.savefig("training_loss.png", dpi=300)

    # Show the figure
    plt.show()


def compute_metrics(model, val_dataset):
    # Initialize the metric accumulators
    psnr_accum = 0
    ssim_accum = 0
    num_images = 0

    # Set the model to eval mode
    model.eval()

    # Iterate over the validation dataset
    for input_images, target_images in val_dataset:
        # Use the model to predict the output images
        output_images = model(input_images)

        # Detach the images from the computational graph to prevent a ValueError
        input_images = input_images.detach().numpy()
        output_images = output_images.detach().numpy()
        target_images = target_images.detach().numpy()

        # Calculate the metrics for the current batch of images
        psnr_batch = peak_signal_noise_ratio(target_images, output_images, data_range=1)
        ssim_batch = structural_similarity(target_images, output_images, data_range=1, multichannel=True)


        # Accumulate the metrics
        psnr_accum += psnr_batch
        ssim_accum += ssim_batch
        num_images += output_images.shape[0]

    # Calculate the mean metrics
    psnr_mean = psnr_accum / num_images
    ssim_mean = ssim_accum / num_images
    
    print(f"PSNR: {psnr_mean:.4f}")
    print(f"SSIM: {ssim_mean:.4f}")

    # Return the metrics in a dictionary
    return {"psnr": psnr_mean, "ssim": ssim_mean}