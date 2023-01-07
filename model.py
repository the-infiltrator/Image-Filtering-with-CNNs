
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
from skimage.metrics import peak_signal_noise_ratio

def create_img2img_model(num_input_channels, num_output_channels):
    class Img2ImgNetwork(nn.Module):
        def __init__(self, num_input_channels, num_output_channels):
            super(Img2ImgNetwork, self).__init__()
            self.conv1 = nn.Conv2d(num_input_channels, num_output_channels, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(num_output_channels, num_output_channels, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(num_output_channels, num_output_channels, kernel_size=3, stride=1, padding=1)

        def forward(self, x):
            x = nn.functional.relu(self.conv1(x))
            x = nn.functional.relu(self.conv2(x))
            x = self.conv3(x)
            return x
    
    model = Img2ImgNetwork(num_input_channels, num_output_channels)
    return model


def train_model(model, 
                train_dataset,
                val_dataset,
                max_epochs=100, 
                early_stopping_patience=5,
                batch_size=128,
                learning_rate=0.001,
                save_best=False,
                save_path=None):
    """
    Trains the model on the training dataset and validates it on the validation dataset.
    The model is trained using the Adam optimizer and the L1 loss function. The learning rate
    is scheduled using the CosineAnnealingLR scheduler. Early stopping is used to prevent overfitting.
    
    Parameters:
        model (torch.nn.Module): The model to be trained.
        train_dataset (torch.utils.data.Dataset): The training dataset.
        val_dataset (torch.utils.data.Dataset): The validation dataset.
        max_epochs (int, optional): The maximum number of epochs to train for. Default is 100.
        early_stopping_patience (int, optional): The number of epochs to wait before stopping
            training if the validation loss does not improve. Default is 5.
        batch_size (int, optional): The batch size to use for training and validation. Default is 128.
        learning_rate (float, optional): The initial learning rate for the Adam optimizer. Default is 0.001.
        save_best (bool, optional): Flag to indicate whether to save the model with the best validation loss.
            Default is False.
        save_path (str, optional): The file path to save the model to. Only used if save_best is True.
            Default is None.
            
    Returns:
        tuple: A tuple containing the lists of training losses and validation losses at each epoch.
    """

    # Create a data iterator for the training dataset
    train_iterator = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_iterator = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True)


    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize the early stopping counter
    early_stopping_counter = 0

    # Initialize the best validation loss
    best_val_loss = float("inf")

    # Keep track of the loss and other metrics during training
    losses = []
    validation_losses = []


    # Initialize the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    # Loop over the epochs
    for epoch in range(max_epochs):
        # Set the model to training mode
        model.train()
        
        # Initialize the epoch loss
        epoch_loss = 0.0
        
        # Loop over the training batches
        for input_images, target_images in train_iterator:
            # Zero the optimizer gradients
            optimizer.zero_grad()

            # Forward pass through the model
            output_images = model(input_images)
            
            # Compute the loss
            loss = loss_fn(output_images, target_images)
            
            # Backward pass through the model
            loss.backward()
            
            # Update the model parameters
            optimizer.step()
            
            # Accumulate the epoch loss
            epoch_loss += loss.item()
        
        # Set the model to evaluation mode
        model.eval()
        
        # Initialize the validation loss
        val_loss = 0.0
        
        # Loop over the validation batches
        for input_images, target_images in val_iterator:
            # Forward pass through the model
            output_images = model(input_images)
            
            # Compute the loss
            loss = loss_fn(output_images, target_images)
            
            # Accumulate the validation loss
            val_loss += loss.item()
        
        # Compute the average training and validation losses
        avg_train_loss = epoch_loss / len(train_iterator)
        losses.append(avg_train_loss)

        avg_val_loss = val_loss / len(val_iterator)
        validation_losses.append(avg_val_loss)
        
        # Print the epoch loss and validation loss
        print(f"Epoch: {epoch+1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        
        # Update the learning rate
        scheduler.step()
        
        # Check if the validation loss is the best seen so far
        if avg_val_loss < best_val_loss:
            # Update the best validation loss
            best_val_loss = avg_val_loss
            if save_best:
              torch.save(model.state_dict(), save_path)
            
            # Reset the early stopping counter
            early_stopping_counter = 0
        else:
            # Increment the early stopping counter
            early_stopping_counter += 1
            
        if early_stopping_counter>=early_stopping_patience:
          print('Early Stopping')
          break
    
    return losses, validation_losses
