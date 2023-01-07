Image Filtering with Convolutional Neural Networks
==================================================
This project aims to demonstrate the use of convolutional neural networks (CNNs) for image filtering tasks. The model is trained on the [Tiny ImageNet dataset](https://paperswithcode.com/dataset/tiny-imagenet), which consists of 64x64 color images in 200 classes.

<center>

<figure>
<img src="https://github.com/the-infiltrator/interview/blob/main/ML_Engineer/Img2Img/img2img_network.png?raw=true" width="50%">
<figcaption>Overview of the Img2Img network with 48 trainable parameters.</figcaption>
</figure>

## Run the Code: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZM7yhoYH8DtDObbqN62H1uPkYuV0ZD1Q?authuser=3#scrollTo=X9KRZiU9Xkx2)

</center>

-----------

Dependencies
------------

To use this code, you'll need the following libraries:

-   [PyTorch](https://pytorch.org/): Deep learning framework for training and deploying models.
-   [NumPy](https://numpy.org/): Library for working with numerical data in Python.
-   [OpenCV](https://opencv.org/): Library for image processing and computer vision tasks.
-   [scikit-image](https://scikit-image.org/): Library for image processing in Python.
-   [matplotlib](https://matplotlib.org/): Library for creating plots and charts in Python.
-   [seaborn](https://seaborn.pydata.org/): Library for statistical data visualization in Python.

Files
-----

The repository contains the following files:

-   `model.py`: Contains the code for creating and training the model.
-   `utils.py`: Contains utility functions for filtering images and visualizing the model's output.




Usage
-----

Please run `Img2Img.ipynb` for a demonstration of the complete end-to-end training and evaluation script which can also be run in the browser [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZM7yhoYH8DtDObbqN62H1uPkYuV0ZD1Q?authuser=3#scrollTo=X9KRZiU9Xkx2).


The model is trained using the L1 loss function and optimized with the Adam optimizer. To improve training speed and prevent overfitting, the following techniques are used:
To train a model using the `train_model` function, you will need to provide the following arguments:

-   `model`: This is the model that you want to train. It should be an instance of a PyTorch `nn.Module` class.
-   `train_dataset`: This is the training dataset that you will use to train the model. It should be an instance of a PyTorch `Dataset` class.
-   `val_dataset`: This is the validation dataset that you will use to evaluate the model during training. It should be an instance of a PyTorch `Dataset` class.

You can also provide the following optional arguments:

-   `max_epochs`: The maximum number of epochs to train for. The default is 100.
-   `early_stopping_patience`: The number of epochs to wait before stopping training if the validation loss does not improve. The default is 5.
-   `batch_size`: The batch size to use for training and validation. The default is 128.
-   `learning_rate`: The initial learning rate for the Adam optimizer. The default is 0.001.
-   `save_best`: Flag to indicate whether to save the model with the best validation loss. The default is False.
-   `save_path`: The file path to save the model to. Only used if `save_best` is True. The default is None.

Example:

```{python}
import torch
from model import create_img2img_model
from utils import load_filtered_dataset

# Create the model
num_input_channels = 3  # RGB color image
num_output_channels = 1  # Filtered image
model = create_img2img_model(num_input_channels, num_output_channels)

# Load the training and validation datasets
train_dataset, val_dataset = load_filtered_dataset()

# Train the model
train_model(model, train_dataset, val_dataset, save_best=True, save_path='best_model.pth')`
```
This will train the model on the TinyImageNet dataset and save the trained model to a file.

To apply a filter to an image, use the `apply_filter` function in `utils.py`. This function takes in the path to an image and the trained model, and returns the filtered image.

To preview the model's output on a dataset, use the `preview_predictions` function in `utils.py`. This function takes in the dataset and the trained model, and displays a grid of filtered images.


To visualize the model's training progress, use the `plot_losses` function in `utils.py`. This function takes in the training and validation loss arrays and displays a plot of the loss over time.




Results
-----

| Fig 1. Model Performance on learning a Sobel Filter                                                               | Fig 2. Model Performance on learning a Random Filter                                                                 |
|-----------------------------------------------------------------------|-----------------------------------------------------------------------|
| ![Fig 1. Model Performance on learning a Sobel Filter](https://github.com/the-infiltrator/interview/blob/main/ML_Engineer/Img2Img/Results/img2img_outputs_sobel.png?raw=true) | ![Caption for image 2](https://raw.githubusercontent.com/the-infiltrator/interview/main/ML_Engineer/Img2Img/Results/img2img_outputs_randomfilter.png) |









Training and Optimization
-------------------------

The model is trained using the L1 loss function and optimized with the Adam optimizer. To improve training speed and prevent overfitting, the following techniques are used in the code:

-   Cosine annealing: The learning rate is gradually decreased over the course of training using a cosine function. This helps to prevent oscillations and convergence to suboptimal solutions.
-   Early stopping: Training is stopped if the validation loss does not improve for a certain number of epochs. This helps to prevent overfitting to the training data.

| Fig 1. Learning a Sobel Filter                                                               | Fig 2. Learning a Random Filter                                                                 |
|-----------------------------------------------------------------------|-----------------------------------------------------------------------|
| ![Fig 1. Model Performance on learning a Sobel Filter](https://github.com/the-infiltrator/interview/blob/main/ML_Engineer/Img2Img/Results/sobel_training_curve.png?raw=true) | ![Caption for image 2](https://github.com/the-infiltrator/interview/blob/main/ML_Engineer/Img2Img/Results/randomfilter_training_curve.png?raw=true) |






Evaluation
----------

To evaluate the performance of the model, you can use the `preview_predictions` function in `utils.py`. This function takes in the trained model and a validation dataset, and displays a grid of input images, model outputs, and target images. You can use the `compute_metrics` function in `utils.py` to compute the peak signal-to-noise ratio (PSNR) and structural similarity index (SSIM) of the model's output.



## Some Answers

- If the image is really large or not of a standard size, it may be necessary to resize the image or perform some form of cropping or padding to ensure that it is compatible with the model. This can be done using image preprocessing techniques such as resizing or padding.

- At the edges of the image, it may be necessary to perform some form of padding to ensure that the model has enough information to make a prediction. This can be done using techniques such as zero padding or reflection padding.

- The runtime performance of training and inference can depend on a number of factors, including the size of the dataset, the complexity of the model, the hardware being used, and the optimization techniques being employed. In general, larger datasets and more complex models will take longer to train and perform inference on, while faster hardware and more advanced optimization techniques can help to improve performance.

- The model described in the code above is not a fully connected network. It is a convolutional neural network (CNN), which means that it consists of multiple convolutional layers that operate on the input image in a sliding window fashion, rather than fully connected layers that connect every input unit to every output unit.

- There are several optimization techniques built into Pytorch that can help to make training and inference faster, including techniques such as batch normalization, data parallelism, and distributed training. Additionally, Pytorch has a number of built-in optimizers that can be used to train deep learning models, such as SGD, Adam, and RMSprop.

- To make inference really fast, it is often necessary to optimize the model for inference by techniques such as model pruning, quantization, and distillation. Additionally, using hardware acceleration techniques such as using a GPU or an inference engine can also help to speed up inference.

- There are a number of ways to know when training is complete, including monitoring the loss and accuracy of the model on the training and validation sets, and looking for signs of overfitting or underfitting. Another approach is to specify an early stopping criterion, such as a maximum number of training epochs or a threshold for the validation loss.

- The Sobel kernel size can be specified in the model by setting the kernel size parameter in the Conv2d layers. For example, in the current model, the kernel size is set to 3x3.

