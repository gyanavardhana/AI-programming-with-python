# Image Classifier Project 

This repository contains code for training and using an image classifier based on the VGG19 model to classify images of flowers. The project was implemented using PyTorch and includes scripts for training the model (`train.py`), predicting classes of images (`predict.py`), and utility functions (`utils.py`). Additionally, there is a `cat_to_name.json` file containing mappings of category labels to flower names.

## Project Structure

- `train.py`: This script is used to train the image classifier. It takes arguments such as data directory, architecture, learning rate, hidden units, epochs, dropout, and GPU usage to train the model.
- `predict.py`: The prediction script loads a trained model checkpoint and predicts the class of an input image. It takes arguments for the input image path, checkpoint path, top k probabilities, category names mapping, and GPU usage.
- `utils.py`: This script provides utility functions used for training and prediction tasks, such as loading checkpoints, processing images, and displaying results.
- `cat_to_name.json`: JSON file containing mappings of category labels to flower names.
- `checkpointer.pth`: Example checkpoint file storing trained model parameters.

## Usage

1. **Training**: To train the image classifier, run the `train.py` script. Example usage:
    ```
    python train.py flowers --arch vgg19 --learning_rate 0.001 --hidden_units 512 --epochs 10 --dropout 0.2 --gpu gpu
    ```

2. **Prediction**: After training or using a pre-trained model, you can use the `predict.py` script to predict the class of an input image. Example usage:
    ```
    python predict.py flowers/test/1/image_06752.jpg --checkpoint checkpointer.pth --top_k 5 --category_names cat_to_name.json --gpu gpu
    ```

3. **Utility Functions**: The `utils.py` script provides various utility functions used in training and prediction tasks.

## Dataset

The dataset used for training and testing the image classifier consists of flower images. The dataset is organized into training, validation, and testing sets, each containing images of different flower species.

## Environment

For practicing and running the code, Google Colab and Kaggle environments were utilized. These platforms provide GPU support, which significantly speeds up model training.

## Acknowledgments

The project uses PyTorch, a powerful open-source machine learning library, and leverages pre-trained models available in the torchvision module. Special thanks to the developers and contributors of PyTorch and torchvision for their valuable contributions.

For any questions or feedback, please feel free to contact the project maintainer.

Thank you for your interest in this image classifier project!
