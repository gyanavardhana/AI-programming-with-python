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

## Running on Google Colab

To run this project on Google Colab, follow these steps:

1. Open [Google Colab](https://colab.research.google.com/).
2. Click on "File" > "Open Notebook". or
3. paste the link of the public Jupyter notebook provided below:
   - [Public Jupyter Notebook Link](https://colab.research.google.com/drive/1cBMladdWBoiQQXb1VTTO3CDzrYcasEVP?usp=sharing)
4. Make sure to upload the required `cat_to_name.json` file.
5. Follow the instructions in the notebook to execute the code cells for training and prediction.

Please note that the `cat_to_name.json` file containing mappings of category labels to flower names should be uploaded to Google Colab before running the notebook.

Before running the notebook, change the current runtime to T4 GPU for faster training. To change the runtime:
- Click on "Runtime" > "Change runtime type".
- Select "GPU" from the "Hardware accelerator" dropdown menu.
- Click "Save".

Please note that training epochs may vary based on the runtime and dataset size. Larger datasets and slower runtimes may require longer training epochs.

the training of epochs will be similar to this screenshot below:

![image](https://github.com/gyanavardhana/AI-programming-with-python/assets/89439095/ce85506d-023e-46a9-9067-36108179773a)


## Dataset

The dataset used for training and testing the image classifier consists of flower images. The dataset is organized into training, validation, and testing sets, each containing images of different flower species.

## Environment

For practicing and running the code, Google Colab and Kaggle environments were utilized. These platforms provide GPU support, which significantly speeds up model training.

## Acknowledgments

The project uses PyTorch, a powerful open-source machine learning library, and leverages pre-trained models available in the torchvision module. Special thanks to the developers and contributors of PyTorch and torchvision for their valuable contributions.

For any questions or feedback, please feel free to contact 
# Contact:
- mail: gyanavardhanmamidisetti@gmail.com
- github usernames: gyanavardhana

Thank you for your interest in this image classifier project!
