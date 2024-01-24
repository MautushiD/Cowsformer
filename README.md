## Project Overview
This repository hosts a pioneering project focused on the detection and tracking of cows in an indoor farm environment. Our aim is to address the unique challenges posed by varying farm lighting conditions and camera angles.

## Dataset Description
The project utilizes a diverse dataset, comprising images captured from multiple perspectives - specifically top view and side view angles. This dataset is unique in its coverage of different lighting conditions, with data collected both in natural sunlight during the day and under artificial conditions at night. This extensive range of data provides a comprehensive understanding of the environment in which the cows are monitored.

Additionally, the dataset includes images of two distinct breeds of cows: Jersey and Holstein, allowing for breed-specific analysis and detection.

## Methodology
At the heart of our project is the utilization of the YOLO NAS (You Only Look Once Neural Architecture Search) model. This advanced model has been fine-tuned with our custom dataset to ensure high accuracy and efficiency in cow detection under various indoor farm conditions.

Key aspects of our methodology include:

- Model Training: The YOLO NAS model has been trained on datasets of varying sizes (10, 25, 50, 100, and 200 images) to evaluate the model's performance against different data volumes.
- Model Variants: We have employed three different sizes of the YOLO NAS model - Small (S), Medium (M), and Large (L) - to determine how model size impacts performance across different dataset sizes.
## Repository Structure
```
.
├── 
├── 
├── 
├── .gitignore # Specifies intentionally untracked files to ignore
├── README.md # The README file you are currently reading


## Getting Started
To get started with this project, clone this repository using:

```bash
git clone https://github.com/MautushiD/CCowsformer.git
```

## Prerequisites
python 3.10+

numpy

pandas

scikit-learn

matplotlib

tensorflow

torch

super-gradients

ultralytics

pathlib

yaml

cv2

torchvision



## Installation
Install the required dependencies:

```bash
pip install -r requirements.txt
```

> Follow the instructions in each notebook to train the models and perform detection and tracking.


## Train the YOLO NAS model:
### nas.py - [.py file](https://github.com/MautushiD/Cowsformer/blob/main/models/nas.py)
This script provides a Python class, Niche_YOLO_NAS, which facilitates the training, evaluation, and prediction of models for object detection using YOLO format datasets. Below is a brief description of the functionality provided:

Run the train_nas.ipynb notebook to train the model on the SODA 10M dataset.
Detection and Tracking:
Use detection_tracking_from_vedio.ipynb to perform detection and tracking on video data.
The output will be saved in ./test_video_output and ./video_predictions.yaml.
Acknowledgements
This project utilizes the SODA 10M dataset, obtained from the SODA 2D official website. We thank the creators for making the dataset publicly available for research purposes.


##########################

### Dependencies:
- webbrowser
- super_gradients
- onemetric
- supervision
- torch
- yaml
- json
- os
- numpy
- ultralytics
- re
- pandas
- matplotlib
  
## Device Compatibility:
This script supports both CUDA and CPU modes. It automatically selects "cuda" if available, otherwise, it falls back to CPU mode.

## Class: Niche_YOLO_NAS
This is the primary class that provides various methods for model operations:

- **__init__**: Constructor to initialize directories, model path, and other necessary parameters.

- **load**: Loads the model. By default, it loads a yolo_nas_l model. However, a custom path can be provided to load other models.

- **train**: Train the model on a YOLO format dataset. Takes paths to YAML configuration, training, and validation text files, along with batch size and number of epochs as arguments.

- **evaluate_trained_model**: Evaluates a trained model on the provided dataset. Uses COCO format for evaluation. Users can select the dataset type as train, test, or val.

- **get_evaluation_matrix**: Gets the evaluation matrix (Confusion Matrix) for the trained model. Useful for visualizing and understanding the performance of the model on different classes.

## How to use:

### Initialization: First, create an instance of Niche_YOLO_NAS class.
```python
niche_yolo_nas = Niche_YOLO_NAS(path_model, dir_train, dir_val, dir_test, name_task)
```
### Training:
```python
niche_yolo_nas.train(path_yaml, path_train_txt, path_val_txt, batch_size, num_epochs)
```

(command I used) 
```shell
python trial_nas.py --iter 1 --n_train 200 --yolo_base yolo_nas_l --suffix trial1 here 200 is the number of images and yolo_nas_l
```

### Evaluation:
evaluation_result = niche_yolo_nas.evaluate_trained_model(best_model, data_yaml_path, data_type="test")


##### Output:
```
{'Precision@0.50': tensor(0.8883),  
 'Recall@0.50': tensor(0.9499),  
 'mAP@0.50': tensor(0.9343),  
 'F1@0.50': tensor(0.9180),  
 'Precision@0.50:0.95': tensor(0.6955),  
 'Recall@0.50:0.95': tensor(0.7437),  
 'mAP@0.50:0.95': tensor(0.7102),  
 'F1@0.50:0.95': tensor(0.7188)}
```
### Get Evaluation Matrix:
matrix = niche_yolo_nas.get_evaluation_matrix(best_model, data_yaml_path, data_type="test", conf=0.5, plot=True)
##### Output:
<p align="center">
<img src='https://github.com/MautushiD/Cowsformer/blob/main/slides/confusion_matrix.png?raw=true' width='70%' height='70%'>
</p>
