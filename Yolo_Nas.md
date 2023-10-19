## Niche_YOLO_NAS
This script provides a Python class, Niche_YOLO_NAS, which facilitates the training, evaluation, and prediction of models for object detection using YOLO format datasets. Below is a brief description of the functionality provided:

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
niche_yolo_nas = Niche_YOLO_NAS(path_model, dir_train, dir_val, dir_test, name_task)
### Training:
niche_yolo_nas.train(path_yaml, path_train_txt, path_val_txt, batch_size, num_epochs)

(command I used) python trial_nas.py --iter 1 --n_train 200 --yolo_base yolo_nas_l --suffix trial1 here 200 is the number of images and yolo_nas_l

### Evaluation:
evaluation_result = niche_yolo_nas.evaluate_trained_model(best_model, data_yaml_path, data_type="test")

### Get Evaluation Matrix:
matrix = niche_yolo_nas.get_evaluation_matrix(best_model, data_yaml_path, data_type="test", conf=0.5, plot=True)
