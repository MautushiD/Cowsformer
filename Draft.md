# Detecting objects in animal and dairy sciences using machine learning models.


## Introduction

Object deteciton, which tracks the positions of interested objects (e.g., cows and sheeps), has been an essential tool to improve the livestock production. <two examples uses object detection, 3 senstences (triats (behaviors), species) for each study>. These studies have alleivate the problem that researchers want to solve.

However, the costs, including labor, time, and computational resources, to implement the object detection were rarely addressed. For example, <one example of large training samples>. Preparing this amount of samples can be labor-intesntive, as formatting and labeling the positions of each object in an image requires professional training in programming language. For example, COCO annotation format is the most common format for object detection. It requires organizing the coordinates of the bounding box, the class of the object, and the image size in a JSON file. Without related expertise, there is a barrier to implement object detection in animal and dairy sciences. Another obstacle of the implementation is the computational resources. Not every computer can implement the modern object detection models, which have millions of parameters and requires up to 12 GB of video memory. <two short example of large CV models (list their number of parameters and recommended VRAM requirement)>. Hence, knowing the computational cost is also an important factor for researchers to consider when implementing object detection. Lastly, the transferability of the published studies is always missed to be discussed. This factor is important when one research want to reproduce the same published work in their own research, which may have different lighting or environment that affects the model performance. Transfering the model to a new environment usually reuqires providing additional training efforts, which is also a cost to consider.

To date, most object detection studies in animal sciences did not address the listed concerns.
<TO BE FILLED>
    - Dr. Dorea's study (2-3 sentences)
    - YOLO models / Mask RCNN in animal science
    - Transformer is great but rare, list two work outside of animal science

In this study, we present a systematic benchmarking study that addresses the concerns and provides a guidence for researchers, regardless of their expertise, to implement object detection in livestock production studies. Specifically, we aims to discuss three major perspectives of the implementation:
- (1) How many training samples are required to achieve a certain accuracy?
- (2) How much computational resources are required to implement the object detection?
- (3) How much marginal efforts (i.e., samples) are required to transfer the model to a new environment?
We will validate each goal by using state of the art object detection models, including YOLOv8, YOLONAS, Mask R-CNN, and transformer-based model DETR, with their variants that have different model sizes. For benchmarking the performance in tranferability, we collected image datasets of dairy cows and ensured the variation of the lighting environment (i.e., day and night), camera angles (i.e., top-down and side-view), and even the breeds (i.e., Holstein and Jersey). We will also discuss the trade-off between the accuracy and the computational resources. With this guidence, researchers who want to implemenet the object detection can have better management of their research resources and efforts.

## Methods and Materials

(last, year)




In the field of animal sciences, computer vision based cow detection has mostly relied on models, like YOLOv3 and YOLOv5. These models have been proven effective in scenarios as discussed by [1]. [2]. Alongside these, there are models like Mask R CNN mentioned by [3] which offers a complementary approach. However it is important to note that these are a part of the wide range of object detection methods available.

This study delves into exploring the potential of models such as DETR, YoloNAS and YOLOv8. While these advanced models haven't gained much popularity in the animal science domain yet they have shown promising performance in various domains as highlighted by [4] in their comparison between different versions of YOLO. These models hold great promise for applications in animal sciences that remain largely unexplored.

Our study specifically focuses on optimizing training processes and identifying the data batches that can maintain accuracy, a move that could reshape the norms of dataset requirements.


## References:

[1] Andrea Pretto a,*, Gianpaolo Savio a, Flaviana Gottardo b, Francesca Uccheddu c, Gianmaria Concheri a a Department of Civil, Environmental and Architectural Engineering, University of Padua, Italy for A novel low-cost visual ear tag based identification system for precision beef cattle livestock farming. 2022. , https://doi.org/10.1016/j.inpa.2022.10.003 <br>
[2] Jan-Hendrik Witte, Jorge Marx Gom ́ ez for Introducing a New Workflow for Pig Posture Classification Based on a Combination of YOLO and EfficientNet. 2022. <br>
[3] Yongliang Qiao⁎, Matthew Truman, Salah Sukkarieh for Cattle segmentation and contour extraction based on Mask R-CNN for precision livestock farming. 2019. <br>
[4] Muhammad Hussain for YOLO-v1 to YOLO-v8, the Rise of YOLO and Its Complementary Nature toward Digital Manufacturing and Industrial Defect Detection. 2023 , https://doi.org/10.3390/machines11070677 <br>