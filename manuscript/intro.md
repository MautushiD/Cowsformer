# Detecting objects in animal and dairy sciences using machine learning models.

## Introduction

Object deteciton, which tracks the positions of interested objects (e.g., cows and sheeps), has been an essential tool to improve the livestock production. For instance, a recent study introduced a method to accurately classify the posture of pigs based on images. They used YOLOv5, which achieved a pig detection accuracy of 0.994 with an APIoU=0.5. Then they utilized EfficientNet to classify the detected pigs into 'lying' and 'notLying' postures achieving a precision rate of 0.93. <added economic impact> This innovative approach demonstrated the advantages of using models, for classifying pig posture resulting in improvements, in accuracy. These studies have addressed the issue that researchers aimed to solve.[Jan-Hendrik Witte, 2022 ]. Another study, in the field of precision livestock farming has introduced an cost effective approach to identify beef cattle and monitor their activities using surveillance camera. This method utilizes real time object detection YOLOv3 to focus on the areas of the cattle and accurately locate ear tags. It also detects water consumption behaviors near drinking areas. By incorporating an Optical Character Recognition (OCR) algorithm the system is able to read cow IDs with accuracy achieving an impressive detection rate of 89% at mAP@0.50[Pretto a, 2022]. These studies have alleivate the problem that researchers want to solve.
<can be less technoical, only focus on economic impacts>

However, the costs, including labor, time, and computational resources, to implement the object detection were rarely addressed. For example, researchers conducted a study, on precision dairy farming, in Hokkaido, Japan. They successfully developed a system that could recognize cows ear tags using the object detector. To achieve this they needed a dataset of 20,000 training samples that were specifically focused on detecting cow heads.[Thi Thi Zin, Shuhei Misawa, Moe Zet Pwint, Kosuke Sumi, Kyohiro Yoshida, 2020]. Preparing this amount of samples can be labor-intesntive, as formatting and labeling the positions of each object in an image requires professional training in programming language. For example, COCO annotation [cite] format is the most common format for object detection. It requires organizing the coordinates of the bounding box, the class of the object, and the image size in a JSON file. Without related expertise, there is a barrier to implement object detection in animal and dairy sciences. Another obstacle of the implementation is the computational resources. Not every computer can implement the modern object detection models, which have millions of parameters and requires up to 12 GB of video memory. For instance, the VGG-16 model [cite] has 138 million parameters and recommends a VRAM of at least 8GB, while the ResNet-152 [cite] has around 60 million parameters with a recommended VRAM of 11GB. Hence, knowing the computational cost is also an important factor for researchers to consider when implementing object detection. Lastly, the transferability of the published studies is always missed to be discussed. This factor is important when one research want to reproduce the same published work in their own research, which may have different lighting or environment that affects the model performance. Transfering the model to a new environment usually reuqires providing additional training efforts, which is also a cost to consider.

To date, most object detection studies in animal sciences did not address the listed concerns.A study have shown results in using pseudo labeling to identify Holstein cows[Rafael E. P. Ferreira,Yong Jae Lee,João R. R. Dórea, 2023]. <address what they have achieved (2 sentences) before addressing the concerns> However there are concerns, about the robustness of the model that haven't been fully addressed. In the iterations of pseudo labeling there is a possibility that incorrect predictions can unintentionally reinforce. For example, cow's appearance is the critical factors for the model to recognize the cow. Different cow breeds show different skin patterns, and individuals of the same breed may also show different appeearence due to age, injuries, and mud from the environment. These changes pose a challenge for model transferability when ones attempt to reproduce the same study in their own research.

In studies conducted in the field of animal science researchers have been using state of the art learning frameworks to explore various applications. These include tasks, like classifying pig postures segmenting cattle instances, monitoring cow feeding behavior etc. Impressive results have been achieved using models such as YOLOv5 [Jan-Hendrik Witte,Jorge Marx Gom ́ez, 2022] Mask R CNN [Yongliang Qiao⁎, Matthew Truman, Salah Sukkarieh, 2019] and DRN YOLO based on YOLOv4 [Yu, Z.; Liu, Y.; Yu, S.; Wang, R.; Song, Z.; Yan, Y.; Li, F.; Wang, Z.; Tian, F. Automatic Detection Method of Dairy Cow Feeding Behaviour Based on YOLO Improved Model and Edge Computing. Sensors 2022,]. These approaches have proven successful in detecting postures performing segmentation and extracting useful features in complex farm environments. While these works have made contributions to precision livestock farming and laid the foundation for real time monitoring systems our research aims to push the boundaries by leveraging advanced models like YOLOv8, YOLONAS and DETR. Particularly noteworthy is DETR—a model that has gained recognition for its performance in object detection but has not yet been explored extensively in the field of animal science. Through harnessing these cutting edge models our goal is not to improve the accuracy and reliability of livestock recognition tasks but to spearhead the adoption of state of the art technologies, within the animal science sector.

Recent advancements in object detection have seen transformers, like those used for language tasks, being applied with great success. A standout example is the DEtection TRansformer (DETR) introduced by [Nicolas Carion?, Francisco Massa?, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko, 2020]. This model simplifies the object detection process, getting rid of many of the traditional steps, and performs as well as older, well-known models. Building on this, Dynamic DETR makes the process even more efficient, reducing the time it takes to train the model and improving its accuracy [Xiyang Dai Yinpeng Chen Jianwei Yang Pengchuan Zhang Lu Yuan Lei Zhang,2021 ]. These models have been used to detect objects in images of people, cars, and other common objects. However, they have not yet been applied to the field of animal science. Our research aims to explore the potential of these models in the context of livestock recognition tasks.

In this study, we present a systematic benchmarking study that addresses the concerns and provides a guidence for researchers, regardless of their expertise, to implement object detection in livestock production studies. Specifically, we aims to discuss three major perspectives of the implementation:
- (1) How many training samples are required to achieve a certain accuracy?
- (2) How much computational resources are required to implement the object detection?
- (3) How much marginal efforts (i.e., samples) are required to transfer the model to a new environment?
We will validate each goal by using state of the art object detection models, including YOLOv8, YOLONAS, Mask R-CNN, and transformer-based model DETR, with their variants that have different model sizes. For benchmarking the performance in tranferability, we collected image datasets of dairy cows and ensured the variation of the lighting environment (i.e., day and night), camera angles (i.e., top-down and side-view), and even the breeds (i.e., Holstein and Jersey). We will also discuss the trade-off between the accuracy and the computational resources. With this guidence, researchers who want to implemenet the object detection can have better management of their research resources and efforts.