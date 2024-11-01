# Face parts detection with YOLOv8 🎯

If you find my work useful, please cite it as:

> Hernández Montilla, I (2024) Face parts detection with YOLOv8, DOI: https://doi.org/10.5281/zenodo.12507625

## Introduction

In this project I use Ultralytics' implementation of [YOLOv8](https://github.com/ultralytics/ultralytics). The goal is to train an algorithm that is able to detect separate face parts without having to use landmark detectors that don't do well when part of the face is occluded or missing. My goal is to also combine frontal, semi-frontal and profile face datasets so that the YOLO model works well on all of them. 

It is also a great opportunity to try out the `supervision` library by [Roboflow](https://github.com/roboflow/supervision). It looks really helpful for some common YOLO-related tasks such as drawing the detections.

![A live demo of YOLOv8 nano](images/live_demo.gif)

This project uses Python 3.11. To install the required packages, use `pip install -r requirements.txt` in a Python 3.11 environment.

## Motivation

All I want these models for is data exploration and check what face parts can be seen in an image. I'm talking about detecting _face parts_, which is not the same as detecting _faces_.

I've been asked many times: why not using facial landmark detectors? And the reason is that these do not work well with close-up images or heavily occluded faces, like this one:

![An example of a close-up image where facial landmark detection is not possible](images/closeup_example.jpg)

_Image source: [Pexels](https://images.pexels.com/photos/977601/pexels-photo-977601.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)_

I know there are several works about facial landmark detection for occluded faces (such as ["Robust face landmark estimation under occlusion"](https://pdollar.github.io/files/papers/BurgosArtizzuICCV13rcpr.pdf)), but a picture of the entire face is always needed. If I wanted to be able to detect face parts in close-up images, I would have to develop something new. And that's exactly what I've done.

## Data

For this experiment I'm using a variety of facial landmark detection datasets. Each dataset came in a different structure, so I had to deal with that in `prepare_full_dataset.py`:

- **Existing datasets**: all these datasets were processed by converting each group of facial landmarks (eye, mouth, nose, eyebrows) to a bounding box in the YOLO format (normalized ``x, y, w, h``).
  - [Helen dataset](http://www.ifp.illinois.edu/~vuongle2/helen/)
  - [Menpo2D dataset](https://github.com/jiankangdeng/MenpoBenchmark)
  - [AFW (Annotated Faces in the Wild) dataset](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
  - [LaPa (Landmark-guided face Parsing) dataset](https://github.com/JDAI-CV/lapa-dataset)
  - [FASSEG (FAce Semantic SEGmentation) dataset](https://github.com/massimomauro/FASSEG-repository): The original labels have eyebrows and hair in the same `hair` class, so I manually annotated subsets V2 and V3 with [CVAT](https://app.cvat.ai/). This dataset adds many low-quality profile images, which probably improves performance.
- **Custom datasets**:
  - [Pexels](https://pexels.com): I downloaded 257 images from this website and annotated them using [CVAT](https://app.cvat.ai/). As of today, I've annotated four batches of images, and I've tried to include pictures where parts of the face are missing.

⚠ **I am not sharing any of these datasets**: they are not mine, and they are 100% accessible from their corresponding sites. I may release the Pexels dataset that I create in the future, though.

## Results

### Data quality

Some datasets such as Helen may generate noisy examples when the images have more than one face but only one set of landmarks (i.e. the ones corresponding to the "main" face in the image). This is probably affecting the precision because the model is actually detecting all the faces in these images (which is good, though). Other datasets such as AFW have as many landmarks as faces in the images.

![A training batch with some images with incomplete labels](images/example_incomplete_labels.jpg)

### Performance

In this section you can see the performance of the _nano_ model. It struggles with eyebrows, but it works really well for eyes, mouths, and noses. I would need to add more close-up images of each part to increase the number of incomplete or occluded faces.

![Yolov8-nano F1 curve](images/F1_curve.png)

Here are the metrics of the _nano_ model:

![YOLOv8-nano results](images/results.png)

## Reports

Use `run.py` to run the model on a folder with images to obtain a CSV with all the detections.

