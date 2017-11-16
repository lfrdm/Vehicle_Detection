# **Vehicle Detection Project**
---
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

---

[//]: # (Image References)
[image1]: ./output_images/heat-detection.gif
[image2]: ./output_images/vehicle.gif
[image3]: ./output_images/non-vehicle.gif


## Histogram of Oriented Gradients (HOG)

### 1. Extracting HOG features
HOG features were extracted in the file `SVMtrain.py` at line #72 with the `get_hog_features()` function from the udacity class. Therefore, all provided `vehicle` and `non-vehicle` images were read in using `os.walk` with the `load_data()` function (line #16 to #33).

Examples for `vehicles` left and `non-vehicles` right:

![alt text][image2] ![alt text][image3]

**Additional negatives**

Additional negative data was added to reduce the possibitly of detecting too many false negatives. Therefore, the existing negatives were flipped horizontally in line #12 in `extra_negatives.py` and added to the data set ([link](./data/non-vehicles/Extra_flipped)). 

Furthermore extra negatives of of a fixed area in a small video section, not containing cars were extraceted and added to the data set to reduce negatives encountered near yellow lines ([link](./data/non-vehicles/Extra_line)).

### 2. HOG Parameters

Parameters are defined in the `trainSVM.py` file at line #121 to #131.In class different parameters were explored and I settled on the following:

| Parameter        | Value   | 
|:-------------:|:-------------:| 
| Colorspace      | YCrCb       | 
| Orientations      | 12      |
| Pixel per cell     | 16x16     |
| Cell per block      | 2        |
| HOG channels      | all        |
| Y-ROI      | 400 - 550       |

* **Colorspace**: YCrCb was used, because the classifier archived, with all other parameters kept, around 4% higher accuracy on the provided dataset than on RGB, only using HOG features.
* **Orientations**: 12 orientations were choosen because fewer resulted in a worse performance and more would not increase performance significantly.
* **Y-ROI**: The region of interest was limited in y-direction by hard coding it from 400 (horizon line) to 550 (little above car hood) pixel.
* **Other**: All other parameters were used as default parameters provided from class.

### 3. Classifier Training

As a classifier the in class provided support vector machine (SVM) was choosen (`LinearSVC()` of `sklearn`, line #165 to #168) in `trainSVM.py`. To train the classifier, the extracted features were split randomly into 90% training and 10% validation, using the `train_test_split()` function of `sklearn.model_selection`. 

The `C=0.01` parameter of the SVC was choosen to train a SVC with a softened maximum margin in line #83 to train a SVM, that generalizes better to outlying data points.

In the end, the trained SVM was saved to `svm_new.pckl` using `pickle`.

## Sliding Window Search

### 1. Implementation

The provided sliding window function `find_cars()` from the udacity class was used in the `pipeline.py` file line #17 to #85. As a basic window size 64x64 pixel were used. A scale series of `(1.0, 1.5)` was used to search for cars with different sizes. To keep time consumption manageable, smaller cars were neglected, which could have been detected by using smaller scaling factors. Furthermore a `cells_per_step = 1` (line #43) was choosen because, smaller cars could have been stepped over.


### 2. Pipeline performance & optimization

The overall pipeline (`pipeline.py`) uses only YCrCb 3-channel HOG features, which provided good performance of the trained SVM with around 97% accuracy on the provided data set with 90-10 train-test-split.

---

## Video Implementation

### 1. Video link

Here's a [link](./out.mp4) to my video result. The discribed approach performs reasonably well over the whole project video.

### 2. Filtering false positives & generating bounding boxes

To reduce the number of false positives, a heatmap of overlapping bounding boxes was computed over the `2 scales` (line #136). Furthermore, the bounding boxes were averaged over `30 consecutive frames` (line #100). After that a threshold of `30` (line #151) overlapping bounding boxes was applied to filter out the false positives. 

To combine the multiple bounding boxes, the outer boundaries of the heatmap after thresholding were used to compute the final detections with the help of the `scipy.label()` function:

![alt text][image1]

Furthermore, the `decision_function()` (line #94) of the SVC was explored to reduce the number of false positives. But in the end, the standard decision of `0.0` was choosen, because higher values of 0.5 and higher resulted in fewer false positives, but also in false negatives.

---

## Discussion

Eventough the discribed algorithm performs well on the project video, it may fail on other road scenarios:

* **Data**: The SVM was trained on roughly 16000 images of type car and non-car. Therefore the algorithm may fail on unseen objects, that have similiar shape and color features. Those may appear in more complex city scenes, while unlikely in the provided highway scene. More data, should increase the performance of the classifier, e.g. with the recently released Udacity data set.

* **Features**: The HOG features are high level and robust against different color and lighting conditions, but may not be discriminant enough to distinguish between similar looking objects in complex city scenes. Convolutional neural networks could provide better features with well designed data set, but are quite slow in comparison.

* **Overlapping objects**: If objects are overlapping in the video, the heatmap fails to distinguish between the two objects. NMS may perform better in finding maxima of overlapping objects in the heatmap.

* **Computation time**: At this point, the algorithm is too slow to be used to detect vehicles in real time applications. Integral images may help to boost the performance.

* **Frame averaging**: As implemented, 30 frames are averaged to get a smooth detection of cars, and reduce false positives and gaps in the detection. The downside is, that the detection can be still active eventough the car has been occluded for a cupple of frames or left the FOV. Furthermore, the car needs to be in the FOV for a couple of frames to be detected, because a threshold of `30` overlapping bounding boxes was choosen.