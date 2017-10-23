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

### 2. HOG Parameters

Parameters are defined in the `trainSVM.py` file at line #121 to #131.In class different parameters were explored and I settled on the following:

| Parameter        | Value   | 
|:-------------:|:-------------:| 
| Colorspace      | YCrCb       | 
| Orientations      | 9      |
| Pixel per cell     | 8x8     |
| Cell per block      | 2        |
| HOG channels      | all        |
| Histogram binning      | 32        |
| Y-ROI      | 400 - 656       |

* **Colorspace**: YCrCb was used, because the classifier archived, with all other parameters kept, around 4% higher accuracy on the provided dataset than on RGB, only using HOG features.
* **Orientations**: 9 orientations were choosen because fewer resulted in a worse performance and more would not increase performance significantly.
* **Y-ROI**: The region of interest was limited in y-direction by hard coding it from 400 (horizon line) to 656 (car hood) pixel.
* **Histogram binning**: A color histogram for each color channel was added as feature to the HOG features, because it yields an increase from around 94% to 98% accuracy on the provided dataset.
* **Other**: All other parameters were used as default parameters provided from class.

### 3. Classifier Training

As a classifier the in class provided support vector machine (SVM) was choosen (`LinearSVC()` of `sklearn`, line #165 to #168). To train the classifier, the extracted features were split randomly into 80% training and 20% validation, using the `train_test_split()` function of `sklearn.model_selection`. 

In the end, the trained SVM was saved to `svm.pckl` using `pickle`.

## Sliding Window Search

### 1. Implementation

The provided sliding window function `find_cars()` from the udacity class was used in the `pipeline.py` file line #17 to #85. As a basic window size 64x64 pixel were used. A scale series of `(0.75, 1.0, 1.25, 1.5, 1.75, 2.0)` was used to search for cars in window sizes between 48 to 128 pixel. The upper bound was used, because cars close to the camera were detected best, using a scale factor of `2` and cars further away by using `0.75`. To keep time consumption manageable, smaller cars were neglected, which could have been detected by using smaller scaling factors. For the same reason a scaling step of 0.25 was choosen. Furthermore a `cells_per_step = 2` (line #43) was choosen because, smaller cars of 48px and smaller could have been stepped over in some frames by using a cell step of `3` or higher, which corresponds to 24px or more.


### 2. Pipeline performance & optimization

The overall pipeline (`pipeline.py`) uses YCrCb 3-channel HOG features plus 32 binned histograms of each RGB channel in the feature vector, which provided good performance of the trained SVM with around 98% accuracy on the provided data set with 80-20 train-test-split.

---

## Video Implementation

### 1. Video link

Here's a [link](./out.mp4) to my video result. The discribed approach performs reasonably well over the whole project video.

### 2. Filtering false positives & generating bounding boxes

To reduce the number of false positives, a heatmap of overlapping bounding boxes was computed over the `6 scales` (line #136). Furthermore, the bounding boxes were averaged over `5 consecutive frames` (line #100). After that a threshold of `20` (line #151) overlapping bounding boxes was applied to filter out the false positives. 

To combine the multiple bounding boxes, the outer boundaries of the heatmap after thresholding were used to compute the final detections with the help of the `scipy.label()` function:

![alt text][image1]

---

## Discussion

Eventough the discribed algorithm performs well on the project video, it may fail on other road scenarios:

* **Data**: The SVM was trained on roughly 16000 images of type car and non-car. Therefore the algorithm may fail on unseen objects, that have similiar shape and color features. Those may appear in more complex city scenes, while unlikely in the provided highway scene. More data, should increase the performance of the classifier, e.g. with the recently released Udacity data set.

* **Features**: The used color histogram and HOG features are high level and robust against different color and lighting conditions, but may not be discriminant enough to distinguish between similar looking objects in complex city scenes. Convolutional neural networks could provide better features with well designed data set, but are quite slow in comparison.

* **Overlapping objects**: If objects are overlapping in the video, the heatmap fails to distinguish between the two objects. NMS may perform better in finding maxima of overlapping objects in the heatmap.

* **Computation time**: At this point, the algorithm is too slow to be used to detect vehicles in real time applications. Integral images may help to boost the performance.