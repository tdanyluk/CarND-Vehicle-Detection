## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[imgCarNotCar]: ./output_images/car_not_car.png
[imgHog]: ./output_images/hog.png
[imgSlidingWindow]: ./output_images/sliding_window.png
[imgHeat]: ./output_images/heat.png
[imgLabeled]: ./output_images/labeled.png
[imgBoundingBoxes]: ./output_images/bounds.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in CarDetector::extract_training_features (car_detector.py lines TODO to TODO).

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Car - not car][imgCarNotCar]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![Hog][imgHog]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and I found `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` good enough.
Orientations is fine grained enough, but not too much.
Pixels_per_cell is small enough to capture details of a car, but not too small.
Cells_per_block is big enough to enable multiple grouping of cells when normalizing the histograms, but it is not too large.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using hog, spatial and color histogram features in CarDetector::fit_classifier (car_detector.py lines TODO to TODO).

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I did the sliding window search in CarDetector::calculate_car_rectangles (car_detector.py lines TODO to TODO).
Actually it is just a function that calls the find_cars function from the lessons.

I decided to use 3 scales: 64x64, 96x96 and 128x128, with 75% overlap.
I have seen that 75% overlap gives a great improvement compared to only 50%, but it still does not generate extremely many windows.
The 3 scales seemed to be enough to cover all cars which are at a reasonable distance and it gave a considerable improvement above using only two scales.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![Result of sliding window search][imgSlidingWindow]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

The corresponding code is CarDetector::_process_frame (car_detector.py lines TODO to TODO)

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![Bounding boxes and heat images][imgHeat]

### Here is the output of `scipy.ndimage.measurements.label()` on the summed and thresholded heatmap from all six frames:
![Labeled image][imgLabeled]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![Bounding boxes][imgBoundingBoxes]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I faced a problem: At the sliding window search I used 50% overlap at first, which made my pipeline unreliable.

My pipeline will fail for example when a car passes us too fast: not enough heat will be concentrated to the same place.
It will also 'fail' when multiple cars are near each other, it will possibly show them as one.