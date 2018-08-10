# object-detection-and-tracking-
This project consists of two componetns one is responsible for vehicle detection and tracking using Kalman filters and HOG+SVM
and the other component is responsible for implementing lane detection
Module 1(Detection consists of the following files)
1)desc.py responsible for computing HOG
2)detect.py main module which our main test file calls.This wraps the descriptor,sliding window ,kalman filter and threshold methods all in one file
3)scan.py performs sliding window detection at different scales
4)train.py performs training using SVm
5)track and match.py are modules responsible for performing kalman filter tracking and then selecting the most likely canidates for the bounding boxes detected by the detection module.For more details that underly the tracking concept please refer to 
https://arxiv.org/abs/1602.00763
The next module is the lane detection module.
it consists of the following files.
For an image obtained from the video,camera callibration and perspective mapping are performed.
this is done by callibrate_cameras.py and perspective_transform.py
Next we perform edge filtering and to get a better understanding of the lane markers we also augment the previous filtered result with an HLS filter.This is implemented in combined_thresh.py
The modules responsible for performing line detection are implemented in lanefit.py
Finally for visualization purposes we provide two files one for still image and one for video these are implemented in generate_examples.py and line_fit_video.py
