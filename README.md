# CV CLASSIC TECHNIQUES 
IMPLEMENTED IN PYTHON

### AutoEncoder for Fashion_MNIST Dataset
4-Layer Autoencoder with a latent space of 16 to represent the Fashion_MNIST Dataset from Keras
TSNE Representation also applied to the encoded latent space, allowing for a 2d plot of the high dimensional abstract extracted features after 2 layers of encoding.
### RANSAC on Open3D PointCloud
RANSAC stands for RANdom Sampling and Consensus. In this application, RANSAC is used for wall plane detection due to inherent property of rejecting outliers. When run, this code will take a sample pcd from Open3D library and highlight the largest plane in red.
Can change some parameters:
- num_iterations for more or less total iterations of the random sampling (sometimes, too many iterations may take too long for large point clouds)
- distance_threshold for different tolerances (if a plane has some textures that you want to include in the segmentation/detection)
- inlier_ratio for a more harsh/lenient RANSAC algorithm
### Webcam Calibration using Checkerboards
Automatically calibrate webcam using an 8x5 checkerboard (may require 9x6 if webcam is not capturing the correct corners).

Simply run the code, hold a checkerboard on phone or printed in front of the webcam.

Every second, the webcam will capture another photo, so change the orientation of the checkerboard within the camera frame to ensure robust calibration.
### Draw 3D Cube on Aruco Tag (given calibration)
Given the calibration matrices ***mtx*** and ***dist***, which this code calculates from a calibration done at the beginning, the webcam will load in the Aruco Tag dictionary for 6x6_250 and detect tags in the frame.

If there is an Aruco Tag detected, this code will render a 3D cube on top of the Aruco Tag with the cube orientation corresponding to the orientation of the tag
