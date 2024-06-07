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
Automatically calibrate webcam using an 8x5 checkerboard (Requires 9x6 checkerboard linked here).[Checkerboard-A3-45mm-8x5.pdf](https://github.com/user-attachments/files/15743410/Checkerboard-A3-45mm-8x5.pdf)


Simply run the code, hold a checkerboard on phone or printed in front of the webcam.

Every second, the webcam will capture another photo, so change the orientation of the checkerboard within the camera frame to ensure robust calibration.
### Draw 3D Cube on Aruco Tag (given calibration)
Given the calibration matrices ***mtx*** and ***dist***, which this code calculates from a calibration done at the beginning, the webcam will load in the Aruco Tag dictionary for 6x6_250 and detect tags in the frame.

If there is an Aruco Tag detected, this code will render a 3D cube on top of the Aruco Tag with the cube orientation corresponding to the orientation of the tag

<img width="479" alt="Perspective1" src="https://github.com/momoroowala/classic_CV/assets/10859547/306135eb-db54-4404-abf6-dc2064b88192">
<img width="477" alt="Perspective2" src="https://github.com/momoroowala/classic_CV/assets/10859547/f01de531-a7db-421b-b708-b1dc25d5302d">
