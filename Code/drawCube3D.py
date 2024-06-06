import numpy as np
import cv2 as cv

def draw_cube(corners ,img, rvec, tvec, size=0.1):
    # Define cube vertices in 3D space
    cube_points = np.array([[-size, -size, 0],
                            [size, -size, 0],
                            [size, size, 0],
                            [-size, size, 0],
                            [-size, -size, 1],
                            [size, -size, 1],
                            [size, size, 1],
                            [-size, size, 1]])

    # Project cube points from 3D to 2D using the camera parameters
    cube_points_2d, _ = cv.projectPoints(cube_points, rvec, tvec, mtx, dist)
    cube_points_2d = np.int32(cube_points_2d).reshape(-1, 2)
    # Draw cube edges
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]

    for edge in edges:
            pt1 = tuple(cube_points_2d[edge[0]].astype(int).ravel())
            pt2 = tuple(cube_points_2d[edge[1]].astype(int).ravel())
            cv.line(img, pt1, pt2, (0, 255, 0), 2)
    return img


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:5].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
# Initialize the webcam
cap = cv.VideoCapture(0)

for i in range(15):
    ret, frame = cap.read()

    if not ret:
        break


    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (8,5), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(frame, (8,5), corners2, ret)
        cv.imshow('img', frame)
        cv.waitKey(1000)
#cap.release()
#cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
parameters = cv.aruco.DetectorParameters()

cv.waitKey(3000)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect AprilTags in the frame
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, _ = cv.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        for i in range(len(ids)):
            rvec, tvec, _ = cv.aruco.estimatePoseSingleMarkers(corners[i], 1, mtx, dist)
            frame = draw_cube(corners, frame, rvec, tvec, size=0.5)

    cv.imshow('AR Cube', frame)

    if cv.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv.destroyAllWindows()