# Project 3: [Camera Calibration and Fundamental Matrix Estimation with RANSAC]


# Project Review
## Project subject: 
Camera Calibration and Fundamental Matrix Estimation with RANSAC

## Project objectives:
- The goal of this project is to introduce you to camera and scene geometry
- Estimate the camera projection matrix, which maps 3D world coordinates to 2D image coordinates.
- Estimate the fundamental matrix, which relates points in one scene to epipolar lines in another perspective of the same scene. 
- Estimating the fundamental matrix for the correspondences of two images using the RANSAC model-fitting algorithm. 

## Steps to local feature matching between two images (image1 & image 2):
1. Estimating the projection matrix:
=>> calculate_projection_matrix()
=>> calculate_camera_center()
2. Estimating the fundamental matrix: 
=>> estimate_fundamental matrix()
3.	Estimating the fundamental matrix with unreliable ORB matches using RANSAC:
=>> ransac_fundamental_matrix()


# Main files to check
- Project report: I have briefly introduced the objectives of the project, reviewed the image processing methods, explained the main functions, described experiments and discussed the results.

- Jupyter notebook: High level code where inputs are given, main functions are called, results are displayed and saved.

- Student code: Image processing functions are defined.


# Setup by Dr. Kin-Choong Yow
- Install [Miniconda](https://conda.io/miniconda). It doesn't matter whether you use 2.7 or 3.6 because we will create our own environment anyways.
- Create a conda environment using the given file by modifying the following command based on your OS (`linux`, `mac`, or `win`): `conda env create -f environment_<OS>.yml`
- This should create an environment named `ense885ay`. Activate it using the following Windows command: `activate ense885ay` or the following MacOS / Linux command: `source activate ense885ay`.
- Run the notebook using: `jupyter notebook ./code/proj3.ipynb`
- Generate the submission once you're finished using `python zip_submission.py`


# Credits and References
This project has been developed based on the project template and high-level code provided by Dr. Kin-Choong Yow, my instructor for the course “ENSE 885AY: Application of Deep Learning in Computer Vision”.

This course is based on Georgia Tech’s CS 6476 Computer Vision course instructed by James Hays.

- Dr. Kin-Choong Yow page: 
http://uregina.ca/~kyy349/

- “CS 6476 Computer Vision” page:
https://www.cc.gatech.edu/~hays/compvision/

- Project source page at “CS 6476 Computer Vision”:
Not found

- James Hays pages:
https://www.cc.gatech.edu/~hays/
https://github.com/James-Hays?tab=repositories


# My contribution
The following files contain the code written by me:

- code/student_code.py >> calculate_projection_matrix() function
- code/student_code.py >> calculate_camera_center() function
- code/student_code.py >> estimate_fundamental_matrix() function
- code/student_code.py >> normalization_function() function
- code/student_code.py >> ransac_fundamental_matrix() function
- code/student_code.py >> tune_ransac_fundamental_matrix() function
- proj3_tune_threshold.ipynb >> Tuned threshold for ransac_fundamental_matrix()

`### TODO: YOUR CODE HERE ###`

`# My lines of code are inside these comments to be separated from the rest of the code.`

`### END OF STUDENT CODE ####`

______________
Marzieh Zamani