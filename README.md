**

## Blur_faces_plates_panoramic_images

**
Blur faces and licenses plates out of panoramic 360 images, this functions were created to use on large images (example data has 11000 x 5500 px) obtained from mobile mapping.

How it works: it takes larges images as input, the function *slice* is used to divided each image into multiple sections,

then, those sections are passed to deep learning models in order to detect and blur faces and license plates, once the detection has finished the function *merge* is used to rebuild the original image.

To detect faces retinaface is used: https://github.com/serengil/retinaface.git
To detect License plates a yolov4 detector was trained, the training dataset is composed of many images of colombian license plates. keep in mind this model was specifically trained for panoramic images
(usually license plates are very low resolution on this kind of photos) weights can be downloaded from here https://drive.google.com/drive/folders/1QWFvzb9p7KGFKpu3VcCRgD4R1X67-1_l?usp=sharing

To detect licenses pla

 #Structure:

 - data folder contains example data
 - results folder store the processed images
 - YoloV4_custom_weights folder store the custom weights for myolov4 detector trained 
 - functions.py contains all the functions created to run
 - anonymizer.py start the program

 
#How to use:  

 - Install requirements
 - Modify blur_faces_licenses function in functions.py to adjust to your own necessity 
 - Run anonymizer.py, the program will ask for **input path** of images and **output path** for
   processed images

There's also a logger built to store results of the processing

python> 3.5 is required
