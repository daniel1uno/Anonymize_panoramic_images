# Blur_faces_plates_panoramic_images
Blur faces and licenses plates out of panoramic 360 images, this functions were created to use on large images (example data has 11000 x 5500 px). 

How it works: it takes larges images as input, the function *slice* is used to divided each image into multiple sections,
then, those sections are passed to deep learning models in order to  detect and blur faces and license plates, once the detection has finished the function  *merge* is used to rebuild the original image.

To detect faces retinaface is used: https://github.com/serengil/retinaface.git
To detect licenses plates lpr_caffe_model is used: https://github.com/scorpiochang/Mobilenet-SSD-License-Plate-Detection.git



