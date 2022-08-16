# Blur_faces_plates_panoramic_images
Blur faces and licenses plates out of panoramic 360 images, this functions were created to use on large images (example data has 11000 x 5500 px). 

How it works: it takes larges images as input, the function *slice* is used to divided each image into multiple sections,
then, those sections are passed to coffe models for face and license plates detection, once the detection has finished, a 
gaussian blur is applied on each section, finally, another function is used to rebuild the original image.
