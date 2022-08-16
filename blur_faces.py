import cv2
import numpy as np

from slice_image import slice_image,merge_image

# https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
prototxt_path = "weights\\deploy.prototxt.txt"
# https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel 
model_path = "weights\\res10_300x300_ssd_iter_140000_fp16.caffemodel"

# load Caffe model
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
# path to image
image_path = "1626541519_0119.jpg"
#define number of horizontal and vertical sections to pass in slice_image function
v_sections = 37 # this was defined applying the formula width/width_expected, width expected for this model is 300 px (ej. 11000/300 =36.6)
h_sections = 18 # this was defined applying the formula height/height_expected, height expected for this model is 300 px (ej. 5500/300 = 18.3)



subsections = slice_image(image_path,v_sections,h_sections)  #get subsections using slice function
aux = []
for i in range(0,len(subsections)):
    image = (subsections[i])
    h, w = image.shape[:2]
    # gaussian blur kernel size depends on width and height of original image 
    kernel_width =  (w // 7) | 1
    kernel_height = (h // 7) | 1
    # preprocess the image: resize and performs mean subtraction 
    blob = cv2.dnn.blobFromImage(image, 1, (h,w), (104,117,124),swapRB=False) #this varies model to model please check model documentation
    # set the image into the input of the neural network
    model.setInput(blob)
    # perform inferencen and get the result
    output = np.squeeze(model.forward())
    for i in range(0, output.shape[0]):
        confidence = output[i, 2]
        # get the confidence
        # if confidence is above 40%, then blur the bounding box (face)
        
        if confidence > 0.3:
            try:
                # get the surrounding box cordinates and upscale them to original image
                box = output[i, 3:7] * np.array([w, h, w, h])
                # convert to integers
                start_x, start_y, end_x, end_y = box.astype(int)
                # get the face image
                face = image[start_y: end_y, start_x: end_x]
                # apply gaussian blur to this face
                face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 0)
                # put the blurred face into the original image
                image[start_y: end_y, start_x: end_x] = face
            except:
                pass     
    aux.append(image)
new_image = merge_image(aux,v_sections,h_sections)

cv2.imwrite("image_blurred.jpg", new_image)
