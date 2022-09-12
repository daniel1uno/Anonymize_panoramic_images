import cv2
import numpy as np
from retinaface import RetinaFace
import ntpath


def read_lpd_model():
    prototxt_path = "weights/license_plate/deploy.prototxt.txt"
    model_path = "weights/license_plate/lpr.caffemodel"
    model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    # load Caffe model
    return model


def slice_image(path, v_sections, h_sections):
    # cv2.imread() -> takes an image as an input
    img = cv2.imread(path)
    h, w = img.shape[:2]  # read  size of image

    section_v = w//v_sections  # calculate width of sections
    section_h = h//h_sections  # calculate height of sections
    images = []
    # slice image into h_sections * v_sections
    for i in range(1, h_sections+1):
        for j in range(1, v_sections+1):
            sliced_image = img[(i*section_h)-section_h:(i*section_h),
                               (j*section_v)-section_v:(j*section_v)]
            images.append(sliced_image)

    return images


def merge_image(images, v_sections, h_sections):
    #image = images[0]
    aux = []
    for i in range(0, h_sections):
        image = np.concatenate(
            (images[(i*v_sections):(i*v_sections)+v_sections]), axis=1)
        aux.append(image)

    merged_image = np.concatenate((aux[:]), axis=0)
    return merged_image


def blur_faces_licenses(in_image, out_path):

    model = read_lpd_model()
    # define number of sections to slice the panoramic image
    v, h = 5, 4

    # get subsections using slice function
    subsections = slice_image(in_image, v, h)

    aux = []
    counter_faces = 0
    counter_plates = 0
    for i in range(0, len(subsections)):
        img = subsections[i]
        he, wi = img.shape[:2]

        detections = RetinaFace.detect_faces(img, threshold=0.4)

        blob = cv2.dnn.blobFromImage(
            img, 0.007843, (he, wi), 127.5, swapRB=True)
        model.setInput(blob)
        output = np.squeeze(model.forward())
        for i in range(0, output.shape[0]):
            confidence = output[i, 2]

            if confidence > 0.40:
                box = output[i, 3:7] * np.array([wi, he, wi, he])
                start_x, start_y, end_x, end_y = box.astype(int)
                plate_detected = img[start_y: end_y, start_x: end_x]
                plate_detected = cv2.blur(plate_detected, (20, 20))
                img[start_y: end_y, start_x: end_x] = plate_detected
                counter_plates += 1

        if type(detections) is dict:  # check that detection was successfull

            for face in detections:
                start_x, start_y, end_x, end_y = detections[face]['facial_area']
                detected_face = img[start_y: end_y, start_x: end_x]
                detected_face = cv2.blur(detected_face, (20, 20))
                img[start_y: end_y, start_x: end_x] = detected_face
                counter_faces += 1
        aux.append(img)
    print('Image ' +
          ntpath.basename(in_image)+'  succesfully processed with '+str(counter_faces)+' faces and ' + str(counter_plates) + ' licenses plates detected')
    new_image = merge_image(aux, v, h)

    return cv2.imwrite(out_path + ntpath.basename(in_image), new_image)
