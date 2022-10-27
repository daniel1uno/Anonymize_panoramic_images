import cv2
import numpy as np
from retinaface import RetinaFace
import json
import os
from PIL import Image
import io


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

    aux = []
    for i in range(0, h_sections):
        image = np.concatenate(
            (images[(i*v_sections):(i*v_sections)+v_sections]), axis=1)
        aux.append(image)
    merged_image = np.concatenate((aux[:]), axis=0)
    return merged_image


def create_logger(out_path):
    with open(os.path.join(out_path, 'log.json'), 'w') as file:
        file_data = {"img_details": []}
        json.dump(file_data, file)
        print("results are being stored in log.json")


def write_json(new_data, out_path):  # this acts as a logger
    with open(os.path.join(out_path, 'log.json'), 'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data["img_details"].append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent=4)


def blur_faces_licenses(in_image, out_path, model, out_folder):

    # Read your image with EXIF data using PIL/Pillow
    imWithEXIF = Image.open(in_image)
    sec_v, sec_h = 7, 6  # this params depends on large of image, adjust to your own necessity

    # get subsections using slice function
    subsections = slice_image(in_image, sec_v, sec_h)
    aux_images = []
    counter_faces = 0
    counter_plates = 0

    for i in range(0, len(subsections)):

        img = subsections[i]
        img_h, img_w = img.shape[:2]

        # face detections
        face_detections = RetinaFace.detect_faces(img, threshold=0.4)

        if type(face_detections) is dict:  # check that detection was successfull

            for face in face_detections:
                start_x, start_y, end_x, end_y = face_detections[face]['facial_area']
                detected_face = img[start_y: end_y, start_x: end_x]
                detected_face = cv2.blur(detected_face, (20, 20))
                img[start_y: end_y, start_x: end_x] = detected_face
                counter_faces += 1

        # license_plates detections
        blob = cv2.dnn.blobFromImage(
            img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        model.setInput(blob)
        output_layer_names = model.getUnconnectedOutLayersNames()
        layer_outputs = model.forward(output_layer_names)
        boxes = []
        confidences = []
        for output in layer_outputs:
            for detection in output:
                confidence = detection[5:]

                if confidence > 0.3:
                    box = detection[0:4] * \
                        np.array([img_w, img_h, img_w, img_h])
                    centerX, centerY, width, height = box.astype("int")
                    x = int(centerX - width / 2)
                    y = int(centerY - height / 2)
                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                coordinates = np.array(
                    [boxes[i][0], boxes[i][1], boxes[i][0]+boxes[i][2], boxes[i][1]+boxes[i][3]])
                coordinates = coordinates.clip(0)
                start_x, start_y, end_x, end_y = coordinates
                #print(start_x, start_y, end_x, end_y)
                plate_detected = img[start_y: end_y, start_x: end_x]
                blur_plate = cv2.blur(plate_detected, (20, 20))
                img[start_y: end_y, start_x: end_x] = blur_plate
                counter_plates += 1

        aux_images.append(img)

    new_image = merge_image(aux_images, sec_v, sec_h)

    log_details = {"image": os.path.basename(in_image),
                   "faces_detected": counter_faces,
                   "licenses_plates_detected": counter_plates
                   }
    write_json(log_details, out_folder)

    #convert BGR to RGB
    im_rgb = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    # Convert OpenCV image onto PIL Image
    OpenCVImageAsPIL = Image.fromarray(im_rgb)

    # Encode newly-created image into memory as JPEG along with EXIF from other image
    OpenCVImageAsPIL.save(os.path.join(out_path, os.path.basename(
        in_image)), exif=imWithEXIF.info['exif'])

    return print('Image ' + os.path.basename(in_image)+'  succesfully processed with '+str(counter_faces)+' faces and ' + str(counter_plates) + ' licenses plates detected')
