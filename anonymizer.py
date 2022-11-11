import glob
from functions import blur_faces_licenses, create_logger
from pathlib import Path
import cv2
from datetime import datetime
import os

print('Started at ' + str(datetime.now()))
print(cv2.getBuildInformation())

# paths to custom yolov4 model
pth_weights = os.path.normpath(
    r"YoloV4_custom_weights/yolov4-custom_best.weights")
pth_cfg = os.path.normpath(r"YoloV4_custom_weights/yolov4-custom.cfg")

model = cv2.dnn.readNet(pth_weights, pth_cfg)  # read the model here
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) #allow inference using GPU
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16) #allow inference using GPU

# example: data
in_path = os.path.normpath(input("Ruta de entrada de las imagenes :"))
in_folder = os.path.basename(in_path)


# example: results
out_path = os.path.normpath(input("Ruta de salida de las imagenes :"))

# this looks fot jpg images, adjust as needed.
images_path = glob.glob(os.path.join(in_path, '**', '*.jpg'), recursive=True)

create_logger(out_path)

for image in images_path:
    subfolder = os.path.dirname(image.split(in_folder)[-1])

    images_out_path = out_path+subfolder+"/"

    if not os.path.exists(os.path.join(images_out_path, os.path.basename(image))):

        Path(images_out_path).mkdir(parents=True, exist_ok=True)
        blur_faces_licenses(image, images_out_path, model,out_path)

    else:
        print('Image ' + os.path.basename(image) + ' already exists')


print('Finished at ' + str(datetime.now()))
