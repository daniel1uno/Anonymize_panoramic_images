import glob
import ntpath
from functions import blur_faces_licenses, create_logger
from pathlib import Path
import cv2
from datetime import datetime

print('Started at '+ str(datetime.now()))

# paths to custom yolov4 model
pth_weights = r'YoloV4_custom_weights\yolov4-custom_best.weights'
pth_cfg = r'YoloV4_custom_weights\yolov4-custom.cfg'

model = cv2.dnn.readNet(pth_weights, pth_cfg)  # read the model here

# example: data/ <-- dont forget the /
in_path = input(r'Ruta de entrada de las imagenes: ')
# example: results/ <-- dont forget the /
out_path = input(r'Ruta de salida de las imagenes: ')

# this looks fot jpg images, adjust as needed.
images_path = glob.glob(in_path + '/**/*.jpg', recursive=True)
create_logger()

for image in images_path:
    images_out_path = out_path+ntpath.split(image)[0]+"/"
    Path(images_out_path).mkdir(parents=True, exist_ok=True)
    blur_faces_licenses(image, images_out_path, model)

print('Finished at '+ str(datetime.now()))