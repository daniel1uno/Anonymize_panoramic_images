import os
from functions import blur_faces_licenses


in_path = input('Ruta de entrada de las imagenes: ') #example data/
out_path = input('Ruta de salida de las imagenes: ') #example results/

listOfFiles = os.listdir(in_path)

for image in listOfFiles:
    blur_faces_licenses(in_path+image, out_path)
