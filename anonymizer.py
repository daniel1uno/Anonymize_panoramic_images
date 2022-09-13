import glob
import ntpath
from functions import blur_faces_licenses
from pathlib import Path
from functions import create_logger

in_path = input('Ruta de entrada de las imagenes: ')  # example: data/ <-- dont forget the /
out_path = input('Ruta de salida de las imagenes: ')  # example: results/ <-- dont forget the /

images_path = glob.glob(in_path + '/**/*.jpg', recursive=True)
create_logger()

for image in images_path:
    images_out_path = out_path+ntpath.split(image)[0]+"/"
    Path(images_out_path).mkdir(parents=True, exist_ok=True)
    blur_faces_licenses(image, images_out_path)
