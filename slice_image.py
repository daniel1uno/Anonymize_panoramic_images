import cv2
import numpy as np

v_sections = 6 #define de number of vertical sections to slice image
h_sections = 4 #define de number of horizontal sections to slice image

def slice_image(path):
    # cv2.imread() -> takes an image as an input 
    img = cv2.imread(path)
    h, w = img.shape[:2]  #read  size of image

    section_v = w//v_sections  #calculate width of sections
    section_h = h//h_sections  #calculate height of sections
    images = []
    #slice image into h_sections * v_sections 
    for i in range(1,h_sections+1):
        for j in range (1,v_sections+1):
            sliced_image = img[(i*section_h)-section_h:(i*section_h),(j*section_v)-section_v:(j*section_v)]
            images.append(sliced_image)

    return images
    

def merge_image(images):
    #image = images[0]
    aux = []
    for i in range(0,h_sections):
        image = np.concatenate((images[(i*v_sections):(i*v_sections)+v_sections]), axis=1)
        aux.append(image)
    
    merged_image = np.concatenate((aux[:]),axis=0)
    return merged_image
   
    
    

