# Function: bicubic
# Author: YuYao
# Time: 10/09/2019

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


if __name__ == '__main__':

    scale = 4
    path = "/home/ubuntu/all_data/benchmark/manga109/HR/"
    path2 = "/home/ubuntu/all_data/benchmark/manga109/HR/"
    for filename in os. listdir(path):
        if os.path.splitext(filename)[1] =='.png':
            print(filename)
            img = cv2.imread(path+filename)
            new_img = cv2.resize(img, (828,1170),interpolation=cv2.INTER_CUBIC)

            cv2.imwrite( path2+filename, new_img)
    print("Finish")
