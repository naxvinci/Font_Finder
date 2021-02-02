# import sys
# import os
import cv2
# import keras
import numpy as np
import matplotlib.pyplot as plt
# import settings
from PIL import Image
# import pandas as pd

import requests
from io import BytesIO
# import urllib
import pandas as pd


# 폰트 이름과 폰트 class 정하기
fontname = 'kor.hygungso-bold.exp0' #'kor.yetr.exp0'

font_class_num = 2
# tif 이미지 불러오기
image = Image.open(f'./box/{fontname}.tif') #but 1페이지만 가져온다능...

print(image.size)
# plt.imshow(image, 'gray')
# plt.show()

# 문자
f = open(f"./box/{fontname}.box", 'r', encoding='utf-8')
boxes = f.readlines()
# print(boxes[:1])
f.close()


letter_list = []

# 문자 리스트
for i in boxes:
    letter, x_lef, width, y_top, height, _ = i.split()
    letter_list.append(letter)

letter_arr = np.array(letter_list).reshape((-1,4))

## crop 하기

#1. array로 변환하여 크롭 - 근데 잘 안맞는다
# image = np.array(image)
# image_crop=image[109: 109+100,  200: 200+100] # 각이 나온다
# image_crop=image[ 0: 109,  0: 1+100] # 안나온다
# plt.imshow(image_crop,'gray')
# plt.show()

## 2. PIL로 crop 하기
## 와 이것도 위치가 안맞는다
# = (start_x, start_y, start_x + width, start_y + height) 
# = (left, upper, right, lower) 
# = (가로 시작점, 세로 시작점, 가로 범위, 세로 범위)

# 총 26 줄
# w_count = 3
# h_count = 2
# one_width = 110
# one_height = 123
# image_crop = image.crop(((w_count)*one_width+100,(h_count)*one_height+100, (w_count)*one_width+200 , (h_count)*one_height +200))

# plt.imshow(image_crop,'gray')
# print("문자는:", letter_arr[h_count,w_count]) ##arr 은 row, col 형식이여서 반대로 slice 해야한다
# plt.title(f"x:{w_count+1}, y:{h_count+1}")
# # plt.imshow(image,'gray')
# plt.show()

startx, starty = 100, 100
one_width = 110
one_height = 118
for w_count in range(4):
    for h_count in range(26):
        image_crop = image.crop(((w_count)*one_width+startx,(h_count)*one_height+starty, (w_count)*one_width+startx+100 , (h_count)*one_height +starty+100))
        # save a image using extension 
        # font-stype , name, version(0:original, 1: move x-axis, 2:move y-axis, 3: rotate30, 4:rotate-30)
        image_crop.save(f"./letters/{font_class_num}_{letter_arr[h_count,w_count]}_0.jpg") 


## 평행이동

def move(startx, starty, version_num):
    startx, starty = 100+startx, 100+starty
    one_width = 110
    one_height = 118
    for w_count in range(4):
        for h_count in range(26):
            image_crop = image.crop(((w_count)*one_width+startx,(h_count)*one_height+starty, (w_count)*one_width+startx+100 , (h_count)*one_height +starty+100))
            # save a image using extension 
            # font-stype , name, version(0:original, 1: move x-axis, 2:move y-axis, 3: rotate30, 4:rotate-30)
            image_crop.save(f"./letters/{font_class_num}_{letter_arr[h_count,w_count]}_{version_num}.jpg") 
    return True

def rotate(angle, version_num):
    startx, starty = 100, 100
    one_width = 110
    one_height = 118
    for w_count in range(4):
        for h_count in range(26):
            white_layer = image.crop((1000,1000,1100,1100)).convert('RGBA')
            image_crop = image.crop(((w_count)*one_width+startx,(h_count)*one_height+starty, (w_count)*one_width+startx+100 , (h_count)*one_height +starty+100))
            # converted to have an alpha layer 
            image_crop=image_crop.convert('RGBA')

            # rotated image 
            rot = image_crop.resize((90,90)).rotate(angle, expand=1) 

            white_layer.paste(rot, (-10, -10), rot )

            white_layer.convert(mode ='RGB').save(f"./letters/{font_class_num}_{letter_arr[h_count,w_count]}_{version_num}.jpg") 
        


move(10,0,1)
move(10,10,2)
rotate(-10,3)
rotate(22.2,4)

    
