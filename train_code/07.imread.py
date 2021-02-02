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

import Algorithmia
# url = "https://blog.cheil.com/wp-content/themes/wpmtv-cheil-blog/img/magazine-image.png" 
# url = "https://www.koreafont.com/wp-content/uploads/2017/03/slide1.png" ##한글 그림
# url = "https://user-images.githubusercontent.com/56625356/79940700-6d7f1b80-849d-11ea-80cf-fec6f408c607.PNG"

# url = "https://user-images.githubusercontent.com/56625356/79943203-44fa2000-84a3-11ea-837a-7e57483e0aa8.png"

# url = "https://postfiles.pstatic.net/MjAyMDA0MDlfMTQx/MDAxNTg2NDA3MTg5NDc1.Kuc8FIqt1-LpjVjCzUwengPd8GlqHPKkWG0Jd1W5avEg.82sPT_dhaFTCF0TQKnsNBb26fn-tWzbPcjMuehAEtwEg.JPEG.cancel9/image1.jpg"

# url = "https://user-images.githubusercontent.com/56625356/79943203-44fa2000-84a3-11ea-837a-7e57483e0aa8.png"

url = "https://postfiles.pstatic.net/MjAyMDA0MDlfMTQx/MDAxNTg2NDA3MTg5NDc1.Kuc8FIqt1-LpjVjCzUwengPd8GlqHPKkWG0Jd1W5avEg.82sPT_dhaFTCF0TQKnsNBb26fn-tWzbPcjMuehAEtwEg.JPEG.cancel9/image1.jpg"



# slack_img= 'https://files.slack.com/files-pri/TRSS2QKK8-F011V5LSWQ7/two_oris.png'

def crop_and_rec(url):
    input = {
    "input": url,
    "output": "data://.algo/character_recognition/TextDetectionCTPN/temp/receipt.png"
    }
    client = Algorithmia.client('simpwOTVV5icdkd+wRHy1O0ByZC1')
    algo = client.algo('character_recognition/TextDetectionCTPN/0.2.0')
    algo.set_options(timeout=300) # optional
    try:
        print(algo.pipe(input).result)

        results=algo.pipe(input).result['boxes']

        for result in results:
        #{'confidence': 0.9713579416275024, 'x0': 61.44, 'x1': 439.68, 'y0': 229.79373046875, 'y1': 284.44853515625}
            xpos = int(result['x0'])
            xpos2 = int(result['x1'])
            ypos = int(result['y0'])
            ypos2 = int(result['y1'])
        
            print(xpos, xpos2, ypos, ypos2)
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            image_rect = np.array(image)

            #이미지 크롭하기
            image_crop=image_rect[ypos:ypos2, xpos:xpos2]
            plt.imshow(image_crop,'gray')
            plt.show()

            #rectangle 만들기
            cv2.rectangle(image_rect,(xpos,ypos), (xpos2 ,ypos2),(255,0,0), thickness=2)
            print(image_rect.shape)
            # image = np.expand_dims(np.array(image), axis=0)
            plt.imshow(image_rect)
            plt.show()
    except:
        print(f"글씨가 없다구!! url은 {url}")
        return False


        


crop_and_rec(url)
# for url in urls:
#     crop_and_rec(url)



# image = Image.open( "./dataset/samples/soybean3.jpg")
# image = image.resize((200, 200))
# print(np.array(image).shape)
# plt.imshow(image,'gray')
# plt.show()
# image_array = np.array(image)[:,:,0].reshape((200,200))
# np.savetxt('test.txt', image_array)
# print("image")
# print(image)