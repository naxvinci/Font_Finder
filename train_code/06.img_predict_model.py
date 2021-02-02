import sys
import os
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import settings
from PIL import Image


import Algorithmia

RETURN_SUCCESS = 0
RETURN_FAILURE = -1
# Inpou Model Directory
INPUT_MODEL_PATH = "./model/model_laplacian.h5"
url = "https://postfiles.pstatic.net/MjAyMDA0MDlfMTQx/MDAxNTg2NDA3MTg5NDc1.Kuc8FIqt1-LpjVjCzUwengPd8GlqHPKkWG0Jd1W5avEg.82sPT_dhaFTCF0TQKnsNBb26fn-tWzbPcjMuehAEtwEg.JPEG.cancel9/image1.jpg"

def crop_and_rec(url):
    input = {
    "input": url,
    "output": ".jpg"
    }
    client = Algorithmia.client('simpwOTVV5icdkd+wRHy1O0ByZC1')
    algo = client.algo('character_recognition/TextDetectionCTPN/0.2.0')
    algo.set_options(timeout=300) # optional
    try:
        print(algo.pipe(input).result)
        result=algo.pipe(input).result['boxes'][0]
        #{'confidence': 0.9713579416275024, 'x0': 61.44, 'x1': 439.68, 'y0': 229.79373046875, 'y1': 284.44853515625}
        xpos = int(result['x0'])
        xpos2 = int(result['x1'])
        ypos = int(result['y0'])
        ypos2 = int(result['y1'])
        
        print(result['x0'])

        
    except:
        print(f"글씨가 없다구!! url은 {url}")
        return False


    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image_rect = np.array(image)

    #이미지 크롭하기
    image_crop=image_rect[ypos:ypos2, xpos:xpos2]
    plt.imshow(image_crop,'gray')
    plt.show()

    #크롭한 이미지를 모델에 넣기(Predict)
    print(image_crop.shape)
    # input shape은 (1,76,76,1) 이렇게
    # print(np.expand_dims(face_image, axis=0).shape)
    # face_image = np.expand_dims(face_image, axis=0)
    # # face_image = ??
    # # 인식한 얼굴에 이름을 표시
    # name = detect_who(model, face_image)
    # cv2.putText(image, name, (xpos, ypos+height+20), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0),2)

    #rectangle 만들기
    cv2.rectangle(image_rect,(xpos,ypos), (xpos2 ,ypos2),(255,0,0), thickness=2)
    # 인식한 얼굴을 1장의 사진으로 합치고 -> 배열 변환
    print(np.expand_dims(image_rect, axis=0).shape)
    image_ready = np.expand_dims(image_rect, axis=0)
    print(image_ready.shape)
    # image = np.expand_dims(np.array(image), axis=0)
    plt.imshow(image_rect)
    plt.show()



def main():
    print("===================================================================")
    print("Keras를 이용한 얼굴인식")
    print("학습 모델과 지정한 이미지 파일을 기본으로 연예인 구분하기")
    print("===================================================================")

    # 인수 체크
    # TO-DO

    argvs = sys.argv
    if len(argvs) !=2 or not os.path.exists(argvs[1]):
        print("이미지 파일을 지정해주세요")
        return RETURN_FAILURE
    image_file_path = argvs[1]

    #이미지 파일 읽기
    # image = cv2.imread(image_file_pa6th)
    image = Image.open(image_file_path)
    image = image.resize((200, 200))
    plt.imshow(image,'gray')
    plt.show()
    print("image")
    print(image)
    if image is None:
        print(f"이미지 파일을 읽을 수 없습니다({image_file_path})")
        return RETURN_FAILURE
    

    # 모델 파일 읽기
    if not os.path.exists(INPUT_MODEL_PATH):
        print("MODEL 파일이 존재하지 않습니다.")
        return RETURN_FAILURE
    
    model = keras.models.load_model(INPUT_MODEL_PATH)

    # 얼굴인식
    # cascade_filepath = settings.CASCADE_FILE_PATH
    # result_image = detect_face(model, cascade_filepath, image)
    image = np.expand_dims(image, axis=0)

    result_image = model.predict(image).reshape((200,200))
    print("after predict")
    print(result_image)
    # result_image = cv2.cvtColor(result_image,cv2.COLOR_BGR2GRAY)
    plt.imshow(result_image,'gray')
    plt.show()

    return RETURN_SUCCESS



if __name__ == "__main__":
    main()