import sys
import os
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
# import settings
from PIL import Image
# import requests
from io import BytesIO
import Algorithmia
import urllib

# def extract_edge(img):
#     #엣지 추출
#     sobelx = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
#     sobely = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)
#     laplacian = cv2.Laplacian(img, cv2.CV_8U)
#     return sobelx, sobely, laplacian

def cascade(url):
    #위치를 알려줌(여러개 가능)
    input = {
    "input": url,
    "output": ".jpg"
    }
    client = Algorithmia.client('simpwOTVV5icdkd+wRHy1O0ByZC1')
    algo = client.algo('character_recognition/TextDetectionCTPN/0.2.0')
    algo.set_options(timeout=300) # optional
    try:
        print(algo.pipe(input).result)
        result=algo.pipe(input).result['boxes']
        return result
        #{'confidence': 0.9713579416275024, 'x0': 61.44, 'x1': 439.68, 'y0': 229.79373046875, 'y1': 284.44853515625}


    except:
        print(f"글씨가 없다구!! url은 {url}")
        return False
def url_to_image(url):
    #url을 넣으면 image를 리턴해줌

	resp = urllib.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	# return the image
	return image

def detect_face(model, image_url):
    #이미지 url을 넣으면 전처리 다 해서 모델에 들어갈 수 있는 문자상태 여러개 이미지를 보내줌

    #이미지 불러오기
    image_rect = url_to_image(url)


    # 얼굴인식 실행
    # cascade = cv2.CascadeClassifier(cascade_filepath)
    # faces = cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=15, minSize=(64,64))

    #위치 정보
    results = cascade(image_url) #위치가 딕셔너리를 담은 리스트로 리턴
    
    # 얼굴이 1개 이상 검출된 경우
    if len(results)>0:
        print(f"인식된 얼굴의 수: {len(results)}")
        result_list =[]
        result_name = []
        for result in results: 
            xpos = int(result['x0'])
            xpos2 = int(result['x1'])
            ypos = int(result['y0'])
            ypos2 = int(result['y1'])
            print(result['x0'])

            #이미지 크롭하기
            image_crop=image_rect[ypos:ypos2, xpos:xpos2]
            plt.imshow(image_crop,'gray')
            plt.show()
            #모델 넣기전 전처리
            image = np.array(cv2.resize(image_crop, (76,76)))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            laplacian = cv2.Laplacian(image, cv2.CV_8U)
            laplacian = np.expand_dims(laplacian, axis=(0,3))

            #rectangle 만들기
            cv2.rectangle(image_rect,(xpos,ypos), (xpos2 ,ypos2),(255,0,0), thickness=2)
            print(image_rect.shape)
            
            # image = np.expand_dims(np.array(image), axis=0)
            plt.imshow(image_rect)
            plt.show()

            print(image_rect.shape)
            # print(np.expand_dims(image_rect, axis=0).shape)
            
            # face_image = ??
            # 인식한 얼굴에 이름을 표시
            # return 이 labeling인지 확인하자
            name, result_msg = detect_who(model, laplacian)

         
            result_list.append(result_msg)
            
            # cv2.putText(image, name, (xpos, ypos+height+20), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0),2)
    else:
        print("지정한 이미지 파일에서 문자를 인식할 수 없습니다.")

    return result_list, image
        
        # for (xpos, ypos, width, height) in faces: 
        #     face_image = image[ypos:ypos+height, xpos:xpos+width]
        #     print(f"인식한 얼굴의 사이즈: {face_image.shape}")
        #     if face_image.shape[0] < 64 or face_image.shape[1]<64:
        #         print("인식한 얼굴의 사이즈가 너무 작습니다")
        #         continue


def detect_who(model, laplacian):
    # 예측해서 폰트종류 알려주기
    result = model.predict(laplacian) #input, output :array
    name_number_label = np.argmax(result)
    print(f"송혜교일 가능성:{result[0][name_number_label]*100: .3f}%")
    #softmax한 값이 나오기 때문에 *100 
    if name_number_label == 0:
        name = "Baemin"
    elif name_number_label == 1:
        name = "Cafe24"
    elif name_number_label == 2:
        name = "Chosun"
    elif name_number_label == 3:
        name = "Mapo"
    elif name_number_label == 4:
        name = "Nanumpen"
    elif name_number_label == 5:
        name = "NanumSquare"
    return (name, result_msg)




RETURN_SUCCESS = 0
RETURN_FAILURE = -1
# Inpou Model Directory
INPUT_MODEL_PATH = "./model/model_laplacian.h5"


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
    # image = cv2.imread(image_file_path)
    # image = Image.open(image_file_path)
    # image = image.resize((200, 200))
    

    # 모델 파일 읽기
    if not os.path.exists(INPUT_MODEL_PATH):
        print("MODEL 파일이 존재하지 않습니다.")
        return RETURN_FAILURE
    
    model = keras.models.load_model(INPUT_MODEL_PATH)

    # url 받아서 crop된 이미지
    image =detect_face(model, image_url)
    plt.imshow(result_image)
    plt.show()

    # 에러나는경우 어떻게 되는지는 다시 생각해보자
    
    if image is None:
        print(f"이미지 파일을 읽을 수 없습니다({image_file_path})")
        return RETURN_FAILURE

    # 얼굴인식
    result_image = detect_face(model, image_url)


    return RETURN_SUCCESS



if __name__ == "__main__":
    main()