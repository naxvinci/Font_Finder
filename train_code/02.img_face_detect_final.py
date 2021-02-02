#-*- coding: utf-8 -*- 
import os
import pathlib
import glob
import cv2
import settings
from PIL import Image
import tifffile as tiff
def load_name_images(image_path_pattern):
    name_images = []
    # 지정한 Path Pattern에 일치하는 파일 얻기
    image_paths = glob.glob(image_path_pattern)
    # 파일별로 읽기
    for image_path in image_paths:
        path = pathlib.Path(image_path)
        # 파일 경로
        fullpath = str(path.resolve())
        print(f"이미지 파일(절대경로):{fullpath}")
        # 파일명
        filename = path.name
        print(f"이미지파일(파일명):{filename}")
        # 이미지 읽기

        image = Image.open(fullpath)

        # image = cv2.imread(fullpath)
        if image is None:
            print(f"이미지파일({fullpath})을 읽을 수 없습니다.")
            continue
        name_images.append((filename, image))
        
    return name_images

def detect_image_face(file_path, image):
   
    image=image.resize((200, 200))
    print(image)
    path = pathlib.Path(file_path)
    directory = str(path.parent.resolve())
    filename = path.stem
    # extension = path.suffix
    face_count=1
    # output_path = os.path.join(directory,f"{filename}_{face_count:03}.jpg")
    output_path = os.path.join(directory,f"{face_count:02}_{filename}.jpg")
    print(f"출력파일(절대경로): {output_path}")
    image.save(output_path) 
    # tiff.imwrite(output_path, image)#이미지파일 생성
    face_count = face_count + 1

    return RETURN_SUCCESS

def delete_dir(dir_path, is_delete_top_dir=True):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    if is_delete_top_dir:
        os.rmdir(dir_path)

RETURN_SUCCESS = 0
RETURN_FAILURE = -1
# Origin Image Pattern
IMAGE_PATH_PATTERN = "./dataset/grass/*"
# Output Directory
OUTPUT_IMAGE_DIR = "./dataset/grass_output"

def main():
    print("===================================================================")
    print("이미지 얼굴인식 OpenCV 이용")
    print("지정한 이미지 파일의 정면얼굴을 인식하고, 64x64 사이즈로 변경")
    print("===================================================================")

    # 디렉토리 작성
    if not os.path.isdir(OUTPUT_IMAGE_DIR):
        os.mkdir(OUTPUT_IMAGE_DIR)
    # 디렉토리 내의 파일 제거
    delete_dir(OUTPUT_IMAGE_DIR, False)

    # 이미지 파일 읽기
    # TO-DO 
    name_images =load_name_images(IMAGE_PATH_PATTERN)
    # 이미지별로 얼굴인식
    
    for name_image in name_images:
        file_path = os.path.join(OUTPUT_IMAGE_DIR,f"{name_image[0]}")
        image = name_image[1]
      
        detect_image_face(file_path, image)

    return RETURN_SUCCESS

if __name__ == "__main__":
    main()