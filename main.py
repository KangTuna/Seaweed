import cv2
import numpy as np
import json

img = cv2.imread('./dataset/train/train_defected_dataset/seaweed_04930.png',cv2.IMREAD_GRAYSCALE)  # 이미지를 불러옵니다

with open('./dataset/train/train_defected_json/seaweed_04930.json', 'r') as f:
    json_dict = json.load(f)

cv2.rectangle(img,(json_dict['top_x'],json_dict['top_y']),(json_dict['bot_x'],json_dict['bot_y']),(255,255,255))
zero = np.zeros((img.shape[0],img.shape[1],1),dtype=np.uint8)

retval, otsu_thresh = cv2.threshold(img, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 오츠 알고리즘으로 얻은 임계값을 기준으로 상한과 하한 설정
lower_thresh = 0.5 * retval  # 하한 임계값: 오츠 임계값의 50%
upper_thresh = retval         # 상한 임계값: 오츠 임계값 자체

# 가우시안 블러처리
blurred_image = cv2.GaussianBlur(img, (5, 5), 0)

# 캐니엣지 검출
canny = cv2.Canny(blurred_image, threshold1=lower_thresh, threshold2=upper_thresh)
sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)  # x축 방향 미분
sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)  # y축 방향 미분
sobel = cv2.magnitude(sobel_x, sobel_y)  # 기울기의 크기를 계산하여 엣지 검출
# prewitt_x = cv2.filter2D(img, -1, np.array([[-1,0,1],[-1,0,1],[-1,0,1]]))  # 수평 방향 프리윗 필터
# prewitt_y = cv2.filter2D(img, -1, np.array([[-1,-1,-1],[0,0,0],[1,1,1]]))  # 수직 방향 프리윗 필터
# prewitt = cv2.magnitude(prewitt_x, prewitt_y)
laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
scharr_x = cv2.Scharr(blurred_image, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(blurred_image, cv2.CV_64F, 0, 1)
scharr = cv2.magnitude(scharr_x, scharr_y)
# roberts_x = cv2.filter2D(blurred_image, -1, np.array([[1, 0], [0, -1]]))  # 수평 필터
# roberts_y = cv2.filter2D(blurred_image, -1, np.array([[0, 1], [-1, 0]]))  # 수직 필터
# roberts = cv2.magnitude(roberts_x, roberts_y)
blur1 = cv2.GaussianBlur(img, (5,5), 1)
blur2 = cv2.GaussianBlur(img, (5,5), 2)
dog = blur1 - blur2

cv2.imshow('img',img)
cv2.imshow('canny',canny)
cv2.imshow('sobel',sobel)
# cv2.imshow('prewitt',prewitt)
cv2.imshow('laplacian',laplacian)
cv2.imshow('scharr',scharr)
# cv2.imshow('roberts',roberts)
cv2.imshow('dog',dog)
cv2.waitKey(0)
cv2.destroyAllWindows()

