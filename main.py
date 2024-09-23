import cv2
import pandas as pd
import json
import os

img = cv2.imread('./dataset/train/train/train_defected_dataset\seaweed_01251.png',cv2.IMREAD_GRAYSCALE)
ret, threshold = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#오츠의 경우 자동으로 임계값을 찾으므로 그냥 초기 임계값을 -1로 지정함.
ret, otsu = cv2.threshold(img,-1,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
#오츠 알고리즘으로 자동으로 지정된 임계값을 출력
print('otsu threshold:',ret)
                                            
cv2.rectangle(img,(200,354),(221,363),(255,255,255))
cv2.rectangle(threshold,(200,354),(221,363),(255,255,255))
cv2.rectangle(otsu,(200,354),(221,363),(255,255,255))

# cv2.imshow('seawead',img)
# cv2.imwrite('./string_type.png',img)
cv2.imshow('threshold', threshold)
cv2.imshow('otsu', otsu)
cv2.imshow('img', img)
cv2.waitKey(0)