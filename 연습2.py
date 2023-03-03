from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

#yolov8x.pt(최대) --> 잘인식 못함, 속도는 1등 | (실시간 카메라 기준) 많이 느림, 인식 정확도는 좋음
#yolov8n.pt(최저) --> 잘인식 못함, 속도 꼴등  | 가장 빠름, 인식 정확도 매우 안좋음
#yolov8s.pt(최저에서 2번째) --> 그나마 나음   | 중간 속도, 적당한 인식률

model = YOLO("yolov8n.pt")

model.predict(source = "0205.png\KakaoTalk_20230223_162409523.mp4", save=False, conf=0.5, save_txt=False, show=True) #비디오 재생
# model.predict(source = '/Users/kimjunho/Desktop/OpenCV_study/bus.jpg', save=True, conf=0.5, save_txt='bus_yolo.jpg') #사진에 적용후 저장


# results = model.predict(source="0", show=True) #0번 카메라
# print(results)