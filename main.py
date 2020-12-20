# 구동하는데 필요한 라이브러리 불러옴 (keras, numpy, cv2, datetime, time, playsound)

# 영상처리에 사용되는 라이브러리
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# 시간 관련 라이브러리
import datetime
import time

# LED 제어를 하는데 사용되는 라이브러리
import serial
import socket

# 마스크 미착용시 효과음을 울리기 위해 사용되는 라이브러리
import pygame



# 마스크 미착용시 효과음 울리기
pygame.init()       # 음악 재생하기 위해 필요한 요소 초기화
pygame.mixer.init() # 음악 재생하기 위해 필요한 요소 초기화
Sound = pygame.mixer.Sound('sound2.mp3') # 불러올 mp3 파일 설정 및 인스턴스 생성
Sound.set_volume(0.1)   # 0.0 ~ 1.0 볼륨 조절

HOST = '192.168.43.237' # 서버의 주소 (LED 쪽의 주소)
PORT = 12345 # 서버에서 (우리가 임의로) 지정해 놓은 포트 번호

# 소켓 객체를 생성
# 주소 체계(address family)로 IPv4, 소켓 타입으로 TCP 사용
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 지정한 HOST와 PORT를 사용하여 서버에 접속
client_socket.connect((HOST, PORT))

# facenet : 얼굴을 찾는 모델
# cv2.dnn.readNet : 네트워크 불러오기(네트워크를 불러와서 객체를 생성하기위해 훈련된 가중치와 네트워크 구성을 저장하고 있는 파일이 필요함)
# 얼굴인식할 때만 딥러닝 프레임워크를 카페로 씀
# prototxt : 네트워크 구성을 저장하고 있는 텍스트 파일. network 구조에 대해서 보여주며, 각 레이어별로 파라메타 사이즈 등 상세한 정보가 들어있음 (ref. https://arclab.tistory.com/209)
# caffemodel : model 파일 확장자. 훈련된 가중치를 저장하고 있는 이진 파일 이름
facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
model = load_model('models/mask_detector.model') # 마스크 검출 모델

# 실시간으로 웹캠을 읽기 위해 SBC와 연결된 카메라 장치를 불러옴
cap = cv2.VideoCapture(0)

# 카메라 성능 개선
cap.set(cv2.CAP_PROP_FPS, 30) # FPS 설정
now_fps = cap.get(cv2.CAP_PROP_FPS) # FPS 값 확인
print(now_fps)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080) # 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 사용되는 변수들 초기화
i = 0
tf=''
now = datetime.datetime.now().strftime("%d_%H-%M-%S")
now1 = '000000'

while cap.isOpened(): # 영상 캡쳐 객체 cap이 정상적으로 open되었으면 실행

	# 영상의 한 프레임씩 읽음
	# ret : 프레임을 제대로 읽으면 ret값이 True, 실패하면 False값이 할당됨
	# img : 영상으로부터 읽은 프레임이 할당됨.
	ret, img = cap.read()

	# 영상의 프레임을 제대로 읽지 못 한 경우 프로그램 종료
	if not ret :
		break

	# 영상의 해상도(높이, 폭)를 h, w에 할당함
	h, w = img.shape[:2]

	# 이미지 전처리 과정으로, img를 가지고 4차원의 blob을 만듦 (ref. https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/)
	# mean subtraction : RGB 값의 일부를 제외해서 dnn이 분석하기 쉽게 단순화해주는 것
	# img : blob을 통해 사전 처리하기를 원하는 이미지
	# scalefactor : 평균빼기를 시전한 후 스케일할 값 (R-ur)/a에서 a값
	# size : 공간 크기
	# mean : 평균 빼기 값
	blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))

	# facenet의 input으로 blob을 설정
	facenet.setInput(blob)

	# facenet 결과 추론, 얼굴 추출 결과를 dets에 할당
	dets = facenet.forward()

	# 영상을 복사
	result_img = img.copy()

	# 마스크를 착용했는지 확인
	for i in range(dets.shape[2]):
		confidence = dets[0, 0, i, 2] # 검출한 결과가 신뢰도
		if confidence < 0.5: # 신뢰도를 0.5로 임계치 지정
			continue

		# Bounding box를 구함
		x1 = int(dets[0, 0, i, 3] * w)
		y1 = int(dets[0, 0, i, 4] * h)
		x2 = int(dets[0, 0, i, 5] * w)
		y2 = int(dets[0, 0, i, 6] * h)

		# 원본 영상에서 얼굴 영역을 추출함
		face = img[y1:y2, x1:x2]

		# 추출한 얼굴 영역을 전처리
		try :
			face_input = cv2.resize(face, dsize=(224,224))
			face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
			face_input = preprocess_input(face_input)
			face_input = np.expand_dims(face_input, axis=0)
			mask, nomask = model.predict(face_input).squeeze()

		# 예외처리 구문
		except Exception as e:
			print(str(e))


	# 마스크를 꼈는지 안 꼈는지에 따라 라벨링
		# 마스크를 착용한 확률이 더 클 경우 실행
		if mask > nomask :
			color = (0, 255, 0) # 마스크를 착용한 사람은 초록색으로 사각형을 나타냄(BGR)
			label = "Mask %d%%" % (mask * 100) # 마스크를 착용한 확률을 나타냄
			client_socket.sendall('1'.encode()) # 마스크를 착용했다고 LED에 전송함

		# 마스크를 착용하지 않은 확률이 더 클 경우 실행
		else :
			color = (0, 0, 255) # 마스크를 착용한 사람은 빨간색으로 사각형을 나타냄(BGR)
			label = "No mask %d%%" % (nomask * 100) # 마스크를 착용하지 않은 확률을 나타냄
			now = datetime.datetime.now().strftime("%d_%H-%M-%S") # 현재 시간을 가져옴(캡처한 사진의 이름을 현재 시간으로 저장하기위해 가져옴)

			# 마스크 미착용시 사진 캡처, 효과음을 울리는 것에 약간의 delay를 주기위해 now1, now2 변수 사용
			now1 = int(now1)
			now2 = now[3] + now[4] + now[6] + now[7] + now[9] + now[10]
			now2 = int(now2)

			if (now2 >= now1 + 5) :
				cv2.imwrite("C:/Users/lattepanda/Desktop/data/"+str(now)+".jpg", img) # 캡처한 사진의 이름을 현재 시간으로 하여 저장함
				client_socket.sendall('0'.encode())  # 마스크를 착용하지 않았다고 LED에 전송한다

				# 효과음이 재생되고있지 않을 경우 효과음 재생
				if (pygame.mixer.get_busy() == False):
					Sound.play()
					print("효과음 재생")

				now1 = now[7] + now[9] + now[10]
				int(now1)

		# result_img에 얼굴영역 부분(ROI)을 bounding box로 나타내기위해 설정
		cv2.rectangle(result_img, pt1=(x1,y1), pt2=(x2,y2), thickness=2, color=color, lineType=cv2.LINE_AA)

		# result_img에 label(Mask 또는 No mask)을 나타내기위해 설정
		cv2.putText(result_img, text=label, org=(x1,y1-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
				color=color, thickness=2, lineType=cv2.LINE_AA)

	# result_img에다가 bounding box와 label을 설정한 결과를 보여줌
	cv2.imshow('img',result_img)

	# 'q'를 누르면 프로그램을 종료
	if cv2.waitKey(1) & 0xFF == ord('q') :
		break

# 소켓을 닫음
client_socket.close()