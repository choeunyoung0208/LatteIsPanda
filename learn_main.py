# 데이터 학습에 사용되는 라이브러리 불러옴
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# 학습에 필요한 기본 경로 설정
ap = argparse.ArgumentParser()
# dataset이 들어있는 폴더 지정
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
# 학습 결과 이미지를 저장할 이름 설정
ap.add_argument("-p", "--plot", type=str, default="plot.png", #
	help="path to output loss/accuracy plot")
# 학습 결과를 저장할 이름 설정
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to output face mask detector model")
args = vars(ap.parse_args())

# 학습에 사용할 Learning Rate, EPOCHS, Batch Size 초기화
INIT_LR = 1e-4
EPOCHS = 20
BS = 1

# dateset directory에 있는 이미지 리스트 불러옴
# Data list, image class list 초기화
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# 앞에서 지정한 imagePaths에 대하여
for imagePath in imagePaths:
	# 파일 경로중 폴더이름으로 label을 지정
	label = imagePath.split(os.path.sep)[-2]

	# 이미지를 224,224 크기로 불러와 학습에 적절하게 변경
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	# data와 label을 업데이트
	data.append(image)
	labels.append(label)

# data, labels를 numpy를 이용해 array형태로 반환
data = np.array(data, dtype="float32")
labels = np.array(labels)

# labels를 One-Hot encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# 학습을 진행할 때 training data를 80%, test data를 20% 사용
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# 학습 이미지의 다양성을 늘리기 위해 ImageDataGenerator를 이용해 불림
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# 학습에 사용할 네트워크인 MobilenetV2를 불러와 baseModel로 지정
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# baseModel의 output을 통해 다시 영상처리를 진행 (추가적인 네트워크)
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel) # 2가지의 클래스를 분류하므로 마지막은 softmax로

# baseModel(MobilnetV2)의 인풋을 input에, headModel을 output으로 두고
# model 구성
model = Model(inputs=baseModel.input, outputs=headModel)

# 학습중 가중치를 업데이트 하지 않기위하여
# layer.trainable = False로 베이스 모델 고정
for layer in baseModel.layers:
	layer.trainable = False

# 모델 학습 방식 설정
print("[INFO] compiling model...")
# Optimizer Adam
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS) # lr 지정, lr decay 지정
model.compile(loss="binary_crossentropy", optimizer=opt, # 2가지 클래스(Mask, Nomask)이므로 Binary cross entropy 사용
	metrics=["accuracy"])

# 모델 학습
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS, # epoch 당 배치 수
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS) # Epochs 지정

# 학습 모델 평가
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS) # testX를 input으로 학습 모델 평가

# 학습한 모델을 평가하여 나온 값 중 최대값의 위치 Index
predIdxs = np.argmax(predIdxs, axis=1)

# sklearn의 classification_report를 이용하여
# mask와 Nomask에 대하여 학습모델 평가 결과를 출력
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# 학습한 모델을 h5 형식으로 저장
print("[INFO] saving mask detector model...")
model.save(args["model"], save_format="h5")

# 모델 학습중의 값들을 Plot
# Epochs에 대하여
# train_loss, acc, val_loss, acc를 출력
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
