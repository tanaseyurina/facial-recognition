"""
< 1010.1630 >
eye消去 alt消去 tab調整 maxindexからfacial_expressionに変更

課題：
　現実と顔の遅延が酷い　⇒　難しい
　モザイクが機能しない　⇒　使わないかも

次：
	画像貼り付け関数、モザイク関数の理解 
	全角スペースの色付け ⇒　無理(´;ω;｀)

< 10130110 >
大きなくくりにコメント
透過画像の合成サンプル（1013110/transparent_image.py 旧sample.py）
ファイルの名前変更。モデル消去

課題：
	画像貼り付け関数、モザイク関数の理解 
	全角スペースの色付け ⇒　無理(´;ω;｀)

次：
	透過画像を合成

< 10130247 >
プログラム中のいらないもの消去
モザイク消去
コメントも一部を覗いてきれいにした
date/model.h5　を　model/model.h5		に変更
modelファイルの中はmodel.h5とhaarcascade_frontalface_default.xmlのみ
ユーザの顔写真はfaceファイルにまとめた
まだaddewightedで画像は処理している

次：
	透過画像を合成させるように入れ込む
"""

import numpy as np
import argparse
import cv2
# import time # FPS調整ありの時、on

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
from PIL import Image

# ----表情認識---------------------------------------------- #  
#  モデルを作成する  // Create the model //
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# 感情をmodelより取得  // emotions will be displayed on your face from the webcam feed // 
model.load_weights('model/model.h5')

# openCLの使用と不要なロギングメッセージを防ぐ  // prevents openCL usage and unnecessary logging messages // 
cv2.ocl.setUseOpenCL(False)

# 感情を割り当て（アルファベット順） // dictionary which assigns each label an emotion (alphabetical order) // 
emotion = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# カスケードファイル呼び出し
face_cascade_default = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
# ----END 表情認識---------------------------------------------- #

# 画像貼り付け関数　paste_image 変更あり！！！！コメントきたないい！！
def paste_image(img, rect):
  (x_tl, y_tl, x_br, y_br) = rect
  wid = x_br - x_tl
  hei = y_br - y_tl
  img_face = cv2.resize(add_faces, (wid, hei)) # add_facesをリサイズ
  paste_img = img.copy() # クローンを生成する
  paste_img[y_tl:y_br, x_tl:x_br] = img_face # (y_tl:y_br)　y軸の縦の長さy_tlからy_brまで, (x_tl:x_br)　x軸の縦の長さx_tlからx_brまでを貼り付け　img_faceが貼り付けた画像 # https://www.qoosky.io/techs/b28ffe314d
  return paste_img
# ----END 画像貼り付け---------------------------------------------- #

cap = cv2.VideoCapture(0)  # カメラの接続が1台の時「0」

# ----main---------------------------------------------- #  
while True:
	ret, frame = cap.read()  # カメラから1コマのデータを取得する
	if not ret:
		break

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	recognized_faces = face_cascade_default.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=7)  # scaleFactor=画像スケールにおける縮小量の設定, minNeighbors=顔の信頼度

	# face_cut
	cv2.imwrite("face/user_face.png",frame)
	user_faces=cv2.imread("face/user_face.png")
	user_faces_gray=cv2.cvtColor(user_faces, cv2.COLOR_BGR2GRAY)
	faces=face_cascade_default.detectMultiScale(user_faces_gray)
	for x,y,w,h in faces:
		face_cut=user_faces[y:y+h,x:x+w]
	cv2.imwrite("face/face_cut_user.png",face_cut)
	face_cut_re = cv2.imread('face/face_cut_user.png')
	
	height, width, channels = face_cut_re.shape[:3]  # サイズ取得

	key=cv2.waitKey(1)&0xFF
	# 顔認証
	for (x, y, w, h) in recognized_faces:
		roi_gray = gray[y:y + h, x:x + w]
		cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
		prediction = model.predict(cropped_img)

	# fps調整あり
	"""
	for i in range(0, 1, 1):
		time.sleep(0.28)
		facial_expression = int(np.argmax(prediction))
	"""
	# fps調整なし
	facial_expression = int(np.argmax(prediction))
	
	print(facial_expression) # 感情の番号表示
	
	# ----by_emotional---------------------------------------------- #  
	# "Angry" : 怒っている
	if facial_expression == 0:
		input_img = cv2.imread('0.png')
		input_img_re = cv2.resize(input_img,(height, width))
		ret, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		add_faces=cv2.addWeighted(input_img_re,0.2,face_cut_re,0.8,0.0)

		frame=paste_image(img, (x, y, x+w, y+h))

	# "Disgusted" : うんざりしている
	elif facial_expression == 1:
		input_img = cv2.imread('1.png')
		input_img_re = cv2.resize(input_img,(height, width))
		ret, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		add_faces=cv2.addWeighted(input_img_re,0.2,face_cut_re,0.8,0.0)

		frame=paste_image(img, (x, y, x+w, y+h))
	
	# "Fearful" : 恐れている
	elif facial_expression ==2:
		input_img = cv2.imread('2.png')
		input_img_re = cv2.resize(input_img,(height, width))
		ret, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		add_faces=cv2.addWeighted(input_img_re,0.2,face_cut_re,0.8,0.0)

		frame=paste_image(img, (x, y, x+w, y+h))
	
	# "Happy" : 幸せ
	elif facial_expression ==3:
		input_img = cv2.imread('3.png')
		input_img_re = cv2.resize(input_img,(height, width))
		ret, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		add_faces=cv2.addWeighted(input_img_re,0.2,face_cut_re,0.8,0.0)

		frame=paste_image(img, (x, y, x+w, y+h))
	
	# "Neutral" : 中立
	elif facial_expression == 4:
		input_img = cv2.imread('4.png')
		input_img_re = cv2.resize(input_img,(height, width))
		ret, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		add_faces=cv2.addWeighted(input_img_re,0.2,face_cut_re,0.8,0.0)

		frame=paste_image(img, (x, y, x+w, y+h))
	
	# "Sad" : 悲しい
	elif facial_expression == 5:
		input_img = cv2.imread('5.png')
		input_img_re = cv2.resize(input_img,(height, width))
		ret, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		add_faces=cv2.addWeighted(input_img_re,0.2,face_cut_re,0.8,0.0)

		frame=paste_image(img, (x, y, x+w, y+h))
	
	# "Surprised" : 驚いた
	elif facial_expression == 6:
		input_img = cv2.imread('6.png')
		input_img_re = cv2.resize(input_img,(height, width))
		ret, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		add_faces=cv2.addWeighted(input_img_re,0.2,face_cut_re,0.8,0.0)

		frame=paste_image(img, (x, y, x+w, y+h))
	# ----END by_emotional---------------------------------------------- #  

	# 'q'で終了
	cv2.imshow('Video', cv2.resize(frame,(600,460),interpolation = cv2.INTER_CUBIC)) # cv2.imshow()メソッド　OSのフレーム(ウィンドウ)に画像を表示
	if key == ord('q'):
		break
# ----END main---------------------------------------------- #  

# 処理完了時、キャプチャを解放
cap.release()
cv2.destroyAllWindows()
