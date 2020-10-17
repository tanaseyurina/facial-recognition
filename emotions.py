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

import tkinter as tk
TkRoot = tk.Tk()

WINDOW_NAME = "ResizeWindow"

#ウィンドウをフルスクリーンに設定
#cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
#cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

#ディスプレイサイズ取得
display_width = TkRoot.winfo_screenwidth()
display_height = TkRoot.winfo_screenheight()


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
	
	height, width, channels = face_cut_re.shape[:3]  # サイズ取得 #6
	size=(width,height) # 透過画像のresize用

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
		ang_img=cv2.imread('0.png')
		ang_img = cv2.resize(ang_img,size)
		rows, cols, channels = ang_img.shape
		roi = face_cut_re[:rows, :cols]
		ang_gray=cv2.cvtColor(ang_img,cv2.COLOR_BGR2GRAY)
		ang_mask = cv2.threshold(ang_gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
		ang_mask_inv = cv2.bitwise_not(ang_mask)
		face_cut_re_bg = cv2.bitwise_and(roi, roi, mask=ang_mask_inv)
		ang_fg = cv2.bitwise_and(ang_img, ang_img, mask=ang_mask)
		
		ret, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		add_faces=cv2.add(face_cut_re_bg, ang_fg)
		face_cut_re[:rows, :cols] = add_faces
		
		frame=paste_image(img, (x, y, x+w, y+h))

	# "Disgusted" : うんざりしている
	elif facial_expression == 1:
		dis_img=cv2.imread('1.png')
		dis_img = cv2.resize(dis_img,size)
		rows, cols, channels = dis_img.shape
		roi = face_cut_re[:rows, :cols]
		dis_gray=cv2.cvtColor(dis_img,cv2.COLOR_BGR2GRAY)
		dis_mask = cv2.threshold(dis_gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
		dis_mask_inv = cv2.bitwise_not(dis_mask)
		face_cut_re_bg = cv2.bitwise_and(roi, roi, mask=dis_mask_inv)
		dis_fg = cv2.bitwise_and(dis_img, dis_img, mask=dis_mask)
		
		ret, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		add_faces=cv2.add(face_cut_re_bg, dis_fg)
		face_cut_re[:rows, :cols] = add_faces
		
		frame=paste_image(img, (x, y, x+w, y+h))
	
	# "Fearful" : 恐れている
	elif facial_expression ==2:
		fea_img=cv2.imread('2.png')
		fea_img = cv2.resize(fea_img,size)
		rows, cols, channels = fea_img.shape
		roi = face_cut_re[:rows, :cols]
		fea_gray=cv2.cvtColor(fea_img,cv2.COLOR_BGR2GRAY)
		fea_mask = cv2.threshold(fea_gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
		fea_mask_inv = cv2.bitwise_not(fea_mask)
		face_cut_re_bg = cv2.bitwise_and(roi, roi, mask=fea_mask_inv)
		fea_fg = cv2.bitwise_and(fea_img, fea_img, mask=fea_mask)
		
		ret, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		add_faces=cv2.add(face_cut_re_bg, fea_fg)
		face_cut_re[:rows, :cols] = add_faces
		
		frame=paste_image(img, (x, y, x+w, y+h))
	
	# "Happy" : 幸せ
	elif facial_expression ==3:
		hap_img=cv2.imread('3.png')
		hap_img = cv2.resize(hap_img,size)
		rows, cols, channels = hap_img.shape
		roi = face_cut_re[:rows, :cols]
		hap_gray=cv2.cvtColor(hap_img,cv2.COLOR_BGR2GRAY)
		hap_mask = cv2.threshold(hap_gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
		hap_mask_inv = cv2.bitwise_not(hap_mask)
		face_cut_re_bg = cv2.bitwise_and(roi, roi, mask=hap_mask_inv)
		hap_fg = cv2.bitwise_and(hap_img, hap_img, mask=hap_mask)
		
		ret, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		add_faces=cv2.add(face_cut_re_bg, hap_fg)
		face_cut_re[:rows, :cols] = add_faces
		
		frame=paste_image(img, (x, y, x+w, y+h))
	
	# "Neutral" : 中立
	elif facial_expression == 4:
		neu_img=cv2.imread('4.png')
		neu_img = cv2.resize(neu_img,size)
		rows, cols, channels = neu_img.shape
		roi = face_cut_re[:rows, :cols]
		neu_gray=cv2.cvtColor(neu_img,cv2.COLOR_BGR2GRAY)
		neu_mask = cv2.threshold(neu_gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
		neu_mask_inv = cv2.bitwise_not(neu_mask)
		face_cut_re_bg = cv2.bitwise_and(roi, roi, mask=neu_mask_inv)
		neu_fg = cv2.bitwise_and(neu_img, neu_img, mask=neu_mask)
		
		ret, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		add_faces=cv2.add(face_cut_re_bg, neu_fg)
		face_cut_re[:rows, :cols] = add_faces
		
		frame=paste_image(img, (x, y, x+w, y+h))
	
	# "Sad" : 悲しい
	elif facial_expression == 5:
		sad_img=cv2.imread('5.png')
		sad_img = cv2.resize(sad_img,size)
		rows, cols, channels = sad_img.shape
		roi = face_cut_re[:rows, :cols]
		sad_gray=cv2.cvtColor(sad_img,cv2.COLOR_BGR2GRAY)
		sad_mask = cv2.threshold(sad_gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
		sad_mask_inv = cv2.bitwise_not(sad_mask)
		face_cut_re_bg = cv2.bitwise_and(roi, roi, mask=sad_mask_inv)
		sad_fg = cv2.bitwise_and(sad_img, sad_img, mask=sad_mask)
		
		ret, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		add_faces=cv2.add(face_cut_re_bg, sad_fg)
		face_cut_re[:rows, :cols] = add_faces
		
		frame=paste_image(img, (x, y, x+w, y+h))
	
	# "Surprised" : 驚いた
	elif facial_expression == 6:
		sur_img=cv2.imread('6.png')
		sur_img = cv2.resize(sur_img,size)
		rows, cols, channels = sur_img.shape
		roi = face_cut_re[:rows, :cols]
		sur_gray=cv2.cvtColor(sur_img,cv2.COLOR_BGR2GRAY)
		cv2.imwrite("sample_img/sur_gray.png",sur_gray)
		sur_mask = cv2.threshold(sur_gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
		cv2.imwrite("sample_img/sur_mask.png",sur_mask)
		sur_mask_inv = cv2.bitwise_not(sur_mask)
		cv2.imwrite("sample_img/sur_mask_inv.png",sur_mask_inv)
		face_cut_re_bg = cv2.bitwise_and(roi, roi, mask=sur_mask_inv)
		cv2.imwrite("sample_img/face_cut_re_bg.png",face_cut_re_bg)
		sur_fg = cv2.bitwise_and(sur_img, sur_img, mask=sur_mask)
		cv2.imwrite("sample_img/sur_fg.png",sur_fg)

		ret, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		add_faces=cv2.add(face_cut_re_bg, sur_fg)
		cv2.imwrite("sample_img/add_faces.png",add_faces)
		face_cut_re[:rows, :cols] = add_faces

		frame=paste_image(img, (x, y, x+w, y+h))
	# ----END by_emotional---------------------------------------------- #  

	# 'q'で終了
	cv2.imshow('Video', cv2.resize(frame,(display_width,display_height),interpolation = cv2.INTER_CUBIC)) # cv2.imshow()メソッド　OSのフレーム(ウィンドウ)に画像を表示
	if key == ord('q'):
		break
# ----END main---------------------------------------------- #  

# 処理完了時、キャプチャを解放
cap.release()
cv2.destroyAllWindows()
