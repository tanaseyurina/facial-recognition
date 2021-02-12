import numpy as np
import cv2
import pyvirtualcam
from pngoverlay import PNGOverlay

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

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

cap = cv2.VideoCapture(0)  # カメラの接続が1台の時「0」

# ----main---------------------------------------------- #  

# 透過画像に変換
ang_img=PNGOverlay('0.png')
dis_img=PNGOverlay('1.png')
fea_img=PNGOverlay('2.png')
hap_img=PNGOverlay('3.png')
neu_img=PNGOverlay('4.png')
sad_img=PNGOverlay('5.png')
sur_img=PNGOverlay('6.png')

# start the webcam feed
cap = cv2.VideoCapture(0)

#cap = cv2.VideoCapture(0)  # カメラの接続が1台の時「0」
ret, frame = cap.read()
with pyvirtualcam.Camera(width=frame.shape[1], height=frame.shape[0], fps=30, delay=0) as cam:

    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        recognized_faces = face_cascade_default.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=7)  # scaleFactor=画像スケールにおける縮小量の設定, minNeighbors=顔の信頼度

        # 顔認証
        for (x, y, w, h) in recognized_faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
        
        # fps調整なし
        facial_expression = int(np.argmax(prediction))
        
        print(facial_expression) # 感情の番号表示
        xx=int(x+w/2)
        yy=int(y+h/2)
        z=(w*h/20000)
        #print("面積：",z)

        # ----by_emotional---------------------------------------------- #  
        # "Angry" : 怒っている
        if facial_expression == 0:
            ang_img.resize(z)
            ang_img.show(frame,xx,yy)
        # "Disgusted" : うんざりしている
        elif facial_expression == 1:
            dis_img.resize(z)
            dis_img.show(frame,xx,yy)
        # "Fearful" : 恐れている
        elif facial_expression == 2:
            fea_img.resize(z)
            fea_img.show(frame,xx,yy)
        # "Happy" : 幸せ
        elif facial_expression == 3:
            hap_img.resize(z)
            hap_img.show(frame,xx,yy)
        # "Neutral" : 中立
        elif facial_expression == 4:
            neu_img.resize(z)
            neu_img.show(frame,xx,yy)
        # "Sad" : 悲しい
        elif facial_expression == 5:
            sad_img.resize(z)
            sad_img.show(frame,xx,yy)
        # "Surprised" : 驚いた
        elif facial_expression == 6:
            sur_img.resize(z)
            sur_img.show(frame,xx,yy)
        # ----END by_emotional---------------------------------------------- #  
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        # 画像を仮想カメラに流す
        cam.send(frame)

        # 画像をスクリーンに表示しなくなったので，pyvirtualcamの機能を使って次のフレームまで待機する
        cam.sleep_until_next_frame()
# ----END main---------------------------------------------- #  

cap.release()
cv2.destroyAllWindows()





