#https://sites.google.com/site/lifeslash7830/home/hua-xiang-chu-li/opencvniyoruhuaxiangchulihuaxiangnohechengtoka?tmpl=%2Fsystem%2Fapp%2Ftemplates%2Fprint%2F&showPrintDialog=1
#C:\Users\tyuri\Desktop\new\transparent_image.py
import cv2
import numpy as np

img1 = cv2.imread("irasutoya.png")
height, width, channels = img1.shape[:3] #背景サイズ取得
#print(str(height))
#print(str(width))
size=(width,height) #size変更の記録
img2 = cv2.imread("6.png")
img2 = cv2.resize(img2,size) #sizeにサイズ変更
#print(img1.shape)
#print(img2.shape)
#透過画像を入れたいので、ROIを作成します
rows, cols, channels = img2.shape #行、列取得
roi = img1[:rows, :cols]
#ロゴのマスクを作成し、その逆マスクも作成します
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
mask_inv = cv2.bitwise_not(mask)
#ROIのロゴの領域を黒く塗りつぶします
img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
#ロゴ画像からロゴの領域のみを取得します。
img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
#ロゴをROIに入れ、メイン画像を変更します
dst = cv2.add(img1_bg, img2_fg)
img1[:rows, :cols] = dst

cv2.imwrite('result.png', img1)

#ROIがつくられていない
