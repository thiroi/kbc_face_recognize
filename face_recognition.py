import cv2
import os
import io
import numpy as np
from PIL import Image

# 評価する画像のpath
test_path = 'face.png'

# Haar-like特徴分類器
cascadePath = "./data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# 顔認識器の構築 for OpenCV 2
# EigenFace
#recognizer = cv2.face.EigenFaceRecognizer_create()
# FisherFace
#recognizer = cv2.face.FisherFaceRecognizer_create()
# LBPH

# 指定されたpath内の画像を取得


def preprocessing_image(image_path):
    # 画像を格納する配列
    images = []
    # ファイル名を格納する配列
    file = []
    # ラベルを格納する配列
    label = []
    # グレースケールで画像を読み込む
    image_pil = Image.open(image_path).convert('L')
    # NumPyの配列に格納
    image = np.array(image_pil, 'uint8')
    # Haar-like特徴分類器で顔を検知
    faces = faceCascade.detectMultiScale(image)
    # 検出した顔画像の処理
    for (x, y, w, h) in faces:
        # 顔を 200x200 サイズにリサイズ
        roi = cv2.resize(image[y: y + h, x: x + w],
                         (200, 200), interpolation=cv2.INTER_LINEAR)
        # 画像を配列に格納
        images.append(roi)
        # ファイル名からラベルを取得
        label.append(faces[0:4])
        # ファイル名を配列に格納
        file.append(faces)

    return images, label, file


recognizer = cv2.face.LBPHFaceRecognizer_create()


def doIt():
    #  学習済みモデルの読み込み
    recognizer.read('trainingData.yml')
    # テスト画像を取得
    test_images, test_labels, test_files = preprocessing_image(test_path)
    i = 0
    while i < len(test_labels):
        # テスト画像に対して予測実施
        label, confidence = recognizer.predict(test_images[i])
        # 予測結果をコンソール出力
        print("LABEL TEST")
        print(label)
        i += 1
    return label
