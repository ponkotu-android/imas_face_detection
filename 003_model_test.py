# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 00:58:59 2018

@author: ponkotu_androido
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import cv2
import glob
import os
from PIL import Image, ImageDraw, ImageFont

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.optimizers import RMSprop

from make_model import mk_model


# テストファイルの指定
f='test_data/renamed/0.jpg'

size = 100
idol = {1:'honda_mio',
        2:'tada_riina'}
inp_size = size
out_size = 2

model = load_model('model/derepa004.h5')
#%%
classifier = cv2.CascadeClassifier('etc/lbpcascade_animeface.xml')
faces = []

output_dir = 'result/fig/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if len(faces) == 0:
    print('cannot ditected!!')

for file in glob.glob('test_data/renamed/*'):
    im = cv2.imread(file)
    print(f)
    gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray_im)
    for i, (x,y,w,h) in enumerate(faces):
        l = max([w,h])
        # 一人ずつ顔を切り抜く
        y1 = max([int(y - l*0.1), 0])
        y2 = min([int(y + l*1.1), im.shape[0]-1])
        x1 = max([int(x - l*0.1), 0])
        x2 = min([int(x + l*1.1), im.shape[1]-1])
        im_face = im[y1:y2, x1:x2]
        im_face = cv2.resize(im_face, (size,size))
        im_face = [im_face.astype('float32') / 255]
        r = model.predict(np.array(im_face))
        res = r[0]
        for i, acc in enumerate(res):
            print(idol[i+1], '=', str(acc * 100), '%')
        print('result---', idol[res.argmax()+1])
        output_path = os.path.join(output_dir,file.split('/')[-1])
        font = cv2.FONT_HERSHEY_SIMPLEX
        if res.argmax()+1 == 1:
            color=(0,0,255)
        else:
            color=(255,0,0)
        cv2.putText(im, idol[res.argmax()+1], (x1, y1), font,1, color=color)
        cv2.rectangle(im, (x1,y1), (x2,y2), color=color, thickness=3)
    cv2.imwrite(output_path, im)
    plt.imshow(im)
    plt.show()
