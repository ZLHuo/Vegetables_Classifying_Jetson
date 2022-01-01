import cv2
import numpy as np
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf

model = tf.keras.models.load_model("resources/mobilenet_fv.h5")
with open('resources/list.txt',encoding='utf-8') as f:
    class_names = f.readlines()
a=time.time()
image_name = "resources/a.jpg"
img = cv2.imread(image_name)
img = cv2.resize(img, (224, 224))
img = np.asarray(img)  # 将图片转化为numpy的数组
outputs = model.predict(img.reshape(1, 224, 224, 3))  # 将图片输入模型得到结果
result_index = int(np.argmax(outputs))
result = class_names[result_index]  # 获得对应的水果名称
print(result)
print("Time Cost:",time.time()-a)