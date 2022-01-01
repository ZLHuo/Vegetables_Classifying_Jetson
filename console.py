import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import subprocess
from jetcam.csi_camera import CSICamera
import cv2
import numpy as np
import time
import threading
import pymysql
import tensorflow as tf

def update_image(change):
    global img
    img = change['new']

camera = CSICamera(capture_device=0, width=640, height=360)
with open('resources/list.txt', encoding='utf-8') as f:
    class_names = f.read().splitlines()
fruit=[]
for i in class_names:
    fruit.append(i.split(','))
    
camera.running = True
camera.observe(update_image, names='value')
model = tf.keras.models.load_model("resources/mobilenet_fv.h5")

process = subprocess.Popen(["python", "/home/jetson/main/resources/example.py"],
                            shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

conn = pymysql.connect(
    host="localhost",
    user="root",
    password="jetson",
    database="fruit"
)
cursor = conn.cursor()


weight = 0.0
prevWeight = 0.0
lock = False

def check():
    global lock,img
    lock=True
    print("Starting")

    time.sleep(1)

    a = time.time()
    img2 = img[0:360,140:500]
    cv2.imwrite("resources/a.png", img2)
    img2 = cv2.resize(img2, (224, 224))
    img2 = np.asarray(img2)  # 将图片转化为numpy的数组
    outputs = model.predict(img2.reshape(1, 224, 224, 3))  # 将图片输入模型得到结果
    result_index = int(np.argmax(outputs))
    result = fruit[result_index][0]  # 获得对应的水果名称

    sql = "SELECT * FROM tradeDetail"
    id = cursor.execute(sql) + 1 
    sql = ('INSERT INTO tradeDetail (id,name,weight,unitPrice,price,time) value (' + str(id)+',"' +
           fruit[result_index][1]+'",'+str(weight)+','+fruit[result_index][2] + ','+str(0.001*weight*float(fruit[result_index][2]))+','+str(round(time.time()))+');')
    cursor.execute(sql)
    conn.commit()
    
    print(result)
    print(weight)
    print("Time Cost:", time.time()-a)
    lock = False

print("Ready")

while True:
    output = process.stdout.readline()
    if process.poll() is not None:
        break
    if output:
        prevWeight = weight
        weight = float(output.decode())
        #print(weight)
        if weight-prevWeight > 50:
            print(lock)
            print("Weight:",weight)
            if lock == False:
                t1=threading.Thread(target=check)
                t1.start()
            print("Detected.")