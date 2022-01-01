import threading
import subprocess
from jetcam.csi_camera import CSICamera
import time
import datetime
import pymysql
import numpy as np
import cv2
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.chdir('/home/jetson/main')
import tensorflow as tf

price=0
class MainWindow(QTabWidget):

    def update_image(self, change):
        self.img = change['new']

    # 初始化
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('resources/images/logo.png'))
        self.setWindowTitle('果蔬识别系统')  # todo 修改系统名称
    #    self.resize(1000, 500)
        # 模型初始化
        self.model = tf.keras.models.load_model("resources/mobilenet_fv.h5")  # todo 修改模型名称
        self.to_predict_name = "resources/images/show.png"  # todo 修改初始图片，这个图片要放在images目录下
        with open('resources/list.txt', encoding='utf-8') as f:
            self.class_names = f.read().splitlines()
        self.fruit = []
        for i in self.class_names:
            self.fruit.append(i.split(','))
        self.list=[0]*len(self.fruit)
        
        self.camera = CSICamera(capture_device=0, width=640, height=360)
        self.camera.running = True
        self.camera.observe(self.update_image, names='value')

        self.process = subprocess.Popen(["python", "/home/jetson/main/resources/example.py"],
                                   shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        self.conn = pymysql.connect(
            host="localhost",
            user="root",
            password="jetson",
            database="fruit"
        )
        self.cursor = self.conn.cursor()
        self.weight = 0.0
        self.prevWeight = 0.0
        self.exitFlag=0
        self.lock = False
        self.initUI()

    # 界面初始化，设置界面布局
    def initUI(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        font = QFont('AR PL UKai CN', 20)

        # 主页面，设置组件并在组件放在布局上
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        img_title = QLabel("购物车")
        img_title.setFont(font)
        img_title.setAlignment(Qt.AlignCenter)
        self.items = QTableWidget(0, 5)
        self.items.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.items.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.items.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.items.setAlternatingRowColors(True)
        self.items.setHorizontalHeaderLabels(
            ['名称', '重量(g)', '单价(元/kg)', '价格(元)', ' '])
        self.priceLable = QLabel("总计金额：0.0元")
        self.priceLable.setFont(font)
        self.priceLable.setAlignment(Qt.AlignRight)

        self.button=[]
        for i in range(1000):
            self.button.append(QPushButton('删除'))
            self.button[i].released.connect(self.deleteRow)
        
        left_layout.addWidget(img_title)
        left_layout.addWidget(self.items)
        left_layout.addWidget(self.priceLable)
        left_widget.setLayout(left_layout)
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        self.btn_predict = QPushButton(" 点击结算 ")
        self.btn_predict.setFont(font)
        btn_exit = QPushButton(" 退出系统 ")
        btn_exit.setFont(font)
        btn_exit.released.connect(self.exit)
        btn_exit.released.connect(self.close)

        label_result = QLabel(' 果蔬名称 ')
        self.result = QLabel("等待识别")
        label_result.setFont(QFont('AR PL UKai CN', 16))
        self.result.setFont(QFont('AR PL UKai CN', 24))
        self.img_label = QLabel()
        img_init = cv2.imread(self.to_predict_name)
        h, w, c = img_init.shape
        scale = 400 / h
        img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
        cv2.imwrite("resources/images/show.png", img_show)
        self.img_label.setPixmap(QPixmap("resources/images/show.png"))
        right_layout.addStretch()
        right_layout.addWidget(label_result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.img_label, 1, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.btn_predict)
        right_layout.addStretch()
        right_layout.addWidget(btn_exit)
        right_layout.addStretch()
        right_widget.setLayout(right_layout)
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        main_widget.setLayout(main_layout)

        # 关于页面，设置组件并把组件放在布局上
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel('第11组-智能果蔬识别系统')  # todo 修改欢迎词语
        about_title.setFont(QFont('楷体', 20))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('resources/images/bj.jpg'))
        about_img.setAlignment(Qt.AlignCenter)
        label_super = QLabel("ver:1.0")  # todo 更换作者信息
        label_super.setFont(QFont('楷体', 15))
        # label_super.setOpenExternalLinks(True)
        label_super.setAlignment(Qt.AlignRight)
        author = QLabel('小组成员：翟厚裕 霍子龙 王兴睿')
        author.setFont(QFont('楷体', 15))
        author.setAlignment(Qt.AlignRight)
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()
        about_layout.addWidget(label_super)
        about_layout.addWidget(author)
        about_widget.setLayout(about_layout)

        # 添加注释
        self.addTab(main_widget, '主页')
        self.addTab(about_widget, '关于')
        self.setTabIcon(0, QIcon('resources/images/主页面.png'))
        self.setTabIcon(1, QIcon('resources/images/关于.png'))

    # 界面关闭事件，询问用户是否关闭
    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     '退出',
                                     "是否要退出程序？",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()

    def deleteRow(self):
        button = self.sender()
        if button:
            row = self.items.indexAt(button.pos()).row()
            print(self.items.item(row,3).text())
            global price
            price = price-float(self.items.item(row, 3).text())
            x.priceLable.setText("总计金额："+str(abs(round(price, 2)))+"元")
            self.items.removeRow(row)

    def check(self):
        self.lock = True
        print("Starting")

        time.sleep(1)

        a = time.time()
        img2 = self.img[0:360, 140:500]
    #    cv2.imwrite("resources/a.png", img2)

        h, w, c = img2.shape
        scale = 400 / h
        img_show = cv2.resize(img2, (0, 0), fx=scale, fy=scale)
        cv2.imwrite("resources/images/show.png", img_show)
        self.img_label.setPixmap(QPixmap("resources/images/show.png"))

        img2 = cv2.resize(img2, (224, 224))
        img2 = np.asarray(img2)  # 将图片转化为numpy的数组
        outputs = self.model.predict(img2.reshape(1, 224, 224, 3))  # 将图片输入模型得到结果
        result_index = int(np.argmax(outputs))
        result = self.fruit[result_index][0]  # 获得对应的水果名称
        self.result.setText(result)

        sql = "SELECT * FROM tradeDetail"
        id = self.cursor.execute(sql) + 1
        sql = ('INSERT INTO tradeDetail (id,name,weight,unitPrice,price) value (' + str(id)+',"' +
            self.fruit[result_index][1]+'",'+str(self.weight)+','+self.fruit[result_index][2] + ','+str(0.001*self.weight*float(self.fruit[result_index][2]))+');')
        self.cursor.execute(sql)
        self.conn.commit()
        self.list[result_index]=1

        item=[result,round(self.weight,2),self.fruit[result_index][2],0.001*self.weight*float(self.fruit[result_index][2])]
        row = self.items.rowCount()
        self.items.insertRow(row)
        for i in range(3):
            newItem = QTableWidgetItem(str(item[i]))
            newItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            self.items.setItem(row, i, newItem)
        global price
        price = price+round(item[3],2)
        newItem = QTableWidgetItem(str(round(item[3],2)))
        self.priceLable.setText("总计金额："+str(round(price,2))+"元")
        newItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.items.setItem(row, 3, newItem)
        self.items.setCellWidget(row, 4, self.button[0])
        self.button.pop(0)

        print(result)
        print(self.weight)
        print("Time Cost:", time.time()-a)
        self.lock = False

    def backend(self):
        while self.exitFlag==0:
            output = self.process.stdout.readline()
            if self.process.poll() is not None:
                break
            if output:
                self.prevWeight = self.weight
                self.weight = float(output.decode())
                #print(weight)
                if self.weight-self.prevWeight > 50:
                    print(self.lock)
                    print("Weight:", self.weight)
                    if self.lock == False:
                        t1 = threading.Thread(target=self.check)
                        t1.start()
                    print("Detected.") 

    def exit(self):
        self.exitFlag=1
        self.process.terminate()
        self.camera.running=False

class newWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('支付系统')  # todo 修改系统名称
        desktop = QApplication.desktop()
        width = desktop.width()
        height = desktop.height()
        self.resize(400, desktop.height())
        self.move((width - self.width())/2, 0)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.initUI()

    def initUI(self):
        font = QFont('AR PL UKai CN', 20)
        layout = QVBoxLayout()
        global price
        self.about_title = QLabel('应付金额：'+str(round(price,2))+'元')
        self.about_title.setFont(QFont('AR PL UKai CN', 18))
        self.about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('resources/images/pay.png'))
        about_img.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.about_title)
        layout.addStretch()
        layout.addWidget(about_img)
        layout.addStretch()

        left_widget = QWidget()
        left_layout = QVBoxLayout()
        btn_back = QPushButton("返回修改商品")
        btn_back.setFont(font)
        left_layout.addWidget(btn_back)
        btn_pay = QPushButton("完成付款")
        btn_pay.setFont(font)
        left_layout.addWidget(btn_pay)
        btn_pay.clicked.connect(self.clear)
        btn_pay.released.connect(self.close)
        btn_back.clicked.connect(self.close)
        layout.addWidget(left_widget)
        left_widget.setLayout(left_layout)

        self.setLayout(layout)

    def clear(self):
        x.items.setRowCount(0)
        x.items.clearContents()
        global price
        price=0
        x.priceLable.setText("总计金额："+str(round(price, 2))+"元")

        sql = "SELECT * FROM orderList"
        id = x.cursor.execute(sql) + 1

        sql = 'INSERT INTO orderList (id'
        for i in x.fruit:
            sql=sql+','+i[1]
        sql=sql+') value ('+str(id)
        for i in x.list:
            sql=sql+','+str(i)
        sql=sql+');'
        print(sql)
        x.cursor.execute(sql)
        x.conn.commit()

        x.list=[0]*len(x.fruit)


    def update(self):
        global price
        self.about_title.setText("总计金额："+str(round(price,2))+"元")



if __name__ == "__main__":

    app = QApplication(sys.argv)
    x = MainWindow()
    y = newWindow()
    x.showFullScreen()
    x.btn_predict.clicked.connect(y.update)
    x.btn_predict.released.connect(y.show)
    thread = threading.Thread(target=x.backend)
    thread.start()

    sys.exit(app.exec_())
