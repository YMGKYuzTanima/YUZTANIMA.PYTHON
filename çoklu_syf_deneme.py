# -*- coding: utf-8 -*-
"""
Created on Fri May 21 23:14:33 2021

@author: Lenovo
"""

#kütüphanelerin yüklenmesi
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.Qt import *
import  sys
import cv2
import os
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import *
import sys
from PyQt5.QtCore import *
import pyodbc
import sqlalchemy as sa
import urllib.parse
from collections import deque
from datetime import datetime
import time
import datetime
from time import gmtime ,strftime, strptime
from datetime import timedelta
from PIL import Image
import numpy as np
import pandas as pd
from threading import Thread
from collections import deque
import imutils
#veritabanı bağlantısı kurulur
CONNECTION_STRING = r'Driver={SQL Server};Server=DESKTOP-DUGGE5O\SQLEXPRESS;Database=facesql;Trusted_Connection=yes;'
engine = sa.create_engine('mssql+pyodbc:///?odbc_connect=' + urllib.parse.quote_plus(CONNECTION_STRING))
conn=engine.connect()
metadata=sa.MetaData()
personel_tab = sa.Table('personel', metadata, autoload=True, autoload_with=engine) 
db = pyodbc.connect(
    'Driver={SQL Server};'
    'Server=DESKTOP-DUGGE5O\SQLEXPRESS;'
    'Database=facesql;'
    'Trusted_Connection=true;'
)
cursor=db.cursor()

giris_ip=""
cikis_ip=""
  
##############################thread#################################################
#multi thread için CameraWidget sınıfı oluşturulur
class CameraWidget():
    """Independent camera feed
    Uses threading to grab IP camera frames in the background

    @param width - Width of the video frame
    @param height - Height of the video frame
    @param stream_link - IP/RTSP/Webcam link,,,,,,,
    @param aspect_ratio - Whether to maintain frame aspect ratio or force into fraame
    """

    def __init__(self, width, height, stream_link=0, aspect_ratio=False, parent=None, deque_size=1):
        

        # Akıştan okunan kareleri depolamak için kullanılan deque'i başlat
        self.deque = deque(maxlen=deque_size)
         
        #genişlik uzunluk ayarlanır
        self.offset = 16
        self.screen_width = width - self.offset
        self.screen_height = height - self.offset
        self.maintain_aspect_ratio = aspect_ratio

        self.camera_stream_link = stream_link

        # Kameranın geçerli/çalışıp çalışmadığını kontrol etmek için ön atamalar yapılır
        self.online = False
        self.capture = None
        

        self.load_network_stream()

        # Arka plan çerçevesi yakalanmaya başlanır
        self.get_frame_thread = Thread(target=self.get_frame, args=())
        self.get_frame_thread.daemon = True
        self.get_frame_thread.start()

        #Periyodik olarak görüntülenecek video karesi ayarlanır
      

        print('Started camera: {}'.format(self.camera_stream_link))

    def load_network_stream(self):
        """Akış bağlantısını doğrular ve geçerliyse yeni akışı açar"""

        def load_network_stream_thread():
            if self.verify_network_stream(self.camera_stream_link):#bağlanılacak kamera
                self.capture = cv2.VideoCapture(self.camera_stream_link)
                self.online = True
        self.load_stream_thread = Thread(target=load_network_stream_thread, args=())
        self.load_stream_thread.daemon = True
        self.load_stream_thread.start()


    def verify_network_stream(self, link):
        """Verilen bağlantıdan bir çerçeve alma girişimleri"""

        cap = cv2.VideoCapture(link)
        if not cap.isOpened():#kamera açık ise false dödür
            return False
        """yüz tanıma """
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        faceCascade = cv2.CascadeClassifier("Cascade/haarcascade_frontalface_default.xml")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cap.set(3, 640)
        cap.set(4, 480)

        minW = 0.1*cap.get(3)
        minH = 0.1*cap.get(4)
        path = "data/" """çekilecek görüntülerin yolu"""
        girisNameDict = dict()
        cikisNameDict = dict()
        def Giris(name):
            """kişi giriş çıkış kaydı fonksiyonu"""
 
            db = pyodbc.connect(
                'Driver={SQL Server};'
                'Server=DESKTOP-DUGGE5O\SQLEXPRESS;'
                'Database=facesql;'
                'Trusted_Connection=true;')
            cursor=db.cursor()
            z_id_cursor=cursor.execute("select ziyaretci_ID from ziyaretci where kimlik_nu=?",(str(name)))
            df_zid=pd.DataFrame(z_id_cursor)
            """Yüz tanımanın algıladığı kişinin id bilgisi çekilir"""
            z_id=df_zid.iloc[0,0][0]
            a=""
            b=""
            if str(name) not in girisNameDict.keys():
                girisNow = datetime.datetime.now()
                girisNameDict[str(name)]= girisNow.strftime('%H:%M:%S')
                b=str(girisNow.strftime('%H:%M:%S'))
                """Kişi giriş yapıyor ise giriş tarihi kaydedilir"""
        
            else:
                """Kişinin çıkış tarihi kaydedilir"""
                cikisNow= datetime.datetime.now()
                cikisNameDict[str(name)]= cikisNow.strftime('%H:%M:%S')
                b=str(cikisNow.strftime('%H:%M:%S'))
                """giriş çıkış saatleri veritabanına işlenir"""
                CONNECTION_STRING = r'Driver={SQL Server};Server=DESKTOP-DUGGE5O\SQLEXPRESS;Database=facesql;Trusted_Connection=yes;'
                engine = sa.create_engine('mssql+pyodbc:///?odbc_connect=' + urllib.parse.quote_plus(CONNECTION_STRING))
                conn=engine.connect()
                metadata=sa.MetaData()
                emp = sa.Table('ziyaretci_raporu', metadata, autoload=True, autoload_with=engine)
                query = sa.insert(emp).values(ziyaretci_ID=z_id, son_konumu=a, kamera_görüs=b ,ihlal=False)
                ResultProxy = conn.execute(query)
                
        while True:
            ret, img =cap.read()
            img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,)

            for (x, y, w, h) in faces:

                """color = img[y:y + h, x:x + w]"""
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                print(id,"ve",link)
                if (confidence < 100):
                    print(id)
                    """benzerlik oranı """
                    confidence = "  {0}%".format(round(100 - confidence))
                    
                    Giris(id)
                    """Giriş fonksiyonu tanınan yüz için çağrılır"""
                    print(girisNameDict ,"ve", cikisNameDict)
                else:
                    """kişi tanınmıyor ise"""
                    id = "bilinmiyor"
                    confidence = "  {0}%".format(round(100 - confidence))
                cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

                """img_item = "my_image.jpg"
                    cv2.imwrite(img_item,color)"""

            cv2.imshow('camera', img)
            """tanıma için kamera ekranı açılır"""
            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        return True

    def get_frame(self):
        """Çerçeveyi okur, yeniden boyutlandırır ve görüntüyü pixmap'e dönüştürür"""

        while True:
            try:
                
                if self.capture.isOpened() and self.online:
                    """Akıştan sonraki kareyi oku ve deque içine ekle"""
                    status, frame = self.capture.read()
                    print("çalıştı")
                    if status:
                        self.deque.append(frame)
                    else:
                        self.capture.release()
                        self.online = False
                else:
                    # Yeniden bağlanmayı dene
                    print('attempting to reconnect', self.camera_stream_link)
                    self.load_network_stream()
                    self.spin(2)
                self.spin(.001)
            except AttributeError:
                pass

    def spin(self, seconds):
        """Ayarlanan saniye kadar duraklat, programın durmaması için time.sleep'in yerini alır"""

        time_end = time.time() + seconds
        while time.time() < time_end:
            QApplication.processEvents()


def exit_application():
    """Program olay işleyicisinden çık"""

    sys.exit(1)
 
class Pencere_personel(QMainWindow):
    """personel penceresi"""
    def __init__(self):
        super().__init__()
        self.setUii()
        
    def setUii(self):
        """bağlanacak ekran için nesne oluşturulur"""
        mainwindow=self.AnaPencere()
        self.setStyleSheet(open("stil_dnm.qss","r").read())#stil dosyası okunur
        self.setWindowTitle("Personel Yüz Tanıma")#pencere başlığı
        self.setCentralWidget(mainwindow)#ekrana ortalanır
        #boyutlar
        self.setMaximumSize(QSize(700,700))
        self.setMinimumSize(QSize(500,500))
        
        
        #self.show()
    def AnaPencere(self):
        """arayüz için egerekli nesneler oluşturulur ve işlevsellik kazandırılır"""
        widget=QWidget()
        v_box=QVBoxLayout()
        img=QLabel()
        img.setPixmap(QPixmap("ymgk_deneme.png"))
        img.setAlignment(Qt.AlignHCenter)
        self.lbl_tc=QLabel("T.C. Kimlik No")
        
        self.tc=QLineEdit()
        
        
        kayit_btn= QPushButton("FOTOĞRAF EKLE")
        ogrenme_btn= QPushButton("İŞLEMİ TAMAMLA")
        
        self.yazi_alani=QLabel()
        
        kayit_btn.clicked.connect(self.kayit)
        ogrenme_btn.clicked.connect(self.ogrenme)
        #kamera_btn.clicked.connect(self.kamera)
        
        
        v_box.addStretch()
        v_box.addWidget(img)
        v_box.addStretch()
        v_box.addWidget(self.lbl_tc)
        v_box.addWidget(self.tc)
       
        v_box.addWidget(self.yazi_alani)
        v_box.addStretch()
        v_box.addWidget(kayit_btn)
        v_box.addWidget(ogrenme_btn)
        
        v_box.addStretch()
        v_box.addStretch()
        
        h_box=QHBoxLayout()
        h_box.addStretch()
        h_box.addLayout(v_box)
        h_box.addStretch()
        
        widget.setLayout(h_box)
        return widget
    def ogrenme(self):
            
            #if self.yazi_alani.text=="Bilgi!/n Yüz yakalama başlatılıyor.\nKameraya bakınız ve bekleyiniz ...":
                """öğrenme için oluşturulan butonda kullanılan fonksiyon tanımlanır"""
                print("ogrenme")
                path = 'data'#ögrenmede kullanılacak data
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                detector = cv2.CascadeClassifier("Cascade/haarcascade_frontalface_default.xml")

                def getImagesAndLabels(path):
                    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
                    faceSamples = []
                    ids = []
                    for imagePath in imagePaths:
                        PIL_img = Image.open(imagePath).convert('L')
                        img_numpy = np.array(PIL_img,'uint8')
                        id = int(os.path.split(imagePath)[-1].split(".")[1])
                        faces = detector.detectMultiScale(img_numpy)
                        print(imagePath)
                        for (x,y,w,h) in faces:
                            faceSamples.append(img_numpy[y:y+h,x:x+w])
                            ids.append(id)
                    return faceSamples,ids

                self.yazi_alani.setText("\n Yüzler taranıyor.Birkaç saniye sürecek lütfen bekleyiniz...")

                faces, ids = getImagesAndLabels(path)

                recognizer.train(faces,np.array(ids))

                recognizer.write('trainer/trainer.yml')
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            #else:
                #self.yazi_alani.setText("İşlemi tamamlamadan önce yüzünüzü\ntanıtmanız gerekmektedir...")

        
    def kayit(self):
        """okuma ve personel kontrolü için oluşturulan butonda kullanılan fonksiyon tanımlanır"""
        #self.hide()
        self.tc_no=self.tc.text()
        cursor.execute("select * from personel where kimlik_numarası =?",(self.tc_no))
        kontrol=cursor.fetchall()
        if len(kontrol)==0:
            self.yazi_alani.setText("Kayıtlı personel bulunamadı.")
            
        else:
            #okuma
            cam = cv2.VideoCapture(0)
            #cam.set(3, 640)
            #cam.set(4, 480)
            face_detector = cv2.CascadeClassifier("Cascade/haarcascade_frontalface_default.xml")
        
            face_id = self.tc_no
            self.yazi_alani.setText("Bilgi!/n Yüz yakalama başlatılıyor./nKameraya bakınız ve bekleyiniz ...")

            count = 0
            while True:
                ret, img = cam.read()
                img = cv2.flip(img, 1)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    count += 1
                    cv2.imwrite("data/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
                    cv2.imshow('image', img)
                k = cv2.waitKey(100) & 0xff
                if k == 27:
                     break
                elif count >= 30:
                     break

            cam.release()
            cv2.destroyAllWindows()
            
            
       

class Pencereler(QMainWindow):
    """ziyaretçi ekranı"""
    def __init__(self):
        super().__init__()
        self.setUii()
        
    def setUii(self):
        mainwindow=self.AnaPencere()
        self.setStyleSheet(open("stil_dnm.qss","r").read())#stil dosyası okunur
        self.setWindowTitle("Ziyaretçi Kontrol")#prncere başlığı
        self.setCentralWidget(mainwindow)
        #pencere boyutları
        self.setMaximumSize(QSize(700,700))
        self.setMinimumSize(QSize(500,500))
        
        
        #self.show()
    def AnaPencere(self):
        """arayüz için egerekli nesneler oluşturulur ve işlevsellik kazandırılır"""
        widget=QWidget()
        v_box=QVBoxLayout()
        img=QLabel()
        img.setPixmap(QPixmap("ymgk_deneme.png"))
        img.setAlignment(Qt.AlignHCenter)
        self.lbl_tc=QLabel("T.C. Kimlik No")
       
        self.tc=QLineEdit()
        """girilecek verinin sadece sayılardan ve max 11 karekterden oluşmasını sağlar"""
        self.tc.setInputMask("99999999999")
        
        
        kayit_btn= QPushButton("ZİYARETÇİYİ KONTROL ET")
        per_btn= QPushButton("PERSONEL İŞLEMİ")
        ogrenme_btn=QPushButton("İşlemi Tamamla")
        ogrenme_btn.clicked.connect(self.ogrenme)
        per_btn.clicked.connect(self.personel)
        
        self.yazi_alani=QLabel()
        
        kayit_btn.clicked.connect(self.kayit)
        
       
        
        
        v_box.addStretch()
        v_box.addWidget(img)
        v_box.addStretch()
        v_box.addWidget(self.lbl_tc)
        v_box.addWidget(self.tc)
       
        v_box.addWidget(self.yazi_alani)
        v_box.addStretch()
        v_box.addWidget(kayit_btn)
        v_box.addWidget(ogrenme_btn)
        v_box.addWidget(per_btn)
        v_box.addStretch()
        v_box.addStretch()
        
        h_box=QHBoxLayout()
        h_box.addStretch()
        h_box.addLayout(v_box)
        h_box.addStretch()
        
        widget.setLayout(h_box)
        return widget
    def ogrenme(self):
        #if self.yazi_alani.text=="Bilgi!/n Yüz yakalama başlatılıyor.\nKameraya bakınız ve bekleyiniz ...":
            """öğrenme için oluşturulan butonda kullanılan fonksiyon tanımlanır"""            
            print("ogrenme")
            path = 'data'
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            detector = cv2.CascadeClassifier("Cascade/haarcascade_frontalface_default.xml")

            def getImagesAndLabels(path):
                imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
                faceSamples = []
                ids = []
                for imagePath in imagePaths:
                    PIL_img = Image.open(imagePath).convert('L')
                    img_numpy = np.array(PIL_img,'uint8')
                    id = int(os.path.split(imagePath)[-1].split(".")[1])
                    faces = detector.detectMultiScale(img_numpy)
                    print(imagePath)
                    for (x,y,w,h) in faces:
                        faceSamples.append(img_numpy[y:y+h,x:x+w])
                        ids.append(id)
                return faceSamples,ids

            self.yazi_alani.setText("\n Yüzler taranıyor.Birkaç saniye sürecek lütfen bekleyiniz...")

            faces, ids = getImagesAndLabels(path)

            recognizer.train(faces,np.array(ids))

            recognizer.write('trainer/trainer.yml')
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        #else:
            #self.yazi_alani.setText("İşlemi tamamlamadan önce yüzünüzü\ntanıtmanız gerekmektedir...")
    def personel(self):
        """personel penceresine bağlanmak için gerçekleştirilen işlemler"""
        self.hide()
        self.personel_win=Pencere_personel()
        self.personel_win.show()
    def kayit(self):
        self.tc_num=self.tc.text()
        
        """okuma ve ziyaretçi kontrolü için oluşturulan butonda kullanılan fonksiyon tanımlanır"""
        cursor.execute("select ziyaret_saati from ziyaretci where kimlik_nu =? and ziyaret_durum=?",(self.tc_num,True))
        kontrol=cursor.fetchall()
        print(kontrol)
        if len(kontrol)==0:
            self.yazi_alani.setText("Randevu kaydı bulunamadı \nveya randevunuz onaylanmadı!")
        else:
            z_saat=kontrol[0][0]
            z_saati=z_saat + timedelta(minutes=5)
            end_time=z_saat - timedelta(minutes=15)
            simdi=datetime.datetime.now()
            if simdi<end_time or simdi>z_saati:
                print("onaylanmadı")
                self.yazi_alani.setText("Ziyaretçinin 15 dakika erken ve\n5 dakika geçikme hakkı bulunmaktadır\nbunun dışında giriş yapılamaz!")
            
            elif z_saati>simdi and end_time<simdi:
                print("onaylandı")
                self.yazi_alani.setText("Bilgi!/n Yüz yakalama başlatılıyor.\nKameraya bakınız ve bekleyiniz ...")
                #########################VERİ TOPLAMA###############################
                cam = cv2.VideoCapture(0)
                #cam.set(3, 640)
                #cam.set(4, 480)
                #C:/Users/Lenovo/Desktop/ymgk/Cascade
                face_detector = cv2.CascadeClassifier("Cascade/haarcascade_frontalface_default.xml")
                
                face_id = self.tc_num
                print("Bilgi...\nYüz yakalama başlatılıyor.\nKameraya bak ve bekle ...")
        
                count = 0
                while True:
                    ret, img = cam.read()
                    img = cv2.flip(img, 1)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_detector.detectMultiScale(gray, 1.3, 5)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        count += 1
                        cv2.imwrite("data/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
                        cv2.imshow('image', img)
                    k = cv2.waitKey(100) & 0xff
                    if k == 27:
                        break
                    elif count >= 30:
                        break
        
                cam.release()
                cv2.destroyAllWindows()
               
class Pencere1(QMainWindow):
    #kullanici girisi
    def __init__(self):
        super().__init__()
        self.setUii()
        
    def setUii(self):
        mainwindow=self.AnaPencere()
        self.setStyleSheet(open("stil_dnm.qss","r").read())#stil dosyası okunur
        self.setWindowTitle("Kullanıcı Giriş")#pencere başlığı
        self.setCentralWidget(mainwindow)
        self.setMaximumSize(QSize(700,700))
        self.setMinimumSize(QSize(500,500))
        
        self.kamera_win=Pencereler()
        self.show()
    def AnaPencere(self):
        """arayüz için egerekli nesneler oluşturulur ve işlevsellik kazandırılır"""
        widget=QWidget()
        v_box=QVBoxLayout()
        img=QLabel()
        img.setPixmap(QPixmap("grs.png"))
        img.setAlignment(Qt.AlignHCenter)
        self.lbl_name=QLabel("Kullanıcı Adınız")
        self.lbl_sfr=QLabel("Şifreniz")
        
        self.name=QLineEdit()
        self.sifre=QLineEdit()
        
        kayit_btn= QPushButton("Giriş Yap")
        #kamera_btn=QPushButton("Kameralar için tıklayın")
        self.yazi_alani=QLabel()
        
        kayit_btn.clicked.connect(self.kayit)
        #kamera_btn.clicked.connect(self.kamera)
        self.sifre.setEchoMode(QLineEdit.Password)
        
        v_box.addStretch()
        v_box.addWidget(img)
        v_box.addStretch()
        v_box.addWidget(self.lbl_name)
        v_box.addWidget(self.name)
        v_box.addWidget(self.lbl_sfr)
        v_box.addWidget(self.sifre)
        v_box.addWidget(self.yazi_alani)
        v_box.addStretch()
        v_box.addWidget(kayit_btn)
        #v_box.addWidget(kamera_btn)
        v_box.addStretch()
        v_box.addStretch()
        
        h_box=QHBoxLayout()
        h_box.addStretch()
        h_box.addLayout(v_box)
        h_box.addStretch()
        
        widget.setLayout(h_box)
        return widget
    def kayit(self):
        self.adi=self.name.text()
        self.sifree=self.sifre.text()
        
        
        cursor.execute("select * from personel where kullanıcı_ADI=? and kullanıcı_sifre=? and birim_ID=?",(self.adi,self.sifree,8))
        kontrol=cursor.fetchall()
        
        if len(kontrol)==0:
            """kullanıcı bulunamaz ise"""
            self.yazi_alani.setText("Kullanıcı adı veya şifre\nhatalı lütfen tekrar deneyin.")
            
        else:
            """kullanıcı bulunur ise yapılan veritabanı işlemleri ve diğer işlemler"""
            self.hide()
            self.kamera_win.show()
            dnm=cursor.execute("select firma_ID from personel where kullanıcı_ADI=? and kullanıcı_sifre=? and birim_ID=?",(self.adi,self.sifree,8))

            df=pd.DataFrame(dnm)
            print(df)
            f_id=int(df.iloc[0,0][0])
            #çıkış kamerasının ip si
            dnm_cks=cursor.execute("select kamera_IP from kamera where kamera_konum_isimlendirme='çıkış' and firma_ID =?",f_id)
            df_cks=pd.DataFrame(dnm_cks)
            cikis_ip=df_cks.iloc[0,0][0]
            print(cikis_ip)
            #giriş kamerasının ipsi
            dnm_grs=cursor.execute("select kamera_IP from kamera where kamera_konum_isimlendirme='giriş' and firma_ID =?",f_id)
            df_grs=pd.DataFrame(dnm_grs)
            giris_ip=df_grs.iloc[0,0][0]
            print(giris_ip)
            #kameraların döngüyle multitheared e atılması
            dnm1= cursor.execute("select kamera_IP from kamera where firma_ID =?",f_id)
            df1=pd.DataFrame(dnm1)
            #CameraWidget(80, 80,0)
            
            for i in range(len(df1)):
                """veritabanından giriş yapan güvenlik birimine ait personelin kayıtlı olduğu firmanın kameralarında 
                giriş yapılınca otomatk olarak tanıma işlemi başlar"""
                CameraWidget(150,150,str(df1.iloc[i,0][0]))
              
                print(df1.iloc[i,0][0])
                
        
if __name__=="__main__":
    
    app=QApplication(sys.argv)
    pencere=Pencere1()
    sys.exit(app.exec_())
