import argparse

from real_time_process import UdpListener, DataProcessor
from radar_config import SerialConfig
from radar_config import DCA1000Config
from queue import Queue
import pyqtgraph as pg
from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import time
import torch
import sys
import numpy as np
from serial.tools import list_ports
import iwr6843_tlv.detected_points as readpoint
import globalvar as gl
# import models.predict as predict
# from models.model import CNet, FeatureFusionNet
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
from colortrans import pg_get_cmap

# -----------------------------------------------
from UI_interface import Ui_MainWindow, Qt_pet
# -----------------------------------------------


datasetfile = 'dataset'
datasetsencefile = ' '
gesturedict = {
                '0':'backward',
                '1':'dbclick',
                '2':'down',
                '3':'front',
                '4':'Left',
                '5':'Right',
                '6':'up',
                '7':'NO'
               }

cnt = 0

_flagdisplay = False

def loadmodel():
    global model
    if (modelfile.currentText()!='--select--'and modelfile.currentText()!=''):
        model_info = torch.load(modelfile.currentText(),map_location='cpu')
        # TODO: 
        model = []
        model.load_state_dict(model_info['state_dict'])
        printlog('加载'+modelfile.currentText()+'模型成功!',fontcolor='blue')
    else:
        printlog("请加载模型!",fontcolor='red')

def cleartjpg():
    view_gesture.setPixmap(QtGui.QPixmap("gesture_icons/"+str(7)+".jpg"))
    subWin.img_update("gesture_icons/"+str(7)+".jpg")   

def Judge_gesture(a,b,c,d,e):
    global _flagdisplay
    if model:
        # TODO:
        fanhui = [] #predict.predictGesture(model,d,b,e,c,a)
        view_gesture.setPixmap(QtGui.QPixmap("gesture_icons/"+str(fanhui)+".jpg"))
        subWin.img_update("gesture_icons/"+str(fanhui)+".jpg")
        QtCore.QTimer.singleShot(2000, cleartjpg)
        _flagdisplay = True
        printlog("输出:" + gesturedict[str(fanhui)],fontcolor='blue')
        return gesturedict[str(fanhui)]

def update_figure():
    global img_rdi, img_rai, img_rti, img_rei, img_dti
    global idx,cnt

    img_rti.setImage(RTIData.get().sum(2)[0:1024:16,:], levels=[0, 1e4])
    # img_rdi.setImage(RDIData.get()[:, :, 0].T, levels=[30, 50])
    img_rdi.setImage(RDIData.get().sum(0)[:, :, 0].T,levels=[2e4, 4e5])
    # img_rei.setImage(REIData.get().T,levels=[0, 3])
    img_rei.setImage(REIData.get()[4:12,:,:].sum(0).T,levels=[0, 8])
    img_dti.setImage(DTIData.get(),levels=[0, 1000])
    # img_rai.setImage(RAIData.get().sum(0).T, levels=[1.2e3, 4e6])
    # img_rai.setImage(RAIData.get()[0,:,:].T, levels=[8e3, 2e4])
    # img_rai.setImage(RAIData.get(),levels=[0, 3])
    img_rai.setImage(RAIData.get()[4:12,:,:].sum(0),levels=[0, 8])


    if gl.get_value('usr_gesture'):
        RT_feature = RTIData.get().sum(2)[0:1024:16,:]
        DT_feature = DTIData.get()
        RDT_feature = RDIData.get()[:, :, :, 0]
        ART_feature = RAIData.get()
        ERT_feature = REIData.get()

        # if Recognizebtn.isChecked():
        if Recognizebtn.isChecked():
            # 识别
            
            time_start = time.time()  # 记录开始时间
            result = Judge_gesture(RT_feature,DT_feature,RDT_feature,
                                            ART_feature,ERT_feature)
            time_end = time.time()  # 记录结束时间
            time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
            printlog('识别时间:'+str(time_sum)+'s, '+'识别结果:'+result,fontcolor='blue')


        elif CaptureDatabtn.isChecked() and datasetsencefile != '':
            idx=idx+1
            # 收集
            np.save(datasetsencefile+'/RT_feature_'+str(idx).zfill(5)+'.npy',RT_feature)
            np.save(datasetsencefile+'/DT_feature_'+str(idx).zfill(5)+'.npy',DT_feature)
            np.save(datasetsencefile+'/RDT_feature_'+str(idx).zfill(5)+'.npy',RDT_feature)
            np.save(datasetsencefile+'/ART_feature_'+str(idx).zfill(5)+'.npy',ART_feature)
            np.save(datasetsencefile+'/ERT_feature_'+str(idx).zfill(5)+'.npy',ERT_feature)
            printlog('采集到特征:'+datasetfilebox.currentText()+'-'+str(idx).zfill(5),fontcolor='blue')
        
        gl.set_value('usr_gesture', False)

    QtCore.QTimer.singleShot(1, update_figure)


def printlog(string,fontcolor):
    logtxt.moveCursor(QtGui.QTextCursor.End)
    gettime = time.strftime("%H:%M:%S", time.localtime())
    logtxt.append("<font color="+fontcolor+">"+str(gettime)+"-->"+string+"</font>")

def getradarparameters():
    if radarparameters.currentIndex() > -1 and radarparameters.currentText() != '--select--':
        radarparameters.setToolTip(radarparameters.currentText())
        configParameters = readpoint.IWR6843AOP_TLV()._initialize(config_file = radarparameters.currentText())
        rangeResolutionlabel.setText(str(configParameters["rangeResolutionMeters"])+'cm')
        dopplerResolutionlabel.setText(str(configParameters["dopplerResolutionMps"])+'m/s')
        maxRangelabel.setText(str(configParameters["maxRange"])+'m')
        maxVelocitylabel.setText(str(configParameters["maxVelocity"])+'m/s')

def openradar(config,com):
    global radar_ctrl
    radar_ctrl = SerialConfig(name='ConnectRadar', CLIPort=com, BaudRate=115200)
    radar_ctrl.StopRadar()
    radar_ctrl.SendConfig(config)
    processor.start()
    processor.join(timeout=1)
    update_figure()

def updatacomstatus(cbox):
    port_list = list(list_ports.comports())
    cbox.clear()
    for i in range(len(port_list)):
        cbox.addItem(str(port_list[i][0]))

def setserialport(cbox, com):
    global CLIport_name
    global Dataport_name
    if cbox.currentIndex() > -1:
        port = cbox.currentText()
        if com == "CLI":
            CLIport_name = port
   
        else:
            Dataport_name = port
    
def sendconfigfunc():
    global CLIport_name
    global Dataport_name
    if len(CLIport_name) != 0  and radarparameters.currentText() != '--select--':
        openradar(radarparameters.currentText(),CLIport_name)
        printlog(string = '发送成功', fontcolor='green')
    else:
        printlog(string = '发送失败', fontcolor='red')


def setintervaltime():
    gl.set_value('timer_2s', True)
    QtCore.QTimer.singleShot(2000, setintervaltime)

# cnt 用来计数 200ms*cnt，代表显示多长时间
cnt = 0
def setdisplaygestureicontime():
    global _flagdisplay, cnt
    if _flagdisplay==True:
        cnt = cnt + 1
        if cnt>4:
            cnt = 0
            view_gesture.setPixmap(QtGui.QPixmap("gesture_icons/"+str(7)+".jpg"))
            subWin.img_update("gesture_icons/"+str(7)+".jpg")
            _flagdisplay=False
    QtCore.QTimer.singleShot(200, setdisplaygestureicontime)

def setcolor():
    if(color_.currentText()!='--select--' and color_.currentText()!=''):
        if color_.currentText() == 'customize':
            pgColormap = pg_get_cmap(color_.currentText())
        else:
            cmap=plt.cm.get_cmap(color_.currentText())
            pgColormap = pg_get_cmap(cmap)
        lookup_table = pgColormap.getLookupTable(0.0, 1.0, 256)
        img_rdi.setLookupTable(lookup_table)
        img_rai.setLookupTable(lookup_table)
        img_rti.setLookupTable(lookup_table)
        img_dti.setLookupTable(lookup_table)
        img_rei.setLookupTable(lookup_table)

def get_filelist(dir,Filelist):
    newDir=dir
    #注意看dir是文件名还是路径＋文件名！！！！！！！！！！！！！！
    if os.path.isfile(dir):
        dir_ = os.path.basename(dir)  
        if (dir_[:2] == 'DT') and (dir_[-4:] == '.npy'):
            Filelist[0].append(dir)
        elif (dir_[:2] == 'RT') and (dir_[-4:] == '.npy'):
            Filelist[1].append(dir)
        elif (dir_[:3] == 'RDT') and (dir_[-4:] == '.npy'):
            Filelist[2].append(dir)
        elif (dir_[:3] == 'ART') and (dir_[-4:] == '.npy'):
            Filelist[3].append(dir)    
        elif (dir_[:3] == 'ERT') and (dir_[-4:] == '.npy'):
            Filelist[4].append(dir)  
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir=os.path.join(dir,s)
            get_filelist(newDir,Filelist)
    return Filelist

def savedatasetsencefile():
    global datasetsencefile,start_captureidx,idx
    datasetsencefile = datasetfile+'/'+ whodatafile.text()+'/'+datasetfilebox.currentText()
    if not os.path.exists(datasetsencefile):  #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(datasetsencefile)

    featurelist = get_filelist(datasetsencefile, [[] for i in range(5)])
    start_captureidx = len(featurelist[0])
    idx = start_captureidx


def show_sub():
    subWin.show()
    MainWindow.hide()



def application():
    global color_,radarparameters,maxVelocitylabel,maxRangelabel,dopplerResolutionlabel,rangeResolutionlabel,logtxt
    global Recognizebtn,CaptureDatabtn,view_gesture,modelfile,datasetfilebox,whodatafile
    global img_rdi, img_rai, img_rti, img_rei, img_dti,ui
    global subWin,MainWindow
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    MainWindow.show()
    ui = Ui_MainWindow()

    ui.setupUi(MainWindow)
    subWin = Qt_pet(MainWindow)
    
    # 改了D:\Applications\anaconda3\Lib\site-packages\pyqtgraph\graphicsItems\ViewBox
    # 里的ViewBox.py第919行padding = self.suggestPadding(ax)改成padding = 0
    view_rdi = ui.graphicsView_6.addViewBox()
    ui.graphicsView_6.setCentralWidget(view_rdi)#去边界
    view_rai = ui.graphicsView_4.addViewBox()
    ui.graphicsView_4.setCentralWidget(view_rai)#去边界
    view_rti = ui.graphicsView.addViewBox()
    ui.graphicsView.setCentralWidget(view_rti)#去边界
    view_dti = ui.graphicsView_2.addViewBox()
    ui.graphicsView_2.setCentralWidget(view_dti)#去边界
    view_rei = ui.graphicsView_3.addViewBox()
    ui.graphicsView_3.setCentralWidget(view_rei)#去边界

    view_gesture = ui.graphicsView_5
    view_gesture.setPixmap(QtGui.QPixmap("gesture_icons/7.jpg"))

    sendcfgbtn = ui.pushButton_11
    exitbtn = ui.pushButton_12
    Recognizebtn = ui.pushButton_15
    CaptureDatabtn = ui.pushButton

    color_ = ui.comboBox
    modelfile = ui.comboBox_2
    datasetfilebox = ui.comboBox_3
    radarparameters = ui.comboBox_7
    Cliportbox = ui.comboBox_8

    logtxt = ui.textEdit
    whodatafile = ui.lineEdit_6
    changepage = ui.actionload
    

    rangeResolutionlabel = ui.label_14
    dopplerResolutionlabel = ui.label_35
    maxRangelabel = ui.label_16
    maxVelocitylabel = ui.label_37

    # ---------------------------------------------------
    # lock the aspect ratio so pixels are always square
    # view_rai.setAspectLocked(True)
    # view_rti.setAspectLocked(True)
    img_rdi = pg.ImageItem(border=None)
    img_rai = pg.ImageItem(border=None)
    img_rti = pg.ImageItem(border=None)
    img_dti = pg.ImageItem(border=None)
    img_rei = pg.ImageItem(border=None)

    # Colormap
    pgColormap = pg_get_cmap('customize')
    lookup_table = pgColormap.getLookupTable(0.0, 1.0, 256)
    img_rdi.setLookupTable(lookup_table)
    img_rai.setLookupTable(lookup_table)
    img_rti.setLookupTable(lookup_table)
    img_dti.setLookupTable(lookup_table)
    img_rei.setLookupTable(lookup_table)

    view_rdi.addItem(img_rdi)
    view_rai.addItem(img_rai)
    view_rti.addItem(img_rti)
    view_dti.addItem(img_dti)
    view_rei.addItem(img_rei)
    

    Cliportbox.arrowClicked.connect(lambda:updatacomstatus(Cliportbox)) 
    Cliportbox.currentIndexChanged.connect(lambda:setserialport(Cliportbox, com = 'CLI'))
    color_.currentIndexChanged.connect(setcolor)
    modelfile.currentIndexChanged.connect(loadmodel)
    radarparameters.currentIndexChanged.connect(getradarparameters)
    datasetfilebox.currentIndexChanged.connect(savedatasetsencefile)
    whodatafile.editingFinished.connect(savedatasetsencefile)
    sendcfgbtn.clicked.connect(sendconfigfunc)
    Recognizebtn.clicked.connect(setintervaltime)
    # Recognizebtn.clicked.connect(setdisplaygestureicontime)
    CaptureDatabtn.clicked.connect(setintervaltime)
    changepage.triggered.connect(show_sub)
    # 2022/2/24 添加小型化控件 不能正常退出了
    exitbtn.clicked.connect(app.instance().exit)
    
    app.instance().exec_()

    try:
        if radar_ctrl.CLIPort:
            if radar_ctrl.CLIPort.isOpen():
                radar_ctrl.StopRadar()
    except:
        pass


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(prog='RadarStream')
    parser.add_argument('--dll-path', default='libs/UDPCAPTUREADCRAWDATA.dll', help='path to the dll file')
    args = parser.parse_args()

    # Queue for access data
    BinData = Queue() # 原始数据队列

    # 时间信息
    RTIData = Queue() # 距离时间队列
    DTIData = Queue() # 多普勒时间队列

    # 连续过程信息
    RDIData = Queue() # 距离多普勒队列
    RAIData = Queue() # 距离方位角队列
    REIData = Queue() # 方位角俯仰角队列

    # Radar config parameters
    NUM_TX = 3
    NUM_RX = 4
    NUM_CHIRPS = 64
    NUM_ADC_SAMPLES = 64

    radar_config = [NUM_ADC_SAMPLES, NUM_CHIRPS, NUM_TX, NUM_RX]
    frame_length = NUM_ADC_SAMPLES * NUM_CHIRPS * NUM_TX * NUM_RX * 2

    # config DCA1000 to receive bin data
    dca1000_cfg = DCA1000Config('DCA1000Config',config_address = ('192.168.33.30', 4096),
                                                FPGA_address_cfg=('192.168.33.180', 4096))
    collector = UdpListener('Listener', BinData, frame_length, args.dll_path)
    processor = DataProcessor('Processor', radar_config, BinData, RTIData, DTIData, 
                                             RDIData, RAIData, REIData)
    collector.start()

    application()

    dca1000_cfg.DCA1000_close()

    collector.join(timeout=1)
    
    print("Program close")
    sys.exit()
