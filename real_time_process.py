import threading as th
import numpy as np
import DSP
from dsp.utils import Window
from ctypes import *

a = np.zeros(1).astype(np.int)
# 内存大小至少是frame_length的两倍 ，双缓冲区
# 98304 的计算方法是例如你的配置如下：

# adc_sample = 64
# chirp = 64
# tx_num = 3
# rx_num = 4
# frame_length = adc_sample * chirp * tx_num * rx_num * 2 = 98304
b = np.zeros(98304*2).astype(c_short)

# 转换为ctypes，这里转换后的可以直接利用ctypes转换为c语言中的int*，然后在c中使用
a_ctypes_ptr = cast(a.ctypes.data, POINTER(c_int))
# 转换为ctypes，这里转换后的可以直接利用ctypes转换为c语言中的int*，然后在c中使用
b_ctypes_ptr = cast(b.ctypes.data, POINTER(c_short))


class UdpListener(th.Thread):
    def __init__(self, name, bin_data, data_frame_length, dll_file_path):
        th.Thread.__init__(self, name=name)
        self.bin_data = bin_data
        self.frame_length = data_frame_length
        self.dll = cdll.LoadLibrary(dll_file_path)

    def run(self):
        global a_ctypes_ptr, b_ctypes_ptr
        self.dll.captureudp(a_ctypes_ptr, b_ctypes_ptr, self.frame_length)


class DataProcessor(th.Thread):
    def __init__(self, name, config, bin_queue, rti_queue, dti_queue,  rdi_queue, rai_queue, rei_queue):

        th.Thread.__init__(self, name=name)
        self.adc_sample = config[0]
        self.chirp_num = config[1]
        self.tx_num = config[2]
        self.rx_num = config[3]
        self.bin_queue = bin_queue
        self.rti_queue = rti_queue
        self.dti_queue = dti_queue
        self.rdi_queue = rdi_queue
        self.rai_queue = rai_queue
        self.rei_queue = rei_queue

    def run(self):
        global frame_count
        frame_count = 0
        lastflar = 0
        while True:
            # 对应dll中的双缓冲区，0区和1区
            if(lastflar != a_ctypes_ptr[0]):
                # print(a_ctypes_ptr[0])
                lastflar = a_ctypes_ptr[0]
                # self.bin_data.put(np.array(b_ctypes_ptr[98304*(1-a_ctypes_ptr[0]):98304*(2-a_ctypes_ptr[0])]))
                # data = self.bin_queue.get()
                data = np.array(
                    b_ctypes_ptr[98304*(1-a_ctypes_ptr[0]):98304*(2-a_ctypes_ptr[0])])
                data = np.reshape(data, [-1, 4])
                data = data[:, 0:2:] + 1j * data[:, 2::]
                # [num_chirps*tx_num, wuli_antennas, num_samples]
                data = np.reshape(
                    data, [self.chirp_num * self.tx_num, -1, self.adc_sample])
                # [num_chirps*tx_num, num_samples, wuli_antennas]
                data = data.transpose([0, 2, 1])
                # 192 = 64*3 记得改
                # TX1的：[num_chirps, num_samples, wuli_antennas]
                ch1_data = data[0: self.adc_sample*3: 3, :, :]
                # TX2的：[num_chirps, num_samples, wuli_antennas]
                ch2_data = data[1: self.adc_sample*3: 3, :, :]
                # TX3的：[num_chirps, num_samples, wuli_antennas]
                ch3_data = data[2: self.adc_sample*3: 3, :, :]
                # channel的排序方式：(0:TX1-RX1,1:TX1-RX2,2:TX1-RX3,3:TX1-RX4,| 4:TX2-RX1,5:TX2-RX2,6:TX2-RX3,7:TX2-RX4,| 8:TX3-RX1,9:TX3-RX2,10:TX3-RX3,11:TX3-RX4)
                data = np.concatenate([ch1_data, ch2_data, ch3_data], axis=2)

                frame_count += 1

                rti, rdi, dti = DSP.RDA_Time(
                    data, window_type_1d=Window.HANNING, axis=1)

                # _, rdi = DSP.Range_Doppler(data, mode=2, padding_size=[128, 64])
                rai, rei = DSP.Range_Angle(
                    data, padding_size=[128, 64, 64])
                self.rti_queue.put(rti)
                self.dti_queue.put(dti)
                self.rdi_queue.put(rdi)
                self.rai_queue.put(rai)
                self.rei_queue.put(rei)
