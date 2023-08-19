import logging

import serial

from lzn_modules.utils import TimeMaster


class SerialPort:
    def __init__(self, port, baudrate, timeout):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.status = 'fulfilled'
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        if self.ser.isOpen():
            print("[sys] Serial port is open")

    def isOpen(self):
        return self.ser.isOpen()

    def close(self):
        self.ser.close()
        if self.ser.isOpen():
            print("[sys] Serial port close failed")
        else:
            print("[sys] Serial port close successfully")

    def send_data_async(self, data):
        myTimer = TimeMaster()
        if self.status == 'fulfilled':
            self.status = 'pending'
            # print('data send: ', data.decode('UTF-8'))
            # 每500ms发送一次数据，直到收到rd
            myTimer.repeat(1000, 0.5, send_data, self.set_status, *(self.ser, data))
        elif self.status == 'pending':
            retSign = self.receive_data()
            # print('retSign: ', retSign)
            if retSign != None and retSign == 'r':
                self.status = 'fulfilled'
                myTimer.stop()
                print('data send successfully')
        elif self.status == 'rejected':
            self.status = 'fulfilled'
            print('data send failed')
        return self.status

    def send_data(self, data):
        try:
            print('data send: ', data.decode('UTF-8'))
        except:
            print('data send: ', data)
        return self.ser.write(data)

    def receive_data(self):
        data = self.ser.read_all()
        rec_str = None
        if data:
            rec_str = data.decode('UTF-8')
            print('receive data: ', rec_str)
        return rec_str

    def set_status(self, status):
        self.status = status


def open_serial_port():
    """
    打开串口
    :return: 返回串口是否打开，串口对象
    """
    # ser = serial.Serial('/dev/cu.usbserial-2140', 115200, timeout=0.5)
    ser = serial.Serial('/dev/ttyTHS1', 115200, timeout=0.01)
    if ser.isOpen():
        print("[sys] Serial port is open")
    else:
        print("[sys] Serial port is not open")
    return ser.isOpen(), ser


def close_serial_port(ser: serial.Serial):
    """
    关闭串口
    :param ser: 串口对象
    :return: 返回串口是否关闭
    """
    ser.close()
    if ser.isOpen():
        print("[sys] Serial port close failed")
    else:
        print("[sys] Serial port close successfully")
    return ser.isOpen()


def send_data(ser, data):
    """
    发送数据\n
    输入的数据为bytearray类型\n
    比如说bytearray([0x01])
    :param ser: 串口对象
    :param data: 发送的数据
    :return: 返回发送的数据长度
    """
    # print(data.decode('UTF-8'))
    return ser.write(data)


def receive_data(ser):
    """
    接收数据
    :param ser: 串口对象
    :return: 返回接收的数据
    """
    data = ser.read_all()
    if data:
        rec_str = data.decode('UTF-8')
        print(rec_str)


def send_data_async(ser, data):
    """
    异步发送数据\n
    输入的数据为bytearray类型\n
    比如说bytearray([0x01])
    :param ser: 串口对象
    :param data: 发送的数据
    :return: 返回发送的数据长度
    """
    pass
