import screen_brightness_control as pct
import serial
import serial.serialwin32
import serial.tools
import serial.tools.list_ports
import time
import numpy as np
from math import e



class ABL():
    def __init__(self):
        self.SerialPort = 0
        self.laptop_display_maxlux = 0
        self.setMaxLux(300)
        self.running = False
    def get_lux_value(self):
        line = self.SerialPort.readline()
        print(line)
        return float(line)

    def getCOMPort(self):
        portlst = serial.tools.list_ports.comports()
        for port in portlst:
            #print(port.usb_info())
            #print(port.description)
            #check if it's a STM or if using Arduino then change to the corresponding manufacturer
            #print(port.manufacturer)
            if(port.manufacturer == "wch.cn") or (port.manufacturer == "Arduino LLC (www.arduino.cc)"):
                #print("YESSSS")
                #open the connected serialport : Baudrate = 9600
                ser = serial.Serial(port.name, 9600)
                #ser.open()
                #print(ser.name)
                self.SerialPort = ser
                return ser
        self.SerialPort = 0
        return 0
            
    # Initialize serial connection

    def setMaxLux(self, nitval):
        # Get initial lux value
        # User input for maximum laptop display nit value
        self.laptop_display_maxlux= 4*(nitval*325)/300
    def serialFlush(self):
        serial.Serial.reset_input_buffer(self = self.SerialPort)
    def main(self):
        while self.running:
            x = self.get_lux_value()
            print(f"Lux: {x:.2f}")
            ln=np.log
            a=ln(x)
            y=(9.9323*a)+27.059
            print (x)
            if self.laptop_display_maxlux <= 3*x:
                output_value = 100
            if self.laptop_display_maxlux >= 3*x:
                output_value = y
                if output_value <=30:
                    output_value=30
            print(f"Current brightness: {pct.get_brightness()[0]}%")
            print(f"Calculated brightness: {output_value}%")

            pct.set_brightness(output_value, display=pct.list_monitors()[0])