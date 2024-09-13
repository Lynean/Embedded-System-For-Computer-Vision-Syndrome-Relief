import customtkinter
import time

import json
from PIL import Image

import GUI.gui as gui
import Notification20_20_20.Notification20_20_20 as Noti
from ABL.abl import ABL
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2 as cv
import threading
import numpy as np

customtkinter.set_default_color_theme("Oceanix.json")

def create_json():
    default = {"users": []}
    file = open("users.json",'w')
    json.dump(default, file, indent= None)
# function to add to JSON
def write_json(new_data, filename='users.json'):
    with open(filename,'r+') as file:
          # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data["users"].append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)


class AddUserWindow(customtkinter.CTkToplevel):
    def __init__(self,root, detector, notification, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = root
        self.geometry("500x500")
        self.title("Register User")

        self.notification = notification
        self.detector = detector

        self.userframe = customtkinter.CTkFrame(master=self)
        self.MaxNitframe = customtkinter.CTkFrame(master=self,width=300)
        self.cameraFrame = customtkinter.CTkFrame(master=self,width = 300)

        self.inputLabel = customtkinter.CTkLabel(self.userframe, text="Name")
        self.inputLabel.pack(side = "left")
        
        self.inputEntry = customtkinter.CTkEntry(self.userframe, width = 200, height= 35,corner_radius= 10, placeholder_text="Input Username")
        self.inputEntry.pack(side = "left")

        self.MaxNitLabel = customtkinter.CTkLabel(self.MaxNitframe, text="Screen Max Brightness (nit)")
        self.MaxNitLabel.pack(side = "left")
        
        self.MaxNitEntry = customtkinter.CTkEntry(self.MaxNitframe, width = 200, height= 35,corner_radius= 10, placeholder_text="300")
        self.MaxNitEntry.pack(side = "left")

        self.Valid = customtkinter.CTkLabel(self.userframe, text = "")
        self.Valid.pack(side = "left")

        self.RegButton = customtkinter.CTkButton(self, text ="Register", command = self.checkValid)
        self.cameraImage = customtkinter.CTkImage(light_image = Image.open("User.png"), size = (200, 200))
        self.cameraLabel = customtkinter.CTkLabel(self.cameraFrame,image= self.cameraImage, text = "", font=('Comic Sans MS', 12))
        self.cameraInstruction = customtkinter.CTkLabel(self.cameraFrame, text = "Look straight at the screen and press Enter", font=('Comic Sans MS', 12))
        
        self.userframe.pack(side = "top", anchor = "center",pady = 10)
        self.MaxNitframe.pack(side = "top", anchor = "center",pady = 10)
        self.RegButton.pack(side = "top", anchor = "center",pady = 10)
        self.grab_set()
    def checkValid(self):
        self.string = self.inputEntry.get()
        if((self.string in self.root.Users) or (self.string.strip() == "")):
            self.Valid.configure(text = "Username not available", text_color = "#FF0000")
        else:
            self.Valid.configure(text = "Username available", text_color = "#7FFC03")
            self.inputEntry.configure(state = "disable")
            self.MaxNitEntry.configure(state = "disable")

            self.cameraLabel.pack(side = "top")
            self.cameraInstruction.pack(side = "top")
            self.cameraFrame.pack(side = "top", anchor = "center",pady = 10)
            #Calibrate camera
            notification.calibrated = False
            calibrateProcess = threading.Thread(target = lambda: self.notification.Face_Calibrate(self.detector, window = self))
            calibrateProcess.daemon = True
            calibrateProcess.start()

            self.bind("<Return>", self.submit)
    def submit(self, event):
        maxNit = self.MaxNitEntry.get()
        try:
            maxNit = int(maxNit)
        except:
            maxNit = 300
        self.notification.calibrated = True
        self.notification.Face.UpdateMax()
        new_user = {"Username": self.inputEntry.get(), "ratio": self.notification.Face.maxRatio, "minDistance": self.notification.Face.minDistanceToNose, "MaxNit": maxNit}
        self.root.AddOptions(new_user)
        self.root.changeUser(new_user["Username"])
        self.root.User = new_user["Username"]
        self.root.optionUsers.configure(variable = customtkinter.StringVar(value=self.root.User))
        write_json(new_user)
        self.destroy()

class App(customtkinter.CTk):
    def __init__(self, detector, notification, *args, **kwargs):
        self.notification = notification
        self.detector = detector

        self.SerialABL = ABL()

        super().__init__(*args, **kwargs)
        self.geometry("600x400")
        self.title("CVSRS")
        self.resizable(False, False)
        #Get User List
        file = open("users.json")
        read_json = json.load(file)
        self.Users = ["Add User"]
        self.info = []
        for user in read_json["users"]:
            self.Users.append(user["Username"])
            self.info.append(user)

        #Class Variable
        self.blink = customtkinter.BooleanVar(value=False)
        self.noti20_20_20 = customtkinter.BooleanVar(value=False)
        self.ABL = customtkinter.BooleanVar(value=False)
        self.USB_available = customtkinter.BooleanVar(value = False)
    
        #Widget
        self.userFrame = customtkinter.CTkFrame(self)
        self.settingFrame = customtkinter.CTkFrame(self)
        self.cameraFrame = customtkinter.CTkFrame(self.settingFrame)
        self.usbFrame = customtkinter.CTkFrame(self.settingFrame)
        self.startFrame =  customtkinter.CTkFrame(self)

        self.UserLabel = customtkinter.CTkLabel(self.userFrame, text = "User:", font=('Comic Sans MS', 12))
        self.TitleLabel = customtkinter.CTkLabel(self.userFrame, text = "Computer Syndrome Relief System",font=('Comic Sans MS', 21, "bold"), text_color="#1fcbd1")
        self.CamAppLabel = customtkinter.CTkLabel(self.cameraFrame, text = "Camera",font=('Comic Sans MS', 21, "bold"), text_color="#1fcbd1")
        self.USBAppLabel = customtkinter.CTkLabel(self.usbFrame, text = "USB Connection",font=('Comic Sans MS', 21, "bold"), text_color="#1fcbd1")
        self.USBAvailabilityLabel = customtkinter.CTkLabel(self.usbFrame, text = "Device not detected", text_color="#ff0000")

        self.startBlinkLabel = customtkinter.CTkLabel(self.startFrame, text = "Blink disable", text_color= "#ff0000")
        self.startNoti20Label = customtkinter.CTkLabel(self.startFrame, text = "20-20-20 noti disabled", text_color= "#ff0000")
        self.startABLLabel = customtkinter.CTkLabel(self.startFrame, text = "ABL disabled", text_color= "#ff0000")

        self.User = customtkinter.StringVar(value="Unknown")
        self.optionUsers = customtkinter.CTkOptionMenu(self.userFrame,values=self.Users,command= self.changeUser,variable=self.User,font=('Comic Sans MS', 12))

        self.blinkButton = customtkinter.CTkSwitch(self.cameraFrame, variable= self.blink, onvalue=True, offvalue=False, text ="Blink drop notification", command = self.Refresh)
        self.noti20Button = customtkinter.CTkSwitch(self.cameraFrame, variable= self.noti20_20_20, onvalue=True, offvalue=False, text ="20-20-20 rule notification", command = self.Refresh)
        self.ABLButton = customtkinter.CTkSwitch(self.usbFrame, variable= self.ABL, onvalue=True, offvalue=False, text ="Adaptive Brightness", command = self.Refresh)
        self.ABLButton.configure(state = "disabled")
        self.startButton =  customtkinter.CTkButton(self.startFrame, text ="START", command = self.start)

        self.UserLabel.pack(side = "left", padx=10, pady=10)
        self.optionUsers.pack(side = "left", padx=0, pady=10)
        self.TitleLabel. pack(side = "left", padx=10, pady=10)


        self.CamAppLabel.pack(side="top",anchor = "center", padx=10, pady=0)
        self.blinkButton.pack(side = "left", anchor = "w", padx=10, pady=5)
        self.noti20Button.pack(side = "left", anchor = "e", padx=10, pady=5)

        self.USBAppLabel.pack(side="top",anchor = "center", padx=100, pady=0)
        self.USBAvailabilityLabel.pack(side = "left", anchor = "e", padx=10, pady=5)
        self.ABLButton.pack(side = "left", anchor = "w", padx=10, pady=5)


        self.startBlinkLabel.pack(side = "top", anchor = "w", padx=10, pady=5)
        self.startNoti20Label.pack(side = "top", anchor = "w", padx=10, pady=5)
        self.startABLLabel.pack(side = "top", anchor = "w", padx=10, pady=5)
        self.startButton.pack(side = "top", anchor = "center", padx=10, pady=5)

        self.cameraFrame.pack(side="top", anchor = "center", padx=10, pady=20)
        self.usbFrame.pack(side="top", anchor = "center", padx=10, pady=20)

        self.userFrame.pack(side="top",anchor = "center", padx=10, pady=10)
        self.settingFrame.pack(side="left", anchor = "n", padx=10, pady=20)
        self.startFrame.pack(side = "left", anchor = "n", padx=10, pady=20)

        self.toplevel_window = None

        self.ABLScan = threading.Thread(target = lambda: self.UpdateABL())
        self.ABLScan.daemon = True
        self.ABLScan.start()
    def changeUser(self, string):
        if string == "Add User":
            self.open_toplevel()
        else:
            file = open("users.json")
            read_json = json.load(file)
            for userinfo in read_json["users"]:
                if userinfo["Username"] == string:
                    print(userinfo)
                    self.notification.Face.maxRatio = float(userinfo["ratio"])
                    self.notification.Face.minDistanceToNose = float(userinfo["minDistance"])
                    self.SerialABL.setMaxLux(int(userinfo["MaxNit"]))
    def open_toplevel(self):
        if self.toplevel_window is None or not self.toplevel_window.winfo_exists():
            self.toplevel_window = AddUserWindow(root = self, notification= notification, detector = self.detector)  # create window if its None or destroyed
        else:
            self.toplevel_window.focus()  # if window exists focus it
    def AddOptions(self,new_user):
        self.Users.append(new_user["Username"])
        self.optionUsers.configure(values = self.Users)
    def Refresh(self):
        if self.blink.get():
            self.startBlinkLabel.configure(text = "Blink enable", text_color = "#7FFC03")
        else:
            self.startBlinkLabel.configure(text = "Blink disable", text_color = "#ff0000")
        if self.noti20_20_20.get():
            self.startNoti20Label.configure(text = "20-20-20 noti enable", text_color = "#7FFC03")
        else:
            self.startNoti20Label.configure(text = "20-20-20 noti disable", text_color = "#ff0000")
        if self.ABL.get():
            self.startABLLabel.configure(text = "ABL enable", text_color = "#7FFC03")
        else:
            self.startABLLabel.configure(text = "ABL disable", text_color = "#ff0000")
    def start(self):
        self.notification.running = True
        self.SerialABL.running = True
        getNoti = threading.Thread(target = lambda: taskUpdateNotification(self.detector, self.notification, blink = self.blink.get(), noti20 = self.noti20_20_20.get()))
        getSerial =  threading.Thread(target = lambda: getABL(self.SerialABL, self.ABL.get()))
        getNoti.daemon = True
        getSerial.daemon = True
        getNoti.start()
        getSerial.start()
        self.SerialABL.serialFlush()
        self.startButton.configure(command = self.stop, text = "STOP")
        self.blinkButton.configure(state = "disabled")
        self.noti20Button.configure(state = "disabled")
        self.optionUsers.configure(state = "disabled")
        self.ABLButton.configure(state = "disabled")
    def stop(self):
        self.notification.running = False
        self.SerialABL.running = False 
        self.startButton.configure(command = self.start, text = "START")
        self.blinkButton.configure(state = "normal")
        self.noti20Button.configure(state = "normal")
        self.ABLButton.configure(state = "normal")
        self.optionUsers.configure(state = "normal")
        self.notification.reset()
    def UpdateABL(self):
        while True:
            while (self.SerialABL.SerialPort) and ( not self.SerialABL.running):
                try:
                    self.SerialABL.getCOMPort()
                    
                except:
                    pass
                self.ABLButton.configure(state = "normal")
                self.USBAvailabilityLabel.configure(text = "Device detected", text_color = "#7FFC03")
            while( not self.SerialABL.SerialPort) and ( not self.SerialABL.running):
                if self.SerialABL.running:
                    break
                try:
                    self.SerialABL.getCOMPort()
                except:
                    pass
                self.ABL = customtkinter.BooleanVar(value=False)
                self.ABLButton.configure(state = "disabled", variable = self.ABL)
                self.Refresh()
                self.USBAvailabilityLabel.configure(text = "Device not detected", text_color = "#FF0000")
            time.sleep(0.5)

        
def get_processVideoCap(cap):
    if cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video stream disrupted")
            return None
        notification.totalFrameCount += 1
        frame = cv.flip(frame, 1)
        frame.flags.writeable = False
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        return frame

def face_detector_init(model_file):
    base_options = python.BaseOptions(model_asset_path=model_file)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=False,
                                        output_facial_transformation_matrixes=False,
                                        num_faces=1)
    return vision.FaceLandmarker.create_from_options(options)
    
def taskUpdateNotification(detector, notification, blink, noti20):
    if blink or noti20:
        cap = cv.VideoCapture(0)
        while(cap.isOpened() and notification.running):
            frame = get_processVideoCap(cap)
            if frame.any() != None:
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(frame))
                results = detector.detect(image)
                #if exist a landmark      
                if results.face_landmarks:
                    #print("frame")
                    notification.Update(results.face_landmarks[0], blinkEnabled = blink, noti20Enabled = noti20)
                    notification.show_notification(blinkEnabled = blink, noti20Enabled = noti20)
        cap.release()

def getABL(ABL, abl):
    if abl:
        print("ABL here")
        ABL.main()

notification = Noti.Notification(r"Notification20_20_20/GazeANN.pt", r"Blink/ear_svm_model.pkl")
detector = face_detector_init('face_landmarker.task')

app = App(detector ,notification)


app.mainloop()



