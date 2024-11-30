from customtkinter import set_default_color_theme, CTkToplevel, CTkFrame, CTkLabel, CTkButton, CTkEntry, CTkImage, StringVar, BooleanVar, CTk, CTkOptionMenu, CTkSwitch
from time import sleep

from json import dump, load
from PIL.Image import open as ImageOpen
from cv2 import cvtColor, CAP_PROP_FPS, COLOR_BGR2RGB, flip, VideoCapture
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarkerOptions, FaceLandmarker

from threading import Thread
from win11toast import toast

import Notification20_20_20.Notification20_20_20 as Noti
from ABL.abl import ABL

set_default_color_theme("lavender.json")

def create_json():
    default = {"users": []}
    file = open("users.json",'w')
    dump(default, file, indent= None)
# function to add to JSON
def write_json(new_data, filename='users.json'):
    with open(filename,'r+') as file:
          # First we load existing data into a dict.
        file_data = load(file)
        # Join new_data with file_data inside emp_details
        file_data["users"].append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to 
        dump(file_data, file, indent = 4)


class AddUserWindow(CTkToplevel):
    def __init__(self,root, detector, notification, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = root
        self.geometry("500x500")
        self.title("Register User")

        self.notification = notification
        self.detector = detector

        self.userframe = CTkFrame(master=self)
        self.MaxNitframe = CTkFrame(master=self,width=300)
        self.cameraFrame = CTkFrame(master=self,width = 300)

        self.inputLabel = CTkLabel(self.userframe, text="Name")
        self.inputLabel.pack(side = "left")
        
        self.inputEntry = CTkEntry(self.userframe, width = 200, height= 35,corner_radius= 10, placeholder_text="Input Username")
        self.inputEntry.pack(side = "left")

        self.MaxNitLabel = CTkLabel(self.MaxNitframe, text="Screen Max Brightness (nit)")
        self.MaxNitLabel.pack(side = "left")
        
        self.MaxNitEntry = CTkEntry(self.MaxNitframe, width = 200, height= 35,corner_radius= 10, placeholder_text="300")
        self.MaxNitEntry.pack(side = "left")

        self.Valid = CTkLabel(self.userframe, text = "")
        self.Valid.pack(side = "left")

        self.RegButton = CTkButton(self, text ="Register", command = self.checkValid)
        self.cameraImage = CTkImage(light_image = ImageOpen("User.png"), size = (200, 200))
        self.cameraLabel = CTkLabel(self.cameraFrame,image= self.cameraImage, text = "", font=('Comic Sans MS', 12))
        self.cameraInstruction = CTkLabel(self.cameraFrame, text = "Look straight at the screen and press Enter", font=('Comic Sans MS', 12))
        
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
            calibrateProcess = Thread(target = lambda: self.notification.Face_Calibrate(self.detector, window = self))
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
        self.root.optionUsers.configure(variable = StringVar(value=self.root.User))
        write_json(new_user)
        self.destroy()

class App(CTk):
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
        read_json = load(file)
        self.Users = ["Add User"]
        self.info = []
        for user in read_json["users"]:
            self.Users.append(user["Username"])
            self.info.append(user)

        #Class Variable
        self.blink = BooleanVar(value=False)
        self.noti20_20_20 = BooleanVar(value=False)
        self.ABL = BooleanVar(value=False)
        self.USB_available = BooleanVar(value = False)
    
        #Widget
        self.userFrame = CTkFrame(self)
        self.settingFrame = CTkFrame(self)
        self.cameraFrame = CTkFrame(self.settingFrame)
        self.usbFrame = CTkFrame(self.settingFrame)
        self.startFrame =  CTkFrame(self)

        self.UserLabel = CTkLabel(self.userFrame, text = "User:", font=('Courier New', 12))
        self.TitleLabel = CTkLabel(self.userFrame, text = "ESCVSR",font=('Courier New', 25, "bold"))
        self.CamAppLabel = CTkLabel(self.cameraFrame, text = "Camera",font=('Courier New', 21, "bold"))
        self.USBAppLabel = CTkLabel(self.usbFrame, text = "USB Connection",font=('Courier New', 21, "bold"))
        self.USBAvailabilityLabel = CTkLabel(self.usbFrame, text = "Device not detected", text_color="#ff0000")

        self.startBlinkLabel = CTkLabel(self.startFrame, text = "Blink disable", text_color= "#ff0000")
        self.startNoti20Label = CTkLabel(self.startFrame, text = "20-20-20 noti disabled", text_color= "#ff0000")
        self.startABLLabel = CTkLabel(self.startFrame, text = "ABL disabled", text_color= "#ff0000")

        self.User = StringVar(value="Unknown")
        self.optionUsers = CTkOptionMenu(self.userFrame,values=self.Users,command= self.changeUser,variable=self.User,font=('Courier New', 12))

        self.blinkButton = CTkSwitch(self.cameraFrame, variable= self.blink, onvalue=True, offvalue=False, text ="Blink drop notification", command = self.Refresh)
        self.noti20Button = CTkSwitch(self.cameraFrame, variable= self.noti20_20_20, onvalue=True, offvalue=False, text ="20-20-20 rule notification", command = self.Refresh)
        self.ABLButton = CTkSwitch(self.usbFrame, variable= self.ABL, onvalue=True, offvalue=False, text ="Adaptive Brightness", command = self.Refresh)
        self.ABLButton.configure(state = "disabled")
        self.startButton =  CTkButton(self.startFrame, text ="START", command = self.start)

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

        self.ABLScan = Thread(target = lambda: self.UpdateABL())
        self.ABLScan.daemon = True
        self.ABLScan.start()
    def changeUser(self, string):
        if string == "Add User":
            self.open_toplevel()
        else:
            file = open("users.json")
            read_json = load(file)
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

        getNoti = Thread(target = lambda: taskUpdateNotification(self.detector, self.notification, blink = self.blink.get(), noti20 = self.noti20_20_20.get()))
        getSerial =  Thread(target = lambda: getABL(self.SerialABL, self.ABL.get()))
        shownotiTask = Thread(target = lambda: self.displayNotification())
        
        shownotiTask.daemon = True
        getNoti.daemon = True
        getSerial.daemon = True
        notification.reset202020()
        notification.resetBlink()
        getNoti.start()
        getSerial.start()
        shownotiTask.start()
        try:
            self.SerialABL.serialFlush()
        except:
            pass
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
        self.notification.resetBlink()
        self.notification.reset202020()
    def UpdateABL(self):
        while True:
            while (self.SerialABL.SerialPort) and ( not self.SerialABL.running):
                try:
                    self.SerialABL.getCOMPort()
                    
                except:
                    pass
                self.ABLButton.configure(state = "normal")
                self.USBAvailabilityLabel.configure(text = "Device detected", text_color = "#7FFC03")
                sleep(0.05)
            while( not self.SerialABL.SerialPort) and ( not self.SerialABL.running):
                if self.SerialABL.running:
                    break
                try:
                    self.SerialABL.getCOMPort()
                except:
                    pass
                self.ABL = BooleanVar(value=False)
                self.ABLButton.configure(state = "disabled", variable = self.ABL)
                self.Refresh()
                self.USBAvailabilityLabel.configure(text = "Device not detected", text_color = "#FF0000")
                sleep(0.05)
            sleep(0.5)
    def displayNotification(self):
        def showNotiQueue(string):
            toast(  "ESCVSR Notifcation", 
                    string,
                    icon=r"C:\Users\STVN\Pictures\Saved Pictures\edx profile pic.jpg",
                    button={'activationType': 'protocol', 'arguments': 'https://google.com', 
                    'content': 'Open Google'})
        while(notification.running):
            if self.notification.notis != []:
                print(self.notification.notis)
                showNotiQueue(self.notification.notis.pop())
            sleep(1)
def get_processVideoCap(cap):
    if cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video stream disrupted")
            return None
        #notification.totalFrameCount += 1
        frame = flip(frame, 1)
        frame.flags.writeable = False
        frame = cvtColor(frame, COLOR_BGR2RGB)
        return frame

def face_detector_init(model_file):
    base_options = BaseOptions(model_asset_path=model_file)
    options = FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=False,
                                        output_facial_transformation_matrixes=False,
                                        num_faces=1)
    return FaceLandmarker.create_from_options(options)
    
def taskUpdateNotification(detector, notification, blink, noti20):
    if blink or noti20:
        cap = VideoCapture(0)
        fps = cap.get(CAP_PROP_FPS)
        while(cap.isOpened() and notification.running):
            frame = get_processVideoCap(cap)
            if frame.any() != None:
                image = notification.array_to_image(frame)
                results = detector.detect(image)
                #if exist a landmark      
                if results.face_landmarks:
                    #print("frame")
                    notification.Update(results.face_landmarks[0], blinkEnabled = blink, noti20Enabled = noti20, fps = fps)
                    notification.push_notification(blinkEnabled = blink, noti20Enabled = noti20)
            sleep(0.05)
        cap.release()

def getABL(ABL, abl):
    if abl:
        print("ABL here")
        ABL.main()

notification = Noti.Notification(r"Notification20_20_20/Gaze_ANN.onnx", r"Blink/ear_svm_maf_ec_model.pkl")
detector = face_detector_init('face_landmarker.task')

app = App(detector ,notification)


app.mainloop()



