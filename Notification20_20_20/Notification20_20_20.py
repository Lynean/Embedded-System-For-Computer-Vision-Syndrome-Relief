import numpy as np
import mediapipe as mp
import cv2 as cv
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from PIL import Image
import joblib
from scipy.spatial import distance as dist
import webbrowser
from win11toast import toast


LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
LEFT_IRIS = [474,475,476,477]
RIGHT_IRIS = [469,470,471,472]
L_IRIS_CENTER = 473
R_IRIS_CENTER = 468

TOP = 10
BOTTOM = 199
LEFT = 50
RIGHT = 280
NOSE = 1

class Blink():
    def __init__(self, svm_model_path):
        # Load the trained SVM model
        self.model = joblib.load(svm_model_path)

        self.ear_buffer = []
        self.frame_count = 0
        self.blink_count = 0
        self.blink_rate = 0
        self.blink_sequence_count = 0
        self.blink_sequence_start = False
        self.start_time = 0
    
    def calculate_left_eye_EAR(self, mesh_points):
        numerator = 0
        denomenator = 0
        numerator += float(dist.euclidean(mesh_points[159], mesh_points[145]))
        numerator += float(dist.euclidean(mesh_points[158], mesh_points[153]))
        denomenator += float(dist.euclidean(mesh_points[33], mesh_points[133]))
        return float((numerator + numerator) / (2 * denomenator))  # return EAR of left eye as a float


    def calculate_right_eye_EAR(self, mesh_points):
        numerator = 0
        denomenator = 0
        
        numerator += float(dist.euclidean(mesh_points[386], mesh_points[374]))
        numerator += float(dist.euclidean(mesh_points[385], mesh_points[380]))
        denomenator += float(dist.euclidean(mesh_points[263], mesh_points[362]))
        return float((numerator + numerator) / (2 * denomenator))  # return EAR of right eye as a float

    def detect_blinks(self, meshmap):
       #create display window
        left_eye_EAR = self.calculate_left_eye_EAR(meshmap)
        right_eye_EAR = self.calculate_right_eye_EAR(meshmap)
        averaged_EAR = float(left_eye_EAR + right_eye_EAR) / 2

        #store the averaged_EAR value in ear_values
        self.ear_buffer.append(averaged_EAR)

        #start blink detection after ear_values have 13 value
        if len(self.ear_buffer) > 13:
            self.ear_buffer.pop(0)

        if len(self.ear_buffer) == 13:
            feature_vector = np.array(self.ear_buffer).reshape(1, -1)
            '''print(feature_vector)'''
            prediction = self.model.predict(feature_vector)
            middle_frame_index = self.frame_count - 6

            if prediction == 1:
                blink_sequence_start = True
                self.blink_sequence_count += 1
            else:
                blink_sequence_start = False
                if 3 <= self.blink_sequence_count <= 12: #this is a problem, blink interval varies
                    self.blink_count += 1
                    print("Blink")
                self.blink_sequence_count = 0

class FaceOrientation():
    def __init__(self):
        self.classification = -1
        self.maxRatio = 0 #Hor to Vert ratio when looking straight
        #Face
        self.Vert = 0 # Real Vertical distance relative to the frame size
        self.Hor = 0 # Real Horizontal distance relative to the frame size
        self.Ratio = 0

        self.top = (0,0)
        self.bottom = (0,0)
        self.left = (0,0)
        self.right = (0,0)
        self.nose = (0,0)
        self.intersectX = 0
        self.intersectY = 0
        self.noseDistance = 0
        self.noseDistanceX = 0
        self.noseDistanceY = 0
        self.minDistanceToNose = 100 #Distance from root-point to nose
        #Iris Ratio
        self.LVRatio = 0 #Left eye Vertical
        self.LHRatio = 0 #Left eye Horizontal
        self.RVRatio = 0 #Right eye Vertical
        self.RHRatio = 0 #Right eye Horizontal

        self.label = ""
        self.running = False
    def UpdateMax(self):
        maxVert = self.Vert
        maxHor = self.Hor
        self.maxRatio = maxHor / maxVert
        self.minDistanceToNose = self.noseDistance
    def euclidean_distance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        return distance
    def Get_Hor_Vert_Ratio(self, top, bottom, left, right):
        self.Vert = self.euclidean_distance(top, bottom)
        self.Hor = self.euclidean_distance(right, left)
        self.Ratio = self.Hor/self.Vert
    def Get_Iris_Ratio(self, mesh):
        left_iris_coords = (mesh[L_IRIS_CENTER].x,mesh[L_IRIS_CENTER].y )
        right_iris_coords = (mesh[R_IRIS_CENTER].x,mesh[R_IRIS_CENTER].y )

        leftmost_left_coords, rightmost_left_coords = (mesh[362].x,mesh[362].y ), (mesh[263].x,mesh[263].y )
        leftmost_right_coords, rightmost_right_coords = (mesh[33].x,mesh[33].y ),(mesh[133].x,mesh[133].y )

        upper_left_coords, lower_left_coords = (mesh[386].x,mesh[386].y ),(mesh[374].x,mesh[374].y )
        upper_right_coords, lower_right_coords = (mesh[159].x,mesh[159].y ),(mesh[145].x,mesh[145].y )

        self.LHRatio = self.euclidean_distance(left_iris_coords, leftmost_left_coords) / self.euclidean_distance(rightmost_left_coords, leftmost_left_coords)
        self.RHRatio = self.euclidean_distance(right_iris_coords, leftmost_right_coords) / self.euclidean_distance(rightmost_right_coords, leftmost_right_coords)

        self.LVRatio =  self.euclidean_distance(left_iris_coords, upper_left_coords) / self.euclidean_distance(lower_left_coords, upper_left_coords)
        self.RVRatio = self.euclidean_distance(right_iris_coords, upper_right_coords) / self.euclidean_distance(lower_right_coords, upper_right_coords)
    def line_intersection(self, p1,p2, P1, P2):
        xdiff = (p1[0] - p2[0], P1[0] - P2[0])
        ydiff = (p1[1] - p2[1], P1[1] - P2[1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = (det(p1,p2), det(P1, P2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y
    def distance_to_line(self, p1, p2, P):
        #y = ax + c ->  - ax + y - c = 0
        a = (p1[1] - p2[1])/(p1[0] - p2[0])
        c = p1[1] - a*p1[0]

        distance = ((-a)*P[0] + P[1] - c)/math.sqrt(a**2 + 1)
        return distance
    def Calibrate(self, frame,cap = None, window = None):
        doneCalibrating = False
        img_height, img_width  = frame.shape[:2]
        cv.line(frame,(int(self.top[0]*img_width), int(self.top[1]*img_height)), (int(self.bottom[0]*img_width), int(self.bottom[1]*img_height)), (255,0,0), 1)
        cv.line(frame,(int(self.left[0]*img_width), int(self.left[1]*img_height)), (int(self.right[0]*img_width), int(self.right[1]*img_height)), (0,0,255), 1)
        cv.circle(frame,(int(self.nose[0]*img_width), int(self.nose[1]*img_height)),1,(0,255,0),5)
        cv.circle(frame,(int(self.intersectX*img_width), int(self.intersectY*img_height)),1,(255,255),5)

        cv.putText(frame,f"MAX_RATIO: {self.maxRatio:.2f}",(int(img_width*0.1),int(img_height*0.25)),cv.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)
        cv.putText(frame,f"CURRENT_RATIO: {self.Ratio:.2f}",(int(img_width*0.1),int(img_height*0.30)), cv.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)
        cv.putText(frame,f"MIN_NOSE_DISTANCE: {self.minDistanceToNose:.2f}",(int(img_width*0.1),int(img_height*0.35)),cv.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)
        cv.putText(frame,f"CURRENT_NOSE_DISTANCE: {self.noseDistance:.2f}",(int(img_width*0.1),int(img_height*0.40)), cv.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)
        cv.putText(frame,f"Left_H: {self.LHRatio:.2f}",(int(img_width*0.7),int(img_height*0.30)), cv.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)
        cv.putText(frame,f"Left_V: {self.LVRatio:.2f}",(int(img_width*0.7),int(img_height*0.35)), cv.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)
        cv.putText(frame,f"Right_H: {self.RHRatio:.2f}",(int(img_width*0.7),int(img_height*0.4)), cv.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)
        cv.putText(frame,f"Right_V: {self.RVRatio:.2f}",(int(img_width*0.7),int(img_height*0.45)), cv.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)
        #cv.putText(frame,f"{self.label}",(int(img_width*0.5),int(img_height*0.8)), cv.FONT_HERSHEY_COMPLEX, 1, (255,0,255), 5)
        if window == None:
            cv.imshow("Display",frame)
            key = cv.waitKey(1)
            if(key == ord("q")):
                cv.destroyAllWindows()
                cap.release()
            elif(key == ord("s")):
                doneCalibrating = True
                self.UpdateMax()
                cv.destroyAllWindows()
                cap.release()
            return doneCalibrating, None
        else:
            frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            try:
                window.cameraImage.configure(light_image = frame, size = (img_width*0.5,img_height*0.5))
            except:
                cap.release()
            return None, None
    def Update(self, mesh):
        #Choose a random scaler from -0.2 to 0.2 which is the amount of "randomness" addition to the Hor (not the Vert because the normalized Hor is the hor to Vert ratio)
        randomScaler = 0
        self.top = (mesh[TOP].x, mesh[TOP].y)
        self.bottom = (mesh[BOTTOM].x, mesh[BOTTOM].y)
        self.left = (mesh[LEFT].x, mesh[LEFT].y)
        self.right = (mesh[RIGHT].x, mesh[RIGHT].y)
        self.nose = (mesh[NOSE].x, mesh[NOSE].y)

        self.Get_Hor_Vert_Ratio(self.top, self.bottom, self.left, self.right)
        self.Get_Iris_Ratio(mesh)
        
        self.intersectX, self.intersectY = (self.line_intersection(self.top,self.bottom,self.left,self.right))
        self.noseDistanceY = self.distance_to_line(self.left, self.right,self.nose) / self.Vert
        self.noseDistanceX = self.distance_to_line(self.top, self.bottom,self.nose) / self.Hor
        self.noseDistance = math.sqrt((self.noseDistanceY**2 + self.noseDistanceX**2))
        #randomness
        rand_MaxRatio = self.maxRatio + randomScaler*self.maxRatio
        rand_Ratio = self.Ratio + randomScaler*self.Ratio
        rand_minDistanceToNose = self.minDistanceToNose + randomScaler*self.minDistanceToNose

        return (rand_MaxRatio,
                rand_Ratio,
                rand_minDistanceToNose,
                self.LVRatio,
                self.LHRatio,
                self.RVRatio,
                self.RHRatio,
                self.noseDistanceX,
                self.noseDistanceY,
                self.nose[0],
                self.nose[1])

class ANNModel(nn.Module):
    def __init__(self):
        super().__init__()

        ### layers
        self.input  = nn.Linear(11,256)

        self.h1 = nn.Linear(256,1024)  # hidden layer
        self.bnorm1 = nn.BatchNorm1d(256)

        self.h2 = nn.Linear(1024,1024)  # hidden layer
        self.bnorm2 = nn.BatchNorm1d(1024)

        self.h3 = nn.Linear(1024,1024)
        self.bnorm3 = nn.BatchNorm1d(1024)

        self.h4 = nn.Linear(1024,1024)
        self.bnorm4 = nn.BatchNorm1d(1024)

        self.h5 = nn.Linear(1024,105)
        self.bnorm5 = nn.BatchNorm1d(1024)

        self.output = nn.Linear(105, 1)

        # forward pass
    def predict(self,x):

        # pass the data through the input layer
        x = F.relu( self.input(x) )
        # pass the data through the hidden layer
        x = self.bnorm1(x)
        x = F.relu( self.h1(x) )

        x = self.bnorm2(x)
        x = F.relu( self.h2(x) )

        x = self.bnorm3(x)
        x = F.relu( self.h3(x) )

        x = self.bnorm4(x)
        x = F.relu( self.h4(x) )

        x = self.bnorm5(x)
        x = F.relu( self.h5(x) )
        # output layer
        x = self.output(x)
        x = F.sigmoid(x)
        # no dropout here!!
        return x
    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()

class Notification():
    def __init__(self, ann_model_path, svm_model_path):
        #Blink model
        self.modelBlink = Blink(svm_model_path)
        self.mesh_coord = np.zeros(500, dtype=np.object_)
        #20-20-20 model
        self.model202020 = ANNModel()
        self.model202020.load_model(ann_model_path)

        self.Face = FaceOrientation()
        self.lookingFrameCount = 0
        self.totalFrameCount = 0
        self.currentTime = 0
        self.lookingTime = 0
        self.calibrated = False
    #return suitable datatype for each model
    def prepareData(self,data_or_mesh, svm = False, ann = False):
        if(svm):
            meshmap = np.zeros(500, dtype=np.object_)
            for idx in LEFT_EYE + RIGHT_EYE + LEFT_IRIS + RIGHT_IRIS:
                point = data_or_mesh[idx]  # results.multi_face_landmarks[0] access the first face in the results and the .landmark[idxex] access the coords of the eyes landmarks
                # point is an object that looks like this:
                # x: normalized_x_value
                # y: normalized_y_value
                # z: normalized_z_value
                # apparently we use point.x, point.y to access the coords
                meshmap[idx] = (point.x, point.y)  # converting the normalized coords into pixel coords and add it to the specified array
                # array[(x,y), (x,y)....] --> fill the previously initialized mesh_coord np array with pixel coords of the eye landmarks on the facemesh
            #print(meshmap)
            return meshmap
        if(ann):
            data_np = np.array(data_or_mesh, dtype=float)
            mean = np.mean(data_np)
            std_dev = np.std(data_np)
            z_scores = (data_np - mean) / std_dev
            #print(z_scores)

            #convert to tensors
            tensor = torch.tensor(z_scores).float()
            input_batch = tensor.unsqueeze(0)
            return input_batch
    def Update(self,mesh, blinkEnabled = False, noti20Enabled = False, calibrate = False):
        if(blinkEnabled):
            input_mesh = self.prepareData(mesh,svm = True)
            self.modelBlink.detect_blinks(input_mesh)
        if(noti20Enabled):
            data = self.Face.Update(mesh)
            if(data != None) and (not calibrate):
                input_batch = self.prepareData(data, ann = True)
                prediction = 0
                with torch.no_grad():
                    prediction = self.model202020.predict(input_batch)
                if prediction >=.5:
                    self.Face.label = "Looking"
                    #print("Looking")
                    self.lookingFrameCount += 1
                elif prediction <.5:
                    self.Face.label = "Not Looking"
                    #print("Not Looking")
    def Face_Calibrate(self, detector, window = None):
        cap = cv.VideoCapture(0)
        doneCalibrating = False
        while (cap.isOpened() and self.calibrated == False):
            success,frame = cap.read()
            if not success:
                print("Video stream disrupted")
                break

            frame = cv.flip(frame, 1)
            frame.flags.writeable = False
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(frame))
            results = detector.detect(image)
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            #if exist a landmark      
            if results.face_landmarks:
                self.Update(results.face_landmarks[0], calibrate=True, noti20Enabled = True)
                if (window == None):
                    doneCalibrating, frame = self.Face.Calibrate(frame, cap)
                else:
                    self.Face.Calibrate(frame, cap, window)

        if doneCalibrating: self.calibrated = True
    def show_notification(self, blinkEnabled = False, noti20Enabled = False):
        if self.currentTime == 0:
            self.currentTime = time.time()
        delta_time = time.time() - self.currentTime
        if(delta_time >= 10):
            if(noti20Enabled):
                percentage = self.lookingFrameCount / self.totalFrameCount
                print(f"You have been looking at the screen {(percentage*100):.2f}% of the last 20s")
                self.lookingTime += delta_time*percentage
                print(f"LookingTime: {self.lookingTime:.2f} seconds")
                if(self.lookingTime >= 20*60):
                    print("I advise you to look way for a while")
                    self.lookingTime = 0
                self.lookingFrameCount = 0
                self.totalFrameCount = 0
            #calculate blink rate: blink/min
            if(blinkEnabled):
                self.modelBlink.blink_rate = round(float((self.modelBlink.blink_count/delta_time)*60), 1)
                if self.modelBlink.blink_rate <=  9:
                    print("Blink rate to low, possible indication of CVS")
                print(f'Blink Count: {self.modelBlink.blink_count}')
                print(f'Blink rate: {self.modelBlink.blink_rate}')
                

                self.modelBlink.blink_count = 0
            toast("Title", "Message",icon=r"C:\Users\STVN\Pictures\Saved Pictures\edx profile pic.jpg",button={'activationType': 'protocol', 'arguments': 'https://google.com', 'content': 'Open Google'})
            self.currentTime = time.time()
    def reset(self):
        self.lookingFrameCount = 0
        self.totalFrameCount = 0
        self.blink_count = 0
        self.currentTime = time.time()
        self.lookingTime = 0
