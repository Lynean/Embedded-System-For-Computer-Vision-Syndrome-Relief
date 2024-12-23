from numpy import zeros, array, object_, asarray
from mediapipe import Image, ImageFormat, solutions
from cv2 import cvtColor,COLOR_BGR2RGB, COLOR_RGB2BGR, flip, VideoCapture, line, circle, putText, destroyAllWindows, imshow, waitKey, FONT_HERSHEY_COMPLEX
from math import sqrt
import onnxruntime
from time import time, sleep
from joblib import load as jobload
from PIL.Image import fromarray

LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
LEFT_IRIS = [474,475,476,477]
RIGHT_IRIS = [469,470,471,472]
L_IRIS_CENTER = 473
R_IRIS_CENTER = 468
BLINK_TIME = 0.3

TOP = 10
BOTTOM = 199
LEFT = 50
RIGHT = 280
NOSE = 1

GAZE_WINDOW_FRAME_SIZE = 25

ALLOWED_LOOKING_TIME = 20*60



class Blink():
    def __init__(self, svm_model_path):
        # Load the trained SVM model
        self.model = jobload(svm_model_path)

        self.frame_count_started = False
        # create the tool
        self.EAR_values = []
        self.moving_sum = 0

        self.ear_buffer = []
        self.frame_count = 0
        self.blink_count = 0
        self.blink_rate = 0
        self.blink_sequence_count = 0
        self.blink_sequence_start = False
        self.start_time = 0
        # Initialize MediaPipe FaceMesh
        self.mp_face_mesh = solutions.face_mesh
        self.mesh_coord = zeros(500, dtype=object_)
    def euclidean_distance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        distance = sqrt((x2-x1)**2 + (y2-y1)**2)
        return distance
    def calculate_left_eye_EAR(self, mesh_points):
        numerator = 0
        denomenator = 0
        numerator += float(self.euclidean_distance(mesh_points[160], mesh_points[144]))
        numerator += float(self.euclidean_distance(mesh_points[158], mesh_points[153]))
        numerator += float(self.euclidean_distance(mesh_points[159], mesh_points[145]))
        numerator += float(self.euclidean_distance(mesh_points[161], mesh_points[163]))
        numerator += float(self.euclidean_distance(mesh_points[157], mesh_points[154]))
        denomenator += float(self.euclidean_distance(mesh_points[33], mesh_points[133]))
        return float((numerator)/(5*denomenator))


    def calculate_right_eye_EAR(self, mesh_points):
        numerator = 0
        denomenator = 0
        numerator += float(self.euclidean_distance(mesh_points[387], mesh_points[373]))
        numerator += float(self.euclidean_distance(mesh_points[385], mesh_points[380]))
        numerator += float(self.euclidean_distance(mesh_points[386], mesh_points[374]))
        numerator += float(self.euclidean_distance(mesh_points[384], mesh_points[381]))
        numerator += float(self.euclidean_distance(mesh_points[388], mesh_points[390]))
        denomenator += float(self.euclidean_distance(mesh_points[263], mesh_points[362]))
        return float((numerator)/(5*denomenator))
    # def landmarksDetection(self, meshmap, img, results):  # meshmap is an array to store the coords, img is the frame, results is the results from mediapipe facemesh detector
    #     img_height, img_width = img.shape[:2]  # get height and width of the frame
    #     for idx in LEFT_EYE + RIGHT_EYE + LEFT_IRIS + RIGHT_IRIS:
    #         point = results.face_landmarks[0][idx]  # results.multi_face_landmarks[0] access the first face in the results and the .landmark[idxex] access the coords of the eyes landmarks
    #         # point is an object that looks like this:
    #         # x: normalized_x_value
    #         # y: normalized_y_value
    #         # z: normalized_z_value
    #         # apparently we use point.x, point.y to access the coords
    #         meshmap[idx] = (int(point.x * img_width), int(point.y * img_height))  # converting the normalized coords into pixel coords and add it to the specified array
    #         # array[(x,y), (x,y)....] --> fill the previously initialized mesh_coord np array with pixel coords of the eye landmarks on the facemesh
    def detect_blinks(self, meshmap, framethresh = 12):

        self.mesh_coord = meshmap
        left_eye_EAR = self.calculate_left_eye_EAR(self.mesh_coord)
        right_eye_EAR = self.calculate_right_eye_EAR(self.mesh_coord)
        averaged_EAR = float(left_eye_EAR + right_eye_EAR) / 2

        #apply moving average filter with a width of 5 to smoothen the data
        filter_width = 2
        # If the number of values in the list is less than filter width, append the EAR value
        if len(self.EAR_values) < filter_width:
            self.EAR_values.append(averaged_EAR)
            self.moving_sum += averaged_EAR  # Add to moving sum
            # Compute the average based on the current number of values
            smoothen_value = self.moving_sum / len(self.EAR_values)
        else:
            # Maintain sliding window
            oldest_value = self.EAR_values.pop(0)  # Remove the oldest value
            self.moving_sum = self.moving_sum - oldest_value + averaged_EAR  # Update moving sum
            self.EAR_values.append(averaged_EAR)
            # Compute the moving average
            smoothen_value = round(self.moving_sum / filter_width,8)

        #store the averaged_EAR value in ear_values
        self.ear_buffer.append(smoothen_value)

        #start blink detection after ear_values have 13 value
        if len(self.ear_buffer) > 13:
            self.ear_buffer.pop(0)

        if len(self.ear_buffer) == 13:
            feature_vector = array(self.ear_buffer).reshape(1, -1)
            '''print(feature_vector)'''
            prediction = self.model.predict(feature_vector)
        else:
            prediction = 2

        if prediction == 1:
            self.frame_count_started = True
        if self.frame_count_started:
            self.frame_count += 1
        if self.frame_count_started and prediction == 0:
            self.frame_count_started = False
            if self.frame_count <= framethresh:
                self.blink_count += 1
                print("BLINK:", self.blink_count)
                #"Blink"
            self.frame_count = 0

class FaceOrientation():
    def __init__(self):
        self.calibrate = False
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

        self.HDistRatio = 0
        self.HDist_MaxRatio = 0 #Left eye HDist

        self.label = ""
        self.running = False
    def UpdateMax(self):
        maxVert = self.Vert
        maxHor = self.Hor
        self.maxRatio = maxHor / maxVert
        self.HDist_MaxRatio = self.HDistRatio
        self.minDistanceToNose = self.noseDistance
    def euclidean_distance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        distance = sqrt((x2-x1)**2 + (y2-y1)**2)
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

        LHDist = self.euclidean_distance(rightmost_left_coords, leftmost_left_coords)
        RHDist = self.euclidean_distance(rightmost_right_coords, leftmost_right_coords)
        LVDist = self.euclidean_distance(lower_left_coords, upper_left_coords)
        RVDist = self.euclidean_distance(lower_right_coords, upper_right_coords)

        self.LHRatio = self.euclidean_distance(left_iris_coords, leftmost_left_coords) / LHDist
        self.RHRatio = self.euclidean_distance(right_iris_coords, leftmost_right_coords) / RHDist

        self.LVRatio =  self.euclidean_distance(left_iris_coords, upper_left_coords) / LVDist
        self.RVRatio = self.euclidean_distance(right_iris_coords, upper_right_coords) / RVDist

        self.HDistRatio = LHDist / RHDist
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

        distance = ((-a)*P[0] + P[1] - c)/sqrt(a**2 + 1)
        return distance
    def Calibrate(self, frame,cap = None, window = None):
        doneCalibrating = False
        img_height, img_width  = frame.shape[:2]
        line(frame,(int(self.top[0]*img_width), int(self.top[1]*img_height)), (int(self.bottom[0]*img_width), int(self.bottom[1]*img_height)), (255,0,0), 1)
        line(frame,(int(self.left[0]*img_width), int(self.left[1]*img_height)), (int(self.right[0]*img_width), int(self.right[1]*img_height)), (0,0,255), 1)
        circle(frame,(int(self.nose[0]*img_width), int(self.nose[1]*img_height)),1,(0,255,0),5)
        circle(frame,(int(self.intersectX*img_width), int(self.intersectY*img_height)),1,(255,255),5)
        putText(frame,f"CURRENT_RATIO: {self.Ratio:.2f}",(int(img_width*0.1),int(img_height*0.30)), FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)
        putText(frame,f"MIN_NOSE_DISTANCE: {self.minDistanceToNose:.2f}",(int(img_width*0.1),int(img_height*0.35)),FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)
        #.putText(frame,f"{self.label}",(int(img_width*0.5),int(img_height*0.8)), .FONT_HERSHEY_COMPLEX, 1, (255,0,255), 5)
        if window == None:
            imshow("Display",frame)
            key = waitKey(1)
            if(key == ord("q")):
                destroyAllWindows()
                cap.release()
            elif(key == ord("s")):
                doneCalibrating = True
                self.UpdateMax()
                destroyAllWindows()
                cap.release()
            return doneCalibrating, None
        else:
            frame = cvtColor(frame,COLOR_BGR2RGB)
            frame = fromarray(frame)
            try:
                window.cameraImage.configure(light_image = frame, size = (img_width*0.5,img_height*0.5))
            except:
                cap.release()
            return None, None
    def Update(self, mesh):
        #Choose a random scaler from -0.2 to 0.2 which is the amount of "randomness" addition to the Hor (not the Vert because the normalized Hor is the hor to Vert ratio)
        #randomScaler = random.rand()*0.1*random.choice([-1,0,1])
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
        self.noseDistance = sqrt((self.noseDistanceY**2 + self.noseDistanceX**2))
        
        if self.calibrate:
            self.UpdateMax()
            self.calibrate = False
            return None
        #print(self.minDistanceToNose)
        #randomness
        self.Ratio = self.Ratio/(0.0000001+self.maxRatio)
        rand_Ratio =  self.Ratio + randomScaler*self.Ratio
        rand_minDistanceToNose = self.minDistanceToNose + randomScaler*self.minDistanceToNose
        return (rand_Ratio,
                rand_minDistanceToNose,
                self.LVRatio,
                self.LHRatio,
                self.RVRatio,
                self.RHRatio,
                #self.HDist_MaxRatio,
                self.HDistRatio,
                self.noseDistanceX,
                self.noseDistanceY,
                #self.nose[0],
                #self.nose[1]
                )

class Notification():
    def __init__(self, ann_model_path, svm_model_path):
        #Notification Queue
        self.notis = []
        #Blink model
        self.modelBlink = Blink(svm_model_path)
        self.mesh_coord = zeros(500, dtype=object_)
        self.start_time = time()
        #20-20-20 model
        self.model202020 = onnxruntime.InferenceSession(ann_model_path, providers=["CPUExecutionProvider"])

        self.Face = FaceOrientation()

        self.lookingFrameCount = 0
        self.timeMark = 0
        self.currentTime = 0
        self.lookingTime = 0
        self.gazeWindowFrame= [0]

        self.calibrated = False
    def array_to_image(self, array):
        return Image(image_format=ImageFormat.SRGB, data=asarray(array))
    #return suitable datatype for each model
    def prepareData(self,data_or_mesh, svm = False, ann = False):
        if(svm):
            meshmap = zeros(500, dtype=object_)
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
            data_np = array(data_or_mesh, dtype=float)
            #print(z_scores)
            ort_inputs = {self.model202020.get_inputs()[0].name: ([data_np.astype(float)])}
            #convert to tensors
            return ort_inputs
    def Update(self,mesh, blinkEnabled = False, noti20Enabled = False, calibrate = False, fps = 30):
        if(blinkEnabled):
            FRAME_THRESH = fps*BLINK_TIME
            #FRAME_THRESH = 12
            input_mesh = self.prepareData(mesh,svm = True)
            self.modelBlink.detect_blinks(input_mesh, framethresh=FRAME_THRESH)
        if(noti20Enabled):
            data = self.Face.Update(mesh)
            if(data != None) and (not calibrate):
                input_batch = self.prepareData(data, ann = True)
                ort_outs = self.model202020.run(None, input_batch)
                if ort_outs[0] >=0:
                    self.Face.label = "Looking"
                    #print("Looking")
                    self.lookingFrameCount += 1
                elif ort_outs[0] <0: 
                    self.Face.label = "Not Looking"
                    #print("Not Looking")
    def Face_Calibrate(self, detector, window = None):
        cap = VideoCapture(0)
        doneCalibrating = False
        while (cap.isOpened() and self.calibrated == False):
            success,frame = cap.read()
            if not success:
                print("Video stream disrupted")
                break

            frame = flip(frame, 1)
            frame.flags.writeable = False
            frame = cvtColor(frame, COLOR_BGR2RGB)
            image = self.array_to_image(frame)
            results = detector.detect(image)
            frame = cvtColor(frame, COLOR_RGB2BGR)
            #if exist a landmark      
            if results.face_landmarks:
                self.Update(results.face_landmarks[0], calibrate=True, noti20Enabled = True)
                if (window == None):
                    doneCalibrating, frame = self.Face.Calibrate(frame, cap)
                else:
                    self.Face.Calibrate(frame, cap, window)
            sleep(0.05)
        if doneCalibrating: self.calibrated = True
    def push_notification(self, blinkEnabled = False, noti20Enabled = False, ):
        self.currentTime = time()
        #print("****************",self.Face.label)
        neg = 1
        pos = 1
        if(noti20Enabled):
            updateLookingFrame = False
            #Prevent division by 0 error
            numToPop = 0
            #Get a 25s window frame moving
            for timeStamp in self.gazeWindowFrame:
                if timeStamp < 0:
                    neg += 1
                else:
                    pos += 1
                #Poping the excess element out, while checking their mark
                if self.currentTime - abs(timeStamp) > GAZE_WINDOW_FRAME_SIZE:
                    numToPop += 1
            if self.Face.label == "Looking":
                self.gazeWindowFrame.append(self.currentTime)
            else:
                self.gazeWindowFrame.append(self.currentTime*(-1))
            for i in range(numToPop):
                mark = abs(self.gazeWindowFrame.pop(0))
                #if this pop == last timeMark -> a whole window frame has been replaced
                if(mark == self.timeMark):
                    updateLookingFrame = True
                    #Set a new mark
                    self.timeMark = self.currentTime
            #print(len(self.gazeWindowFrame))
            #Calculate lookingTime
            noLookTime = (neg/(neg+pos))
            if updateLookingFrame:
                self.lookingTime += (1 - noLookTime)*(abs(self.gazeWindowFrame[-1]) - abs(self.gazeWindowFrame[0]))
                print("LookingTime:",self.lookingTime)

            #neg = 5; pos = 1

            #If Non looktime > 80% of 25s frame
            if noLookTime > 0.8 and (abs(self.gazeWindowFrame[-1]) - abs(self.gazeWindowFrame[0]) >= GAZE_WINDOW_FRAME_SIZE-1):
                lookingFrameCount = 0
                self.lookingTime = 0
                #print("You looked away !!!!!!!!!!!!!!!! Neg:",neg,"Pos:", pos)
            if self.lookingTime >= ALLOWED_LOOKING_TIME:
                self.lookingTime /= 2
                #print("Look Away for 20s")
                self.notis.append("20-20-20 Notification: You should look at least 20 meters away for at least 20 seconds!")
        #calculate blink rate: blink/min
        if(blinkEnabled):
            #calculate blink rate: blink/min
            elapsed_time = time() - self.start_time
            if elapsed_time >= 30:
                self.modelBlink.blink_rate = self.modelBlink.blink_count
                self.notis.append(f"You blinked {self.modelBlink.blink_rate} times")
                '''if len(blink_rates) >= 3:
                    removed = blink_rates.pop(0)
                    blink_rates.append(blink_rate)
                else:
                    blink_rates.append(blink_rate)'''
                self.modelBlink.blink_count = 0 # reset blink count for the next minute
                

                #if blink rate falls in the range from 9 to 17 per minute, flash warning
                if (self.modelBlink.blink_rate >= 0) and (self.modelBlink.blink_rate <=  6):
                    self.modelBlink.blink_rate = -1
                    #print("Blink drop")
                    self.notis.append("Blink Drop Warning: Your blink rate is a possible indicator of Computer Vision Syndrome! You should blink more!")
                self.start_time = time()   
    def reset202020(self):
        self.lookingFrameCount = 0
        self.totalFrameCount = 0
        self.currentTime = time()
        self.lookingTime = 0
        self.timeMark = 0
        self.gazeWindowFrame = [0]
        self.start_time = time()
    def resetBlink(self):
        self.modelBlink.blink_count = 0
