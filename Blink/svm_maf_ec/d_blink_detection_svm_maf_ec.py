import joblib
import mediapipe as mp
import cv2 as cv
from scipy.spatial import (
    distance as dist,
) 
import numpy as np
import time

# Load the trained SVM model
model = joblib.load(r"C:\Users\STVN\Desktop\PYTHON\Thuc_tap_BK\blink_notification\new_svm_maf_and_extracoords\ear_svm_maf_ec_model.pkl")

EAR_values = []
moving_sum = 0

mp_face_mesh = mp.solutions.face_mesh
mesh_coord = np.zeros(500, dtype=np.object_)
LEFT_EYE = [
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
]
RIGHT_EYE = [
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
]

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

L_IRIS_CENTER = [473]
R_IRIS_CENTER = [468]

mesh_coord = np.zeros(500, dtype=np.object_)  # create a list with 500 elements initially having the value of 0
# each element is a python object, which means it can be a list, int, dict, ...


def frameProcess(frame, cvt_code):
    frame = cv.resize(frame, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)  # double the size of the frame
    frame_height, frame_width = frame.shape[:2]  # get the height and width of the frame
    output_frame = cv.cvtColor(frame, cvt_code)  # convert the frame into another specified color channel
    return frame_height, frame_width, output_frame

#create display window
win_name = "display"
cv.namedWindow(win_name, cv.WINDOW_NORMAL)

# load the tools
mp_face_mesh = mp.solutions.face_mesh

# create the tool
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh:
    
    ear_buffer = []
    frame_count = 0
    blink_count = 0
    blink_sequence_count = 0
    blink_sequence_start = False
    start_time = time.time()

    # take video stream from video path
    video_stream = cv.VideoCapture(0)

    # process frames one frame at a time
    while video_stream.isOpened():
        success, frame = video_stream.read()

        if not success:
            print("Video stream interupted")  # display error message if stream is interupted
            break

        frame = cv.flip(frame, 1)
        frameProcess(frame, cv.COLOR_BGR2RGB)  # double the frame height and width as well as convert frame to RGB. WHY RESIZE?

        # get resutls from face mesh, the result is an "object" that contained normalized cooridnates of the detected landmarks
        results = face_mesh.process(frame)

        # check if there are any detected FaceLandmarkerResult
        if results.multi_face_landmarks:
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            #convert results to 2D np array for easy accessing and modification. The normalized coords in results are also converted into pixel coords.
            #retrieve the height and width of frame for coords conversion
            img_h, img_w = frame.shape[:2]
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            #helper function to calculate EAR of an eye
            def calculate_left_eye_EAR():
                numerator = 0
                denomenator = 0
                numerator += float(dist.euclidean(mesh_points[160], mesh_points[144]))
                numerator += float(dist.euclidean(mesh_points[158], mesh_points[153]))
                numerator += float(dist.euclidean(mesh_points[159], mesh_points[145]))
                numerator += float(dist.euclidean(mesh_points[161], mesh_points[163]))
                numerator += float(dist.euclidean(mesh_points[157], mesh_points[154]))
                denomenator += float(dist.euclidean(mesh_points[33], mesh_points[133]))
                return float((numerator)/(5*denomenator))
            
            def calculate_right_eye_EAR():
                numerator = 0
                denomenator = 0
                numerator += float(dist.euclidean(mesh_points[387], mesh_points[373]))
                numerator += float(dist.euclidean(mesh_points[385], mesh_points[380]))
                numerator += float(dist.euclidean(mesh_points[386], mesh_points[374]))
                numerator += float(dist.euclidean(mesh_points[384], mesh_points[381]))
                numerator += float(dist.euclidean(mesh_points[388], mesh_points[390]))
                denomenator += float(dist.euclidean(mesh_points[263], mesh_points[362]))
                return float((numerator)/(5*denomenator))
            
            left_eye_EAR = calculate_left_eye_EAR()
            right_eye_EAR = calculate_right_eye_EAR()
            averaged_EAR = float(left_eye_EAR+right_eye_EAR)/2

            #apply moving average filter with a width of 5 to smoothen the data
            filter_width = 2
            # If the number of values in the list is less than filter width, append the EAR value
            if len(EAR_values) < filter_width:
                EAR_values.append(averaged_EAR)
                moving_sum += averaged_EAR  # Add to moving sum
                # Compute the average based on the current number of values
                smoothen_value = moving_sum / len(EAR_values)
            else:
                # Maintain sliding window
                oldest_value = EAR_values.pop(0)  # Remove the oldest value
                moving_sum = moving_sum - oldest_value + averaged_EAR  # Update moving sum
                EAR_values.append(averaged_EAR)
                # Compute the moving average
                smoothen_value = round(moving_sum / filter_width,4)

            #store the averaged_EAR value in ear_values
            ear_buffer.append(smoothen_value)

            #start blink detection after ear_values have 13 value
            if len(ear_buffer) > 13:
                ear_buffer.pop(0)

            if len(ear_buffer) == 13:
                feature_vector = np.array(ear_buffer).reshape(1, -1)
                '''print(feature_vector)'''
                prediction = model.predict(feature_vector)

                middle_frame_index = frame_count - 6
                
                if prediction == 1:
                    blink_sequence_start = True
                    blink_sequence_count += 1
                else:
                    blink_sequence_start = False     
                    if 3 <= blink_sequence_count <= 12: #this is a problem, blink interval varies
                        blink_count += 1
                        cv.putText(frame, 'Blink Detected', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
                    else:
                        cv.putText(frame, 'No Blink', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
                    blink_sequence_count = 0
                

            #calculate blink rate: blink/min 
            elapsed_time = time.time() - start_time
            blink_rate = round(float((blink_count/elapsed_time)*60), 1)

            #if blink rate falls in the range from 9 to 17 per minute, flash warning
            if blink_rate <=  9:
                cv.putText(frame, f'Blink rate too low.', (10, 300), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2, cv.LINE_AA)
                cv.putText(frame, f'Possible indication of CVS', (10, 400), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2, cv.LINE_AA)
                

            cv.putText(frame, f'Blink Count: {blink_count}', (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2, cv.LINE_AA)
            cv.putText(frame, f'Blink rate: {blink_rate}', (300, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2, cv.LINE_AA)

        cv.imshow(win_name, frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

video_stream.release()
cv.destroyAllWindows()