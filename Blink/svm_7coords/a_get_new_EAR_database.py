paths = [
    r"C:\Users\STVN\Desktop\PYTHON\Thuc_tap_BK\eyeblink8\1\26122013_223310_cam.avi",
    r"C:\Users\STVN\Desktop\PYTHON\Thuc_tap_BK\eyeblink8\2\26122013_224532_cam.avi",
    r"C:\Users\STVN\Desktop\PYTHON\Thuc_tap_BK\eyeblink8\3\26122013_230103_cam.avi",
    r"C:\Users\STVN\Desktop\PYTHON\Thuc_tap_BK\eyeblink8\4\26122013_230654_cam.avi",
    r"C:\Users\STVN\Desktop\PYTHON\Thuc_tap_BK\eyeblink8\8\27122013_151644_cam.avi",
    r"C:\Users\STVN\Desktop\PYTHON\Thuc_tap_BK\eyeblink8\9\27122013_152435_cam.avi",
    r"C:\Users\STVN\Desktop\PYTHON\Thuc_tap_BK\eyeblink8\10\27122013_153916_cam.avi",
    r"C:\Users\STVN\Desktop\PYTHON\Thuc_tap_BK\eyeblink8\11\27122013_154548_cam.avi",
]

import mediapipe as mp
import cv2 as cv
import numpy as np
import math
from scipy.spatial import (
    distance as dist,
)  # this allows us to calulate the Euclidean distance between 2 point (not x1-x2!!!)

frame_counter = 0
CEF_COUNTER = 0
TOTAL_BLINKS = 0
ALPHA = 0.5
# constants

# Minimum number of frame that the eye shall be closed in order to be count as a blink
CLOSED_EYES_FRAME = 1
FONTS = cv.FONT_HERSHEY_COMPLEX

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

mesh_coord = np.zeros(
    500, dtype=np.object_
)  # create a list with 500 elements initially having the value of 0
# each element is a python object, which means it can be a list, int, dict, ...


def frameProcess(frame, cvt_code):
    frame = cv.resize(
        frame, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC
    )  # double the size of the frame
    frame_height, frame_width = frame.shape[:2]  # get the height and width of the frame
    output_frame = cv.cvtColor(
        frame, cvt_code
    )  # convert the frame into another specified color channel
    return frame_height, frame_width, output_frame


def landmarksDetection(
    meshmap, img, results, draw=False
):  # meshmap is an array to store the coords, img is the frame, results is the results from mediapipe facemesh detector
    img_height, img_width = img.shape[:2]  # get height and width of the frame
    for idx in LEFT_EYE + RIGHT_EYE + LEFT_IRIS + RIGHT_IRIS:
        point = results.multi_face_landmarks[0].landmark[
            idx
        ]  # results.multi_face_landmarks[0] access the first face in the results and the .landmark[idxư access the coords of the eyes landmarks
        # point is an object that looks like this:
        # x: normalized_x_value
        # y: normalized_y_value
        # z: normalized_z_value
        # apparently we use point.x, point.y to access the coords
        meshmap[idx] = (
            int(point.x * img_width),
            int(point.y * img_height),
        )  # converting the normalized coords into pixel coords and add it to the specified array
        # array[(x,y), (x,y)....] --> fill the previously initialized mesh_coord np array with pixel coords of the eye landmarks on the facemesh


def calculate_left_eye_EAR(mesh_points):
    numerator = 0
    denomenator = 0
    numerator += float(dist.euclidean(mesh_points[159], mesh_points[145]))
    numerator += float(dist.euclidean(mesh_points[158], mesh_points[153]))
    numerator += float(dist.euclidean(mesh_points[157], mesh_points[154]))
    numerator += float(dist.euclidean(mesh_points[173], mesh_points[155]))
    numerator += float(dist.euclidean(mesh_points[160], mesh_points[144]))
    numerator += float(dist.euclidean(mesh_points[161], mesh_points[163]))
    numerator += float(dist.euclidean(mesh_points[246], mesh_points[7]))
    denomenator += float(dist.euclidean(mesh_points[33], mesh_points[133]))
    return float((numerator)/(7*denomenator))


def calculate_right_eye_EAR(mesh_points):
    numerator = 0
    denomenator = 0
    numerator += float(dist.euclidean(mesh_points[386], mesh_points[374]))
    numerator += float(dist.euclidean(mesh_points[387], mesh_points[373]))
    numerator += float(dist.euclidean(mesh_points[388], mesh_points[390]))
    numerator += float(dist.euclidean(mesh_points[466], mesh_points[249]))
    numerator += float(dist.euclidean(mesh_points[385], mesh_points[380]))
    numerator += float(dist.euclidean(mesh_points[384], mesh_points[381]))
    numerator += float(dist.euclidean(mesh_points[398], mesh_points[382]))
    denomenator += float(dist.euclidean(mesh_points[263], mesh_points[362]))
    return float((numerator)/(7*denomenator))


def get_video_EAR(video_path):
    counts = 0
    print(counts)

    video_EAR = []

    # load the tools
    mp_face_mesh = mp.solutions.face_mesh

    # create the tool
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        # take video stream from video path
        video_stream = cv.VideoCapture(video_path)

        # process frames one frame at a time
        while video_stream.isOpened():
            counts += 1
            print(counts)  # print frame count

            EAR_list = []

            success, frame = video_stream.read()
            if not success:
                print(
                    "Video stream interupted"
                )  # display error message if stream is interupted
                break

            frameProcess(
                frame, cv.COLOR_BGR2RGB
            )  # double the frame height and width as well as convert frame to RGB. WHY RESIZE?

            # get resutls from face mesh, the result is an "object" that contained normalized cooridnates of the detected landmarks
            results = face_mesh.process(frame)

            # check if there are any detected FaceLandmarkerResult
            if results.multi_face_landmarks:
                """convert results to 2D np array for easy accessing and modification. The normalized coords in results are also converted into pixel coords.
                retrieve the height and width of frame for coords conversion
                img_h, img_w = frame.shape[:2]
                #mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
                """

                landmarksDetection(
                    mesh_coord, frame, results, False
                )  # fill in the mesh_coord array with pixel coords of eyes landmarks on the facemesh from the result
                # print(mesh_coord)

                left_eye_EAR = calculate_left_eye_EAR(mesh_coord)
                right_eye_EAR = calculate_right_eye_EAR(mesh_coord)
                averaged_EAR = float(left_eye_EAR + right_eye_EAR) / 2

                """EAR_list.append(left_eye_EAR)
                    EAR_list.append(right_eye_EAR)
                    EAR_list.append(averaged_EAR)
                    video_EAR.append(EAR_list)"""

                video_EAR.append(averaged_EAR)
            else:
                video_EAR.append("None")

    video_stream.release()
    cv.destroyAllWindows()

    return video_EAR


# Example usage:
"""ear_results = get_video_EAR(paths[0])
print(ear_results)
f = open(f"ear1.txt", "w")
for line in ear_results:
    f.write(f"{line}\n")
f.close()"""

for index, video_path in enumerate(paths):
    ear_results = get_video_EAR(video_path)
    f = open(f"new_ear{index+1}.txt", "w")
    for line in ear_results:
        f.write(f"{line}\n")
    f.close()
