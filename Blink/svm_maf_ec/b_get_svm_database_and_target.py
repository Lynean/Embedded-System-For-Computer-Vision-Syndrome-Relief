# Define file paths
EAR_paths = [
    r"C:\Users\STVN\Desktop\PYTHON\Thuc_tap_BK\blink_notification\new_svm_maf_and_extracoords\_filtered_ear1.txt",
    r"C:\Users\STVN\Desktop\PYTHON\Thuc_tap_BK\blink_notification\new_svm_maf_and_extracoords\_filtered_ear2.txt",
    r"C:\Users\STVN\Desktop\PYTHON\Thuc_tap_BK\blink_notification\new_svm_maf_and_extracoords\_filtered_ear3.txt",
    r"C:\Users\STVN\Desktop\PYTHON\Thuc_tap_BK\blink_notification\new_svm_maf_and_extracoords\_filtered_ear4.txt",
    r"C:\Users\STVN\Desktop\PYTHON\Thuc_tap_BK\blink_notification\new_svm_maf_and_extracoords\_filtered_ear5.txt",
    r"C:\Users\STVN\Desktop\PYTHON\Thuc_tap_BK\blink_notification\new_svm_maf_and_extracoords\_filtered_ear6.txt",
    r"C:\Users\STVN\Desktop\PYTHON\Thuc_tap_BK\blink_notification\new_svm_maf_and_extracoords\_filtered_ear7.txt",
    r"C:\Users\STVN\Desktop\PYTHON\Thuc_tap_BK\blink_notification\new_svm_maf_and_extracoords\_filtered_ear8.txt",
]

tag_paths = [
    r"C:\Users\STVN\Desktop\PYTHON\Thuc_tap_BK\eyeblink8\1\26122013_223310_cam.tag",
    r"C:\Users\STVN\Desktop\PYTHON\Thuc_tap_BK\eyeblink8\2\26122013_224532_cam.tag",
    r"C:\Users\STVN\Desktop\PYTHON\Thuc_tap_BK\eyeblink8\3\26122013_230103_cam.tag",
    r"C:\Users\STVN\Desktop\PYTHON\Thuc_tap_BK\eyeblink8\4\26122013_230654_cam.tag",
    r"C:\Users\STVN\Desktop\PYTHON\Thuc_tap_BK\eyeblink8\8\27122013_151644_cam.tag",
    r"C:\Users\STVN\Desktop\PYTHON\Thuc_tap_BK\eyeblink8\9\27122013_152435_cam.tag",
    r"C:\Users\STVN\Desktop\PYTHON\Thuc_tap_BK\eyeblink8\10\27122013_153916_cam.tag",
    r"C:\Users\STVN\Desktop\PYTHON\Thuc_tap_BK\eyeblink8\11\27122013_154548_cam.tag",
]

target_path = r"c:\Users\STVN\Desktop\PYTHON\Thuc_tap_BK\blink_notification\zzz_target.txt"


# Read EAR files and create a nested list
EAR_lists = []
for path in EAR_paths:
    file_list = []
    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            try:
                file_list.append(float(line))
            except ValueError:
                file_list.append(line)
        EAR_lists.append(file_list)


# Read the tag files and create a nested list
def read_tag_file(path):
    tags = []
    tag_features = [
        "frameID",
        "blinkID",
        "NF",
        "LE_FC",
        "LE_NV",
        "RE_FC",
        "RE_NV",
        "F_X",
        "F_Y",
        "F_W",
        "F_H",
        "LE_LX",
        "LE_LY",
        "LE_RX",
        "LE_RY",
        "RE_LX",
        "RE_LY",
        "RE_RX",
        "RE_RY",
        "endln",
    ]
    with open(path, "r") as tag_file:
        data = tag_file.readlines()
        for line in data:
            line = line.strip().split(":")
            tag = {}
            for i, ele in enumerate(line):
                tag[tag_features[i]] = ele
            tags.append(tag)
    return tags


tag_lists = []
for path in tag_paths:
    tag_lists.append(read_tag_file(path))


'''# Read the target file and create a list
targets = []
with open(target_path, "r") as file:
    for line in file:
        target = list(map(int, line.split()))
        targets.append(target)'''


# get blink_frameID_lists and non_blink_framdeID from tag_lists
blink_frameID_lists = []
non_blink_frameID_lists = []
for tag_list in tag_lists:
    blink_frameID_list = []
    non_blink_frame_ID_list = []
    for frame in tag_list:
        if frame["blinkID"] != "-1":
            blink_frameID_list.append(int(frame["frameID"]))
        else:
            non_blink_frame_ID_list.append(int(frame["frameID"]))
    blink_frameID_lists.append(blink_frameID_list)
    non_blink_frameID_lists.append(non_blink_frame_ID_list)


# clarify the blink_frameID_lists and the non_blink_frameID_lists
# to make sure that no frame have the EAR of None
# and create a new clarified_blink_frameID_lists and clarified_non_blink_frameID_lists
clarified_blink_frameID_lists = []
for index, lst in enumerate(blink_frameID_lists):
    clarified_blink_frameID_list = []
    EAR_list = EAR_lists[index]
    for frameID in lst:
        try:
            if EAR_list[int(frameID)] != "None":
                clarified_blink_frameID_list.append(int(frameID))
        except IndexError:
            pass
    clarified_blink_frameID_lists.append(clarified_blink_frameID_list)

# clarifying the non_blink_frameID_lists 
clarified_non_blink_frameID_lists = []
for index, lst in enumerate(non_blink_frameID_lists):
    clarified_non_blink_frameID_list = []
    EAR_list = EAR_lists[index]
    #clarified_blink_frameID_list = clarified_blink_frameID_lists[index]
    for frameID in lst:
        try:
            if EAR_list[int(frameID)] != "None":
                clarified_non_blink_frameID_list.append(int(frameID))
        except IndexError:
            pass   
    clarified_non_blink_frameID_lists.append(clarified_non_blink_frameID_list)


'''# The negatives are those that are sampled from parts of the videos where no blink occurs, 
# with 5 frames spacing and 7 frames margin from the ground-truth blinks.
final_non_blink_frameID_lists = []
for lst in clarified_non_blink_frameID_lists:
    try:
        final_non_blink_frameID_list = lst[::7]
    except IndexError:
        pass
    final_non_blink_frameID_lists.append(final_non_blink_frameID_list)'''


# get database.data - 13 dimension support vector for each frame, starting with the blink frames
# first get database for clarified_blink_frameID_lists as well as blink_target
blink_database = []
for index, lst in enumerate(clarified_blink_frameID_lists):
    EAR_list = EAR_lists[index]
    for frameID in lst:
        frame_vector = EAR_list[frameID - 6 : frameID + 7]
        if len(frame_vector) == 13 and "None" not in frame_vector:  # only add the vectors with 13 elements
            blink_database.append(frame_vector)

# get database for non_blink_frameID_lists
non_blink_database = []
for index, lst in enumerate(clarified_non_blink_frameID_lists):
    EAR_list = EAR_lists[index]
    for frameID in lst:
        frame_vector = EAR_list[frameID - 6 : frameID + 7]
        if len(frame_vector) == 13 and "None" not in frame_vector:
            non_blink_database.append(frame_vector)

# join the blink_database and non_blink_database to get database and write to file
database = blink_database + non_blink_database

with open("b_svm_maf_ec_database.txt", "w") as file:
    for sublist in database:
        # Convert each sublist to a string where elements are space-separated
        line = " ".join(map(str, sublist))
        # Write the string to the file with a newline character
        file.write(line + "\n")


# get database.target - the target file
# every blink is 1 and every non_blink is 0
target = []
for i in range(len(blink_database)):
    target.append(1)
for j in range(len(non_blink_database)):
    target.append(0)

with open("b_svm_maf_ec_target.txt", "w") as file:
    for ele in target:
        # Convert each sublist to a string where elements are space-separated
        line = ele
        # Write the string to the file with a newline character
        file.write(str(line) + "\n")