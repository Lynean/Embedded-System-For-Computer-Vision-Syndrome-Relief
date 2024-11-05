**User guide**

Optional: Create a virtualenv

1. Install requirements packages: pip install -r /path/to/requirements.txt

2. Run: python app.py

**Project details**

This repository contains the code for a research project "Embedded System for Computer Vision Syndrome Relief." This project aims to ultilize software and hardware to help prevent and treat computer vision syndrome. The system has 3 main features:

1. **_Blink rate drop notification:_** An Support Vector Machine (SVM) model is deployed to detect blink and calculate blink rate in real-time. If the blink rate drops below a threshold, which indicates that the user is likely to experience computer vision syndrome, the user is notified. The programs for this feature is located in the folder _blink_. We have developed 3 SVM models for blink detection. The first one is based on Soukupova T. & Cech J. 2016 paper "Eye blink detection using facial landmarks"; as proposed in the paper, blink detection is based on the Eye Aspect Ratio (EAR). The second one is an update of the first model: rather than using 3 eye coordinate pairs, we used 7 eye coordinate pairs. The last model is another update of the first one: for this model, we added a Moving Average Filter with a width of 2 to reduce noise when calculating EAR in real-time. We used 5 eye coordinate pairs to calculate EAR for this model. Regarding the accuracy, the models achieved 0.98, 0.95, and 0.98 respectively.

2. **_20-20-20 notification:_** An ANN model is deployed to perform gaze-tracking in real time to identify if the user is looking at the computer screen. Copies of an ANN was trained on the Columbia Gaze Data Set, Gaze360 dataset and custom made dataset. The copies performed equally well under normal testing condition, and slightly worse in strict conditions. After 20 minutes of undistrupted gazing (without more than 20 seconds of looking away), the program will automatically notify the user to look (at least 2 feet) away for at least 20 seconds. The package for this feature is located in the folder _Notification20_20_20_. 

3. **_Automatic brightness adjustment:_** Using the library screen_brightness_control

4. **_App:_** Every software feature of this project is collected into a complete application. The users are required to log in each time they use the application.

**Future plans**

In the near future, we are completing each feature. This project is expected to be completed in the end of this November (November 2024).
