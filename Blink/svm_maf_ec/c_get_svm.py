import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# read database and create database list and convert to numpy array
with open(r"C:\Users\STVN\Desktop\PYTHON\Thuc_tap_BK\blink_notification\new_svm_maf_and_extracoords\b_svm_maf_ec_database.txt", "r") as file:
    database = []
    for line in file:
        feature_vector = list(map(float, line.strip().split()))
        database.append(feature_vector)
np_database = np.array(database)


# read target and create target list and convert to numpy array
with open(r"C:\Users\STVN\Desktop\PYTHON\Thuc_tap_BK\blink_notification\new_svm_maf_and_extracoords\b_svm_maf_ec_target.txt", "r") as file:
    target = []
    for line in file:
        try:
            label = int(line.strip())
            target.append(label)
        except ValueError as e:
            print(f"Warning: Skipping line due to error: {e}")
np_target = np.array(target)


'''#apply SMOTE for data better balancing
from imblearn.over_sampling import SMOTE

# Initialize SMOTE
sm = SMOTE(random_state=42)

# Apply SMOTE to the dataset
np_database_resampled, np_target_resampled = sm.fit_resample(np_database, np_target)'''


# split data into training set and teseting set
X_train, X_test, y_train, y_test = train_test_split(
    np_database, np_target, test_size=0.3, random_state=42
)


'''from sklearn.preprocessing import StandardScaler

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)'''


# Create a linear SVM classifier
clf = svm.SVC(kernel="linear", random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Evaluate the classifier
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))


import joblib

# Save the model to a file
joblib.dump(clf, 'ear_svm_maf_ec_model.pkl')

# Load the model from the file (for later use)
clf = joblib.load('ear_svm_maf_ec_model.pkl')