from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os

# Initialize video capture and face detection
video = cv2.VideoCapture(0)
faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load preprocessed names and faces datasets
dataset_dir = 'D:\\Mini_project\\Dataset'
names_path = os.path.join(dataset_dir, 'names.pkl')
faces_path = os.path.join(dataset_dir, 'facesData.pkl')

with open(names_path, 'rb') as f:
    LABELS = pickle.load(f)

with open(faces_path, 'rb') as f:
    FACES = pickle.load(f)

# Print the shape of FACES to debug
print(f"Loaded face data shape: {FACES.shape}")  # Debugging line

# Ensure the number of features matches the dataset
n_features = FACES.shape[1]  # Get the correct number of features
print(f"Expected number of features per image: {n_features}")

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Video stream for live predictions
while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Unable to capture video.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cropImg = frame[y:y+h, x:x+w, :]
        resizedImg = cv2.resize(cropImg, (50, 50)).flatten().reshape(1, -1)

        # Check if the resized image matches dataset features
        if resizedImg.shape[1] != n_features:
            print(f"Resized image shape {resizedImg.shape} does not match dataset feature size {n_features}. Skipping...")
            continue

        # Make prediction using the KNN classifier
        output = knn.predict(resizedImg)

        # Display prediction on video feed
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)

    cv2.imshow("AttendanceWindow", frame)

    # Break condition
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()
