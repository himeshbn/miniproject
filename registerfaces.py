import cv2
import pickle
import numpy as np
import os

# Initialize the video capture
video = cv2.VideoCapture(0)
faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# List to store face data
facesData = []
i = 0

# Get the name of the user
name = input("Enter Your Name: ")

# Directory to save data
dataset_dir = 'D:\\Mini_project\\Dataset'
os.makedirs(dataset_dir, exist_ok=True)

# Number of images to capture
max_images = 5

# Fixed image dimensions for consistency
image_size = (50, 50)  # Resize all images to 50x50 pixels

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Unable to capture video.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cropImg = frame[y:y+h, x:x+w, :]
        resizedImg = cv2.resize(cropImg, image_size)

        # Append data at intervals of 10 frames
        if len(facesData) < max_images and i % 10 == 0:
            facesData.append(resizedImg)
        i += 1

        # Display count on the video frame
        cv2.putText(frame, f"Count: {len(facesData)}/{max_images}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 255, 50), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

    cv2.imshow("RegistrationWindow", frame)

    # Break condition
    k = cv2.waitKey(1)
    if k == ord('q') or len(facesData) == max_images:
        break

# Release the video capture and close the display window
video.release()
cv2.destroyAllWindows()

# Convert facesData to a NumPy array and reshape
facesData = np.asarray(facesData)
facesData = facesData.reshape(max_images, -1)

# Load or initialize the names and faces data
names_path = os.path.join(dataset_dir, 'names.pkl')
faces_path = os.path.join(dataset_dir, 'facesData.pkl')

if not os.path.exists(names_path):
    names = [name] * max_images
    with open(names_path, 'wb') as f:
        pickle.dump(names, f)
else:
    with open(names_path, 'rb') as f:
        names = pickle.load(f)
    names += [name] * max_images
    with open(names_path, 'wb') as f:
        pickle.dump(names, f)

if not os.path.exists(faces_path):
    with open(faces_path, 'wb') as f:
        pickle.dump(facesData, f)
else:
    with open(faces_path, 'rb') as f:
        faces = pickle.load(f)

    # Ensure consistent feature dimensions
    if faces.shape[1] != facesData.shape[1]:
        print(f"Dimension mismatch: Existing data ({faces.shape[1]}) vs New data ({facesData.shape[1]})")
        print("Please ensure all face images are processed with consistent dimensions.")
    else:
        faces = np.vstack((faces, facesData))  # Stack the new data vertically
        with open(faces_path, 'wb') as f:
            pickle.dump(faces, f)

print(f"Data for {max_images} face images saved successfully!")
