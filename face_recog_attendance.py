import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas as pd

# ✅ Step 1: Load images from "Images" folder
path = 'Images'
images = []
names = []

print("🔍 Loading images from folder...")

if not os.path.exists(path):
    print("❌ 'Images' folder not found!")
    exit()

for filename in os.listdir(path):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(path, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Couldn't read image: {filename}")
            continue

        images.append(img)
        names.append(os.path.splitext(filename)[0])

print(f"✅ Loaded {len(images)} valid image(s).")

if len(images) == 0:
    print("❌ No valid face images found. Please add images to 'Images' folder.")
    exit()

# ✅ Step 2: Encode faces safely
def encode_faces(images):
    encodings = []
    for img in images:
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_encodings(img_rgb)
            if faces:
                encodings.append(faces[0])
                print("✅ Face encoded successfully.")
            else:
                print("⚠️ No face found in image.")
        except Exception as e:
            print(f"❌ Encoding error: {e}")
    return encodings

known_encodings = encode_faces(images)

# ✅ Step 3: Attendance logger
def mark_attendance(name):
    now = datetime.now()
    dt_string = now.strftime('%Y-%m-%d %H:%M:%S')

    if not os.path.exists('Attendance.csv'):
        with open('Attendance.csv', 'w') as f:
            f.write('Name,Time\n')

    df = pd.read_csv('Attendance.csv')
    if name not in df['Name'].values:
        with open('Attendance.csv', 'a') as f:
            f.write(f'{name},{dt_string}\n')
        print(f"📝 Marked attendance for {name}")

# ✅ Step 4: Try opening webcam from index 0 → 2
cap = None
for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"📸 Webcam opened successfully at index {i}")
        break

if not cap or not cap.isOpened():
    print("❌ Could not access any webcam.")
    exit()

# ✅ Step 5: Start recognition loop
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ Frame not captured. Retrying...")
        continue

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    try:
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    except:
        print("❌ Could not convert frame to RGB.")
        continue

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_loc in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        if len(face_distances) == 0:
            continue

        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = names[best_match_index].upper()
            y1, x2, y2, x1 = [v * 4 for v in face_loc]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mark_attendance(name)

    cv2.imshow("🎥 Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
