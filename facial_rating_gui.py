

#it is for calculating the overall rating based on the the every second data
#multiple faces recognitized at atime


import cv2
import csv
from datetime import datetime
import os
import tkinter as tk
from PIL import Image, ImageTk
from keras.models import load_model
from utils import preprocess_face, predict_emotion  # Make sure these work for all emotions

# Constants
MODEL_PATH = "test.h5"
CSV_FILE = "ratings_log.csv"

# Load model and face detector
model = load_model(MODEL_PATH)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# GUI Setup
root = tk.Tk()
root.title("⭐ Facial Emotion-Based Star Rating ⭐")
root.geometry("900x750")
root.configure(bg="black")

tk.Label(root, text="⭐ Facial Emotion-Based Star Rating ⭐", font=("Arial", 22), bg="black", fg="white").pack(pady=10)

video_label = tk.Label(root)
video_label.pack()

rating_display = tk.Text(root, font=("Arial", 14), height=8, width=60, bg="black", fg="lime", borderwidth=0)
rating_display.pack(pady=10)

overall_rating_label = tk.Label(root, font=("Arial", 16), bg="black", fg="gold")
overall_rating_label.pack(pady=10)

cap = cv2.VideoCapture(0)
logged_faces = set()


def draw_star_rating(frame, x, y, stars):
    for i in range(5):
        color = (0, 255, 255) if i < stars else (100, 100, 100)
        cv2.circle(frame, (x + i * 20, y), 8, color, -1)


def emotion_to_stars(emotion):
    mapping = {
        'Happy': 5,
        'Surprise': 4,
        'Neutral': 3,
        'Sad': 2,
        'Fear': 1,
        'Angry': 1,
        'Disgust': 1
    }
    return mapping.get(emotion, 3)


def log_to_csv(person_id, emotion, stars):
    file_exists = os.path.isfile(CSV_FILE)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(CSV_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Timestamp", "Person", "Emotion", "Stars"])
            writer.writerow([now, person_id, emotion, stars])
    except PermissionError:
        print("⚠️ Please close 'ratings_log.csv' before writing again.")


def calculate_average_rating():
    if not os.path.isfile(CSV_FILE):
        return 0.0
    try:
        with open(CSV_FILE, mode='r') as file:
            reader = csv.DictReader(file)
            ratings = [int(row['Stars']) for row in reader if row['Stars'].isdigit()]
        return round(sum(ratings) / len(ratings), 2) if ratings else 0.0
    except Exception:
        return 0.0

# Add at the top
session_ratings = []  # Track current session ratings

def update_frame():
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    frame_feedback = ""

    for i, (x, y, w, h) in enumerate(faces):
        face_signature = f"{x}-{y}-{w}-{h}"

        if face_signature in logged_faces:
            continue

        roi = preprocess_face(gray, (x, y, w, h))
        emotion, _ = predict_emotion(model, roi)
        stars = emotion_to_stars(emotion)

        # Draw rectangle and rating
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
        cv2.putText(frame, f"{emotion} | {stars}★", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        draw_star_rating(frame, x, y + h + 10, stars)

        person_id = f"Person_{i + 1}"
        log_to_csv(person_id, emotion, stars)
        session_ratings.append(stars)  # Append to session list
        logged_faces.add(face_signature)

        frame_feedback += f"{person_id} | {emotion} | {stars}★ (Saved)\n"

    # GUI display
    if frame_feedback:
        rating_display.insert(tk.END, frame_feedback)
        rating_display.see(tk.END)

    # --- UPDATED: Use session_ratings instead of reading CSV ---
    if session_ratings:
        avg_rating = round(sum(session_ratings) / len(session_ratings), 2)
        full_stars = int(avg_rating)
        half_star = "½" if avg_rating - full_stars >= 0.5 else ""
        overall_rating_label.config(
            text=f"Overall Rating: {'★' * full_stars}{half_star}{'☆' * (5 - full_stars - (1 if half_star else 0))} ({avg_rating}/5)"
        )

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgtk = ImageTk.PhotoImage(Image.fromarray(img))
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)


def on_closing():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_closing)
update_frame()
root.mainloop()
