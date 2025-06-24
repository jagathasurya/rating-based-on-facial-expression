import cv2
import numpy as np

# Labels for the emotion model (make sure these match your model's training order)
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_face(gray_frame, face_coords):
    x, y, w, h = face_coords
    roi = gray_frame[y:y+h, x:x+w]
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype("float32")
    roi = np.expand_dims(roi, axis=-1)  # Add channel dimension
    return roi

def predict_emotion(model, face_image):
    face_image = face_image.reshape(1, 48, 48, 1) / 255.0  # Normalize input
    prediction = model.predict(face_image)[0]
    max_index = np.argmax(prediction)
    return labels[max_index], prediction[max_index]
