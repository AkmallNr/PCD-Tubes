import cv2
import numpy as np
import onnxruntime as ort
from mtcnn import MTCNN

# Load MTCNN untuk deteksi wajah
detector = MTCNN()

# Load model ONNX untuk prediksi umur dan gender
gender_age_session = ort.InferenceSession('model/genderage.onnx')

def detect_face(img):
    # Deteksi wajah dengan MTCNN
    faces = detector.detect_faces(img)
    if len(faces) == 0:
        return None
    # Ambil wajah pertama yang terdeteksi
    x1, y1, width, height = faces[0]['box']
    return (x1, y1, x1 + width, y1 + height)

def predict_age_and_gender(img, box):
    # Crop wajah dari gambar berdasarkan box yang dideteksi
    x1, y1, x2, y2 = box
    face_crop = img[y1:y2, x1:x2]
    face_crop = cv2.resize(face_crop, (112, 112))  # Resize sesuai input model
    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)  # Ubah ke RGB
    face_crop = np.expand_dims(face_crop, axis=0).astype(np.float32)
    face_crop = (face_crop - 127.5) / 128.0  # Normalisasi sesuai InsightFace

    # Prediksi umur dan gender menggunakan model ONNX
    input_name = gender_age_session.get_inputs()[0].name
    output = gender_age_session.run(None, {input_name: face_crop})[0]

    # Ambil umur dan gender dari output
    predicted_age = output[0].flatten()[0]  # Ambil umur
    predicted_gender = "Male" if output[1].flatten()[0] < 0.5 else "Female"  # Ambil gender

    return predicted_age, predicted_gender
