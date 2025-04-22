import streamlit as st
import os
import uuid
import gdown
import io
from PIL import Image

# Download model kalau belum ada (seperti sebelumnya)
model_path = 'model/siamese_cosine_model_best3.h5'
if not os.path.exists(model_path):
    os.makedirs('model', exist_ok=True)
    url = 'https://drive.google.com/uc?id=1ytBWLPBDzOXHxm6jltFZHowVqjO3Vfqr'

    with st.spinner('📥 Model belum ditemukan. Sedang mendownload model...'):
        gdown.download(url, model_path, quiet=False)

    st.success('✅ Model berhasil didownload!')
    st.rerun()

from predict_similiarity import predict_similarity
# Nanti kamu tambahkan import predict_ethnicity dan predict_age_emotion

# Folder uploads
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Session state
if "reset" not in st.session_state:
    st.session_state.reset = False

if "uploaded_file1" not in st.session_state:
    st.session_state.uploaded_file1 = None
if "uploaded_file2" not in st.session_state:
    st.session_state.uploaded_file2 = None

def upload_or_camera_input(mode="single"):
    """
    mode: 'single' untuk upload/capture 1 gambar
          'double' untuk upload/capture 2 gambar (seperti face similarity)
    """
    uploaded = False
    images = []

    input_mode = st.radio("Pilih metode input:", ("Upload Gambar", "Ambil dari Webcam"))

    if input_mode == "Upload Gambar":
        if mode == "single":
            uploaded_file = st.file_uploader("Upload file gambar", type=['jpg', 'jpeg', 'png'])
            if uploaded_file:
                st.image(uploaded_file, caption="Gambar yang diupload", use_container_width=True)
                images.append(uploaded_file)
                uploaded = True
        elif mode == "double":
            uploaded_file1 = st.file_uploader("Upload file gambar 1", type=['jpg', 'jpeg', 'png'])
            uploaded_file2 = st.file_uploader("Upload file gambar 2", type=['jpg', 'jpeg', 'png'])

            if uploaded_file1 and uploaded_file2:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(uploaded_file1, caption="Gambar 1", use_container_width=True)
                with col2:
                    st.image(uploaded_file2, caption="Gambar 2", use_container_width=True)
                images.append(uploaded_file1)
                images.append(uploaded_file2)
                uploaded = True

    else:  # Ambil dari Webcam
        if mode == "single":
            captured_image = st.camera_input("Ambil gambar dari webcam")
            if captured_image:
                st.image(captured_image, caption="Gambar dari Webcam", use_container_width=True)
                images.append(captured_image)
                uploaded = True
        elif mode == "double":
            st.subheader("Ambil gambar 1 dari webcam")
            captured_image1 = st.camera_input("Ambil gambar dari webcam (Gambar 1)")

            st.subheader("Ambil gambar 2 dari webcam")
            captured_image2 = st.camera_input("Ambil gambar dari webcam (Gambar 2)")

            if captured_image1 and captured_image2:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(captured_image1, caption="Gambar 1", use_container_width=True)
                with col2:
                    st.image(captured_image2, caption="Gambar 2", use_container_width=True)
                images.append(captured_image1)
                images.append(captured_image2)
                uploaded = True

    return uploaded, images

def similarity_page():
    st.header("🔎 Face Similarity Detection")
    uploaded, images = upload_or_camera_input(mode="double")

    if uploaded:
        result, score = predict_similarity(images[0], images[1])
        st.subheader("Hasil Face Similarity")
        st.metric(label="Similarity Score", value=f"{score:.4f}")
        st.metric(label="Prediction", value=result)

def ethnicity_page():
    st.header("🌎 Ethnicity Detection")
    algorithm_choose = st.selectbox("Pilih algoritma yang ingin digunakan", ["CNN(Convolution Neural Network)","Ensemble Method","Custom Feature Engineering"])
    uploaded, images = upload_or_camera_input(mode="single")

    if uploaded:
        # result = predict_ethnicity(images[0])  # Misal fungsinya predict_ethnicity
        result = "Contoh: Asian"  # Dummy dulu
        st.subheader("Hasil Deteksi Etnis")
        st.success(f"Etnis wajah terdeteksi: **{result}**")

def age_emotion_page():
    st.header("🎭 Age & Emotion Detection")
    uploaded, images = upload_or_camera_input(mode="single")

    if uploaded:
        # age, emotion = predict_age_emotion(images[0])
        age, emotion = 25, "Happy"  # Dummy dulu
        st.subheader("Hasil Deteksi")
        st.metric(label="Prediksi Usia", value=f"{age} tahun")
        st.metric(label="Prediksi Emosi", value=emotion)

# Sidebar Navigation
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Face Similarity", "Ethnicity Detection", "Age & Emotion Detection"])

# Judul besar
st.title("PCD 2024 - Face Recognition App")

# Routing page
if page == "Face Similarity":
    similarity_page()
elif page == "Ethnicity Detection":
    ethnicity_page()
elif page == "Age & Emotion Detection":
    age_emotion_page()
