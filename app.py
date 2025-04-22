import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import io
import random
import uuid  # untuk generate key unik
import gdown

model_path = 'model/siamese_cosine_model_best3.h5'

if not os.path.exists(model_path):
    os.makedirs('model', exist_ok=True)
    url = 'https://drive.google.com/uc?id=1ytBWLPBDzOXHxm6jltFZHowVqjO3Vfqr'

    with st.spinner('📥 Model belum ditemukan. Sedang mendownload model...'):
        gdown.download(url, model_path, quiet=False)

    st.success('✅ Model berhasil didownload!')
    st.rerun()  # Reload ulang agar model bisa langsung dipakai

from predict_similiarity import CosineSimilarity, detect_and_crop_face, predict_similarity

# Bikin folder uploads kalau belum ada
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Inisialisasi session state untuk reset
if "reset" not in st.session_state:
    st.session_state.reset = False

# Menyimpan gambar di session state jika sudah di-upload
if "uploaded_file1" not in st.session_state:
    st.session_state.uploaded_file1 = None
if "uploaded_file2" not in st.session_state:
    st.session_state.uploaded_file2 = None

def similarity_page():
    uploaded = False

    # --- UI ---
    input_mode = st.radio("Pilih metode input:", ("Upload Gambar", "Ambil dari Webcam"), key="input_mode_radio")

    if input_mode == "Upload Gambar":
        # Jika sudah ada gambar sebelumnya, tampilkan
        uploaded_file1 = st.file_uploader(
            "Upload file gambar 1", type=['jpg', 'jpeg', 'png'], key="upload1" + str(st.session_state.get("upload_key1", ""))
        )
        uploaded_file2 = st.file_uploader(
            "Upload file gambar 2", type=['jpg', 'jpeg', 'png'], key="upload2" + str(st.session_state.get("upload_key2", ""))
        )

        if uploaded_file1 and uploaded_file2:
            col1, col2 = st.columns(2)

            with col1:
                st.image(uploaded_file1, caption="Gambar 1", use_container_width=True)
                st.session_state.uploaded_file1 = uploaded_file1

            with col2:
                st.image(uploaded_file2, caption="Gambar 2", use_container_width=True)
                st.session_state.uploaded_file2 = uploaded_file2

            uploaded = True
        upload_1 = True


    elif input_mode == "Ambil dari Webcam":
        capture_1 = True
        capture_2 = True
        st.subheader("Ambil gambar 1 dari webcam")
        captured_image1 = st.camera_input("Ambil gambar dari webcam", key="webcam1_" + str(st.session_state.get("webcam_key1", "")))

        st.subheader("Ambil gambar 2 dari webcam")
        captured_image2 = st.camera_input("Ambil gambar dari webcam", key="webcam2_" + str(st.session_state.get("webcam_key2", "")))
 
        # captured_image = st.camera_input("Ambil gambar dari webcam", key="webcam")

        if captured_image1 and captured_image2 is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Gambar 1")
                bytes_data1 = captured_image1.getvalue()
                image = Image.open(io.BytesIO(bytes_data1))
                st.image(image, caption="Gambar dari Webcam", use_container_width=True)
                st.session_state.webcam_1 = captured_image1
            with col2:
                st.subheader("Gambar 2")
                bytes_data2 = captured_image2.getvalue()
                image = Image.open(io.BytesIO(bytes_data2))
                st.image(image, caption="Gambar dari Webcam", use_container_width=True)
                st.session_state.webcam_2 = captured_image2
        
                uploaded = True

    if uploaded:
        if upload_1 == True:
            result, score = predict_similarity(st.session_state.uploaded_file1, st.session_state.uploaded_file2)
        else:
            result, score = predict_similarity(st.session_state.webcam_1, st.session_state.webcam_2)

        st.subheader("🔍 Hasil Face Similarity")
        st.metric(label="Similarity Score : ", value=f"{score:.4f}")
        st.metric(label="Prediction : ", value=result)

        # Tombol reset: ubah state
        if st.button("Ambil Gambar atau Upload Lagi"):
            st.session_state.uploaded_file1 = None
            st.session_state.uploaded_file2 = None
            st.session_state.input_mode = None
            st.session_state.reset = True
            # generate key baru untuk reset file_uploader
            st.session_state.upload_key1 = uuid.uuid4()
            st.session_state.upload_key2 = uuid.uuid4()
            st.session_state.webcam_key1 = uuid.uuid4()
            st.session_state.webcam_key2 = uuid.uuid4()
            st.rerun()


# Judul utama aplikasi
st.title("PCD 2024 - Face Upload & Capture")
st.subheader("Upload atau ambil gambar dari webcam untuk deteksi wajah dan similarity.")

# Reset page bila tombol ditekan
if st.session_state.reset:
    st.session_state.reset = False  # clear state
    st.experimental_set_query_params()  # bersihkan param URL
    st.rerun()

# Tampilkan halaman utama
similarity_page()
