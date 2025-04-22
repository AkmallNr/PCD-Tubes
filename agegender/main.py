import cv2
from utils import detect_face, predict_age_and_gender

# Load gambar yang ingin dianalisis
img = cv2.imread('coba1.jpg')

# Deteksi wajah dalam gambar
box = detect_face(img)
if box is not None:
    # Prediksi umur dan gender
    age, gender = predict_age_and_gender(img, box)
    print(f"Predicted Age: {age:.2f}")
    print(f"Predicted Gender: {gender}")

    # Gambar bounding box wajah, umur dan gender
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"Age: {age:.1f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(img, f"Gender: {gender}", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # Tampilkan gambar dengan bounding box, umur, dan gender
    cv2.imshow('Age and Gender Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Tidak ada wajah terdeteksi.")
