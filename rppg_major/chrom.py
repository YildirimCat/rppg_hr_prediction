import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Videodan çerçeve almak için kullanılan sınıf
class FrameCapture:
    def __init__(self, video_path):
        self.video = cv2.VideoCapture(0)

    def get_frame(self):
        ret, frame = self.video.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Nabız hesaplama sınıfı
class HeartRateEstimator:
    def __init__(self, video_path):
        self.frame_capture = FrameCapture(video_path)

    def estimate_heart_rate(self):
        while True:
            frame = self.frame_capture.get_frame()
            if frame is None:
                break

            # Yüz tespiti için Cascade sınıflandırıcısını yükleme
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            # Frame'i gri tonlamalıya dönüştürme
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Yüz tespiti
            faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

            for (x, y, w, h) in faces:
                # Yüz bölgesini kırpma
                face_roi = gray_frame[y:y+h, x:x+w]

                # Yüz bölgesini normalize etme
                face_roi = cv2.equalizeHist(face_roi)

                # Histogram eşitleme
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                face_roi = clahe.apply(face_roi)

                # Yüz bölgesini renkli forma dönüştürme
                face_roi_color = frame[y:y+h, x:x+w]

                # Yüz bölgesindeki kanalları ayrıştırma
                r, g, b = cv2.split(face_roi_color)

                # Renk kanallarının chrominance değerlerini hesaplama
                cr = np.mean(r)
                cg = np.mean(g)
                cb = np.mean(b)

                # Chrominance değerleri ile nabız tahmini
                pulse = (cr + cg + cb) / 3.0

                # Tahmini nabızı ekrana yazdırma
                print("Tahmini Nabız: ", int(pulse))

            # Frame'i gösterme
            plt.imshow(frame)
            plt.show()

        self.frame_capture.video.release()

# Ana program
if __name__ == '__main__':
    video_path = 'video.mp4'  # İşlem yapılacak video dosyasının yolu
    estimator = HeartRateEstimator(video_path)
    estimator.estimate_heart_rate()
