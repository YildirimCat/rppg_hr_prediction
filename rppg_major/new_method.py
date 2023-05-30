# Gerekli kütüphaneleri yükleyin
import cv2
import numpy as np
from scipy import signal

modelFile = "C:/Users/Yldrm/Desktop/HR_Monitoring/rppg_test/rppg-pos/rppg_hr_prediction/rppg_major/face_detection/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "C:/Users/Yldrm/Desktop/HR_Monitoring/rppg_test/rppg-pos/rppg_hr_prediction/rppg_major/face_detection/deploy.prototxt.txt"

face_detector = cv2.dnn.readNetFromCaffe(configFile, modelFile)

print('Starting video...')

# Video dosyasını açın
cap = cv2.VideoCapture(0)
rois = []

# Nabız ölçümü için geçmiş değerleri tutacak değişkenleri tanımla
past_pos = []
past_bpm = []

while True:
    # Video akışından kareleri oku
    ret, frame = cap.read()

    (h, w) = frame.shape[:2]

    # Yüz tespiti için ilgili boyuta yeniden boyutlandır
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), [104, 177, 123])

    # Modeli kullanarak yüz tespiti yap
    face_detector.setInput(blob)
    detections = face_detector.forward()

    # Tespit edilen yüzleri dolaşarak ilgi bölgesini belirle
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Güven seviyesi eşik değerinin altındaysa atla
        if confidence < 0.5:
            continue

        # Yüz bölgesini hesapla ve dikdörtgeni çiz
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        text = '{:2f}%'.format(confidence * 100)

        if startY - 10 > 10:
            y = startY - 10
        else:
            y = startY + 10

        # ROI'yi belirleme
        roi_left_cheek = (int(startX + (endX - startX) / 10), int(startY + 2 * (endY - startY) / 4),
                          int(startX + 3 * (endX - startX) / 10), int(startY + 3 * (endY - startY) / 4))
        roi_right_cheek = (int(startX + 7 * (endX - startX) / 10), int(startY + 2 * (endY - startY) / 4),
                           int(startX + 9 * (endX - startX) / 10), int(startY + 3 * (endY - startY) / 4))
        roi_forehead = (int(startX + (endX - startX) / 6), int(startY + (endY - startY) / 10),
                        int(startX + 3 * (endX - startX) / 4), int(startY + (endY - startY) / 4))

        # Append all regions
        rois.append(roi_forehead)
        rois.append(roi_right_cheek)
        rois.append(roi_left_cheek)

        # ROI'leri dikdörtgen içinde gösterme
        cv2.rectangle(frame, (int(startX + (endX - startX) / 10), int(startY + 2 * (endY - startY) / 4)),
                      (int(startX + 3 * (endX - startX) / 10), int(startY + 3 * (endY - startY) / 4)), (0, 255, 0),
                      1)  # Sol elmacık kemiği
        cv2.rectangle(frame, (int(startX + 7 * (endX - startX) / 10), int(startY + 2 * (endY - startY) / 4)),
                      (int(startX + 9 * (endX - startX) / 10), int(startY + 3 * (endY - startY) / 4)), (0, 255, 0),
                      1)  # Sağ elmacık kemiği
        cv2.rectangle(frame, (int(startX + (endX - startX) / 6), int(startY + (endY - startY) / 10)),
                      (int(startX + 3 * (endX - startX) / 4), int(startY + (endY - startY) / 4)), (0, 0, 255),
                      1)  # Alın bölgesi

        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # ROI bölgelerini seç
    rois_frames = [frame[y:y+h, x:x+w] for (x, y, w, h) in rois]

    # Get FPS rate
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Maskeleme işlemi için renk aralığını belirle
    lower_skin = np.array([0, 40, 80], dtype=np.uint8)
    upper_skin = np.array([255, 180, 135], dtype=np.uint8)

    # POS yöntemi ile nabız ölçümü yap
    bpm_values = []
    for i, roi_frame in enumerate(rois_frames):

        # RGB maskeleme işlemini uygula
        roi_mask = cv2.inRange(roi_frame, lower_skin, upper_skin)

        # Maskeleme sonrası ROI çerçevesindeki sadece cilt piksellerini tut
        roi_frame = cv2.bitwise_and(roi_frame, roi_frame, mask=roi_mask)

        g_channel = roi_frame[:, :]  # Sadece G kanalı seçiliyor

        pos = signal.correlate2d(cv2.cvtColor(g_channel, cv2.COLOR_BGR2GRAY), np.array([[-1, 0, 1]]), mode='same')
        f_pos = np.sum(pos, axis=0)
        f_pos = signal.medfilt(f_pos, 3)

        # Nabzı hesaplayın ve listenin sonuna ekle
        bpm = int(signal.find_peaks(f_pos)[0].size * 60 / fps)
        bpm_values.append(bpm)

    # Tüm nabız değerlerini ortalamak için geçmiş nabız değerleri ile birlikte işle
    if past_pos:
        past_pos.append(bpm_values)
        past_pos.pop(0)
        mean_bpm_values = np.mean(past_pos, axis=0)
    else:
        past_pos.append(bpm_values)
        mean_bpm_values = bpm_values

    # Ortalama nabız değerini hesapla ve ekrana yazdır
    mean_bpm = int(np.mean(mean_bpm_values))
    past_bpm.append(mean_bpm)
    past_bpm = past_bpm[-30:]

    # Display HR
    cv2.putText(frame, 'BPM: ' + str(bpm), (h, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)

    # Display FPS Rate
    cv2.putText(frame, "FPS: " + str(int(fps)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    # Video akışını göster
    cv2.imshow('frame', frame)

    # Çıkış yapmak için q tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tüm pencereleri kapat
cap.release()
cv2.destroyAllWindows()
