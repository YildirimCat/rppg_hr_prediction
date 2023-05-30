import numpy as np
import argparse
import cv2


def detect_face():

    # Argparser
    """
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-p', '--prototxt', required=True, help='help to prototxt file')
    arg_parser.add_argument('-a', '--model', required=True, help='path to caffe model file')
    arg_parser.add_argument('-c', '--confidence', type=float, default=0.5, help='Min probability for filtering flow detections')

    args = vars(arg_parser.parse_args())

    """

    #face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #face_detector = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])

    modelFile = "C:/Users/Yldrm/Desktop/HR_Monitoring/rppg_test/rppg-pos/rppg_hr_prediction/rppg_major/face_detection/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "C:/Users/Yldrm/Desktop/HR_Monitoring/rppg_test/rppg-pos/rppg_hr_prediction/rppg_major/face_detection/deploy.prototxt.txt"

    face_detector = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    print('Starting video...')

    # Kamera aç
    cap = cv2.VideoCapture(0)
    rois = []
    while True:
        # Kameradan bir kare al
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
            #cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            text = '{:2f}%'.format(confidence * 100)

            if startY - 10 > 10:
                y = startY - 10
            else:
                y = startY + 10

            # ROI'yi belirleme
            roi_left_cheek = (int(startX + (endX - startX) / 10), int(startY + 2 * (endY - startY) / 4),
                             int(startX + 3*(endX - startX) / 10), int(startY + 3 * (endY - startY) / 4))
            roi_right_cheek = (int(startX + 7*(endX - startX) / 10), int(startY + 2 * (endY - startY) / 4),
                              int(startX + 9 * (endX - startX) / 10), int(startY + 3 * (endY - startY) / 4))
            roi_forehead = (int(startX + (endX - startX) / 6), int(startY + (endY - startY) / 10),
                           int(startX + 3*(endX - startX) / 4), int(startY + (endY - startY) / 4))

            rois.append(roi_forehead)
            rois.append(roi_right_cheek)
            rois.append(roi_left_cheek)

            # ROI'leri dikdörtgen içinde gösterme
            cv2.rectangle(frame, (int(startX + (endX - startX) / 10), int(startY + 2*(endY - startY) / 4)),
                          (int(startX + 3 * (endX - startX) / 10), int(startY + 3 * (endY - startY) / 4)), (0, 255, 0), 1)  # Sol elmacık kemiği
            cv2.rectangle(frame, (int(startX + 7 * (endX - startX) / 10), int(startY + 2*(endY - startY) / 4)),
                          (int(startX + 9 * (endX - startX) / 10), int(startY + 3 * (endY - startY) / 4)), (0, 255, 0), 1)  # Sağ elmacık kemiği
            cv2.rectangle(frame, (int(startX + (endX - startX) / 6), int(startY + (endY - startY) / 10)),
                          (int(startX + 3 * (endX - startX) / 4), int(startY + (endY - startY) / 4)), (0, 0, 255), 1)  # Alın bölgesi

            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)


        # Kullanıcıya görüntüyü göster
        cv2.imshow("Output", frame)

        # q tuşuna basarak çıkış yap
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return rois

detect_face()