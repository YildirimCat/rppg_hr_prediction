from face_detection import face_detector


def main():
    # Get ROI's
    rois = []
    rois.extend(face_detector.detect_face())
    print(rois)

if __name__ == '__main__':
    main()