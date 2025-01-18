import cv2
import mediapipe as mp
import os

#function to detect faces
def detect_face(img):
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = face_detection.process(rgb_img)

        if out.detections is not None:  #making sure a page is not detected
            for detection in out.detections:
                bbox = detection.location_data.relative_bounding_box    #detecting face
                x, y, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height  #retrieving relative location of face

                #converting to actual location
                x1, y1 = int(x*img.shape[1]), int(y*img.shape[0]) 
                x2, y2 = x1 + int(w*img.shape[1]), y1 + int(h*img.shape[0])

                #drawing a rectangle to mark the face
                img_detected = cv2.rectangle(img.copy(), (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img_detected

# Set the directory path that contains the bird images
directory_path = '.venv/Data'

# reading images from the directory, detecting the faces and then displaying the original and results
for file_name in os.listdir(directory_path):
        image_path = directory_path + '/' + file_name
        img = cv2.imread(image_path)        #reading images
        img_detected = detect_face(img) #detecting faces
        cv2.imshow('Original Image', img)
        cv2.waitKey(0)
        cv2.imshow('Face Detected Image', img_detected)
        cv2.waitKey(0)
