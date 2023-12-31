import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(0.6)

img = cv2.imread("image.jpg")
def detector(frame):
    count = 0
    height, width, channels = frame.shape
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_detection.process(imgRGB)
    for detection in result.detections:
        boxC = detection.location_data.relative_bounding_box
        ih, iw, _ = imgRGB.shape
        x, y, w, h = int(boxC.xmin * iw), int(boxC.ymin * ih), int(boxC.width * iw), int(boxC.height * ih)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        count += 1
    return count, frame
count, output = detector(img)
cv2.putText(output, "Number of Faces: " + str(count), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow("output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()