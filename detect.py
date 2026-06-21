import cv2
import torch

# Load YOLOv5 pretrained model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Webcam feed
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("D:/Real-Time-Object-Detection-YOLOv5/sample_videos/test_video.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Draw bounding boxes
    frame = results.render()[0]

    cv2.imshow("Real-Time Object Detection", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()