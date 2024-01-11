import cv2
import os

from yolov8 import YOLOv8

video = cv2.VideoCapture("rtsp://reolink:reolink123@192.168.1.227:554/h264Preview_01_sub")
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)
#out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 12, size)

# Initialize YOLOv7 model
model_path = "models/yolov8m.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

#cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while video.isOpened():

    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

    try:
        # Read frame from the video
        ret, frame = video.read()
        if not ret:
            break
    except Exception as e:
        print(e)
        continue

    # Update object localizer
    boxes, scores, class_ids = yolov8_detector(frame)

    combined_img = yolov8_detector.draw_detections(frame)
    #cv2.imshow("Detected Objects", combined_img)
    #out.write(combined_img)

#out.release()
