import numpy as np
import cv2
import datetime

image_path = '/Users/skmirajulislam/Documents/MyPython/ML/ObjectDection/img/car-road.jpg'
yolov3_weights = '/Users/skmirajulislam/Documents/Python/ML/ObjectDection/Data/yolov3.weights'
yolov3_cfg = '/Users/skmirajulislam/Documents/Python/ML/ObjectDection/Data/yolov3.cfg'
min_confidence = 0.2  # Adjust this threshold as needed

net = cv2.dnn.readNet(yolov3_weights, yolov3_cfg)
np.random.seed(543210)

classes = []
with open('/Users/skmirajulislam/Documents/Python/ML/ObjectDection/Data/coco.txt', 'r') as f:
    classes = f.read().splitlines()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# image = cv2.imread(image_path)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error occurred in video stream or file")

# List for database with time
Mongo = []
# Adjust this value to control the bounding box reduction (e.g., 0.8 for 80% reduction)
bbox_reduction_factor = 0.8

while cap.isOpened():
    ret, image = cap.read()
    if ret:
        height, width = image.shape[0], image.shape[1]
        blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

        net.setInput(blob)
        output_layer_names = net.getUnconnectedOutLayersNames()
        detected_objects = net.forward(output_layer_names)

        boxes = []  # To store bounding box information

        for detection in detected_objects:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > min_confidence:
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)

                    # Reduce the bounding box dimensions
                    w *= bbox_reduction_factor
                    h *= bbox_reduction_factor

                    upper_left_x = int(center_x - w / 2)
                    upper_left_y = int(center_y - h / 2)
                    lower_right_x = int(center_x + w / 2)
                    lower_right_y = int(center_y + h / 2)

                    boxes.append([upper_left_x, upper_left_y, lower_right_x, lower_right_y, confidence, class_id])

        # Apply Non-Maximum Suppression (NMS)
        if len(boxes) > 0:
            boxes = np.array(boxes)
            confidences = boxes[:, 4].astype(float)
            class_ids = boxes[:, 5].astype(int)

            indices = cv2.dnn.NMSBoxes(boxes[:, :4], confidences, min_confidence, 0.4)

            for i in indices:
                box = boxes[i]  # Use i directly
                x1, y1, x2, y2, confidence, class_id = box
                class_id = int(class_id)  # Ensure class_id is an integer

                # Ensure the coordinates are integers and within image boundaries
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width - 1, x2)
                y2 = min(height - 1, y2)

                prediction_text = f"{classes[class_id]}: {confidence:.2f}%"
                cv2.rectangle(image, (x1, y1), (x2, y2), colors[class_id], 3)
                cv2.putText(image, prediction_text, (x1, y1 - 15 if y1 > 30 else y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[class_id], 2)

                Mongo.append(classes[class_id])

        cv2.imshow("Detected Objects", image)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("Detected items are:")
            now = datetime.datetime.now()
            Curr_time = now.strftime("%B/%d/%Y %H:%M:%S")
            data_db_set = set(Mongo)
            data_db_list = list(data_db_set)
            data_db_list.insert(0, Curr_time)
            print(data_db_list)
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
