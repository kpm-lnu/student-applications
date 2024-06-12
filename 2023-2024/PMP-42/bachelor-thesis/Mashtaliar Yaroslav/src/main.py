import cv2
import math

from history import History
from yolo_utils import load_latest_model
from config import *


# load the history
history = History.load_from_json_file(HISTORY_FILE_PATH)
area = history['abdomen-right']
latest_cycle = area.get_latest_cycle()


# load the model from the latest train results
model = load_latest_model()


# set up the camera input
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(3, CAPTURE_WIDTH)
cap.set(4, CAPTURE_HEIGHT)
show_history = True


class_names = ['inserter', 'navel']


def display_inserter(box, img):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    center = cx, cy = int((x1 + x2) * .5), int((y1 + y2) * .5)
    radius = int((x2 - x1 + y2 - y1) * .25)
    color = (55, 55, 255)
    # contour
    cv2.circle(img, center, radius, color, 3)
    # center
    cv2.circle(img, center, 2, color, 3)


def display_navel(box, img):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    center = cx, cy = int((x1 + x2) * .5), int((y1 + y2) * .5)
    color = (255, 0, 255)
    cv2.line(img, center, (cx, 0), color, 3)
    cv2.line(img, (0, cy), (CAPTURE_WIDTH, cy), color, 3)


# main loop
while True:
    success, img = cap.read()
    results = model(img, stream=True, max_det=2)

    # collect results
    objects = [None, None]
    confidences = [0, 0]

    for r in results:
        boxes = r.boxes

        for box in boxes:
            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            if confidence > confidences[cls]:
                objects[cls] = box
                confidences[cls] = confidence

    inserter, navel = objects
    if navel:
        display_navel(navel, img)
    if inserter:
        display_inserter(inserter, img)

    origin = origin_x, origin_y = 0, 0
    if not navel:
        # display warning
        pass
    else:
        x1, y1, x2, y2 = map(int, navel.xyxy[0])
        origin = origin_x, origin_y = (x1+x2)*.5, (y1+y2)*.5

    unit = 0
    inserter_x, inserter_y = None, None
    if not inserter:
        # display warning
        pass
    else:
        x1, y1, x2, y2 = map(int, inserter.xyxy[0])
        w, h = x2 - x1, y2 - y1
        diameter = (w + h) * .5
        unit = diameter
        inserter_x = (x1 + x2)*.5
        inserter_y = (y1 + y2)*.5

    # display history
    if show_history and inserter and navel:
        oldest_timestamp = latest_cycle.oldest_timestamp()
        latest_timestamp = latest_cycle.latest_timestamp()
        timespan = latest_timestamp - oldest_timestamp
        for insertion in latest_cycle.insertions:
            x = int((insertion.offset_x + latest_cycle.x)*unit + origin_x)
            y = int((insertion.offset_y + latest_cycle.y) * unit + origin_y)
            time_delta = insertion.timestamp - oldest_timestamp
            color = (0, 0, int(255 * (1 - time_delta / timespan)))
            print(color)
            cv2.circle(img, (x, y), 2, color, 3, cv2.FILLED)


    cv2.imshow(WINDOW_NAME, img)

    key = cv2.waitKey(1)

    if key == EXIT_KEY:
        break
    elif key == CONFIRM_KEY:
        if navel and inserter:
            ix = (inserter_x - origin_x) / unit
            iy = (inserter_y - origin_y) / unit
            insertion = latest_cycle.append(ix, iy)
    elif key == SHOW_HISTORY_KEY:
        pass
    elif key == SAVE_HISTORY_KEY:
        history.save_to_json_file(HISTORY_FILE_PATH)
    elif key == RELOAD_HISTORY_KEY:
        history = History.load_from_json_file(HISTORY_FILE_PATH)


cap.release()
cv2.destroyAllWindows()