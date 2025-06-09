import cv2 as cv2
import numpy as np

click_coords = None


"""
Running Average Algorithm
This algorithm applies a running average to video frames for motion detection. 
"""


def running_average(video_path):
    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()
    average_background = np.float32(frame)

    while True:
        _, frame = cap.read()

        if frame is None:
            break

        cv2.accumulateWeighted(frame, average_background, 0.02)
        background_model = cv2.convertScaleAbs(average_background)

        resized_background = cv2.resize(background_model,
                                        (background_model.shape[1] // 2, background_model.shape[0] // 2))
        # resized_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

        cv2.imshow('Running Average', resized_background)
        #cv2.imshow('Input', resized_frame)

        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


"""
Background Subtraction MOG2 Algorithm
This algorithm uses the MOG2 (Mixture of Gaussian) method for background subtraction in video streams. 
"""


def background_subtraction_mog2(video_path):
    video_capture = cv2.VideoCapture(video_path)
    background_subtractor = cv2.createBackgroundSubtractorMOG2()

    while True:
        is_frame_read, current_frame = video_capture.read()

        if not is_frame_read:
            break

        foreground_mask = background_subtractor.apply(current_frame)

        colored_mask = cv2.cvtColor(foreground_mask, cv2.COLOR_GRAY2BGR)

        colored_mask[foreground_mask != 0] = [255, 0, 0]

        alpha = 0.5
        overlayed_frame = cv2.addWeighted(current_frame, 1, colored_mask, alpha, 0)

        resized_frame = cv2.resize(overlayed_frame, (overlayed_frame.shape[1] // 2, overlayed_frame.shape[0] // 2))

        cv2.imshow('Background Subtraction MOG2 - Overlay', resized_frame)

        pressed_key = cv2.waitKey(30) & 0xff
        if pressed_key == 27:  # ESC key
            break

    video_capture.release()
    cv2.destroyAllWindows()


"""
Dense Optical Flow Algorithm
This algorithm calculates the motion between two consecutive frames at every pixel, providing a dense optical flow field.
"""


def dense_optical_flow(video_path):
    video_capture = cv2.VideoCapture(video_path)

    successful_frame_read, first_frame = video_capture.read()
    if not successful_frame_read:
        video_capture.release()
        return

    previous_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    while True:
        successful_frame_read, next_frame = video_capture.read()
        if not successful_frame_read:
            break

        next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        flow_vectors = cv2.calcOpticalFlowFarneback(previous_frame_gray, next_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_magnitude, flow_angle = cv2.cartToPolar(flow_vectors[..., 0], flow_vectors[..., 1])

        normalized_flow_magnitude = cv2.normalize(flow_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


        hsv_color = np.zeros_like(first_frame)
        hsv_color[..., 0] = flow_angle * 180 / np.pi / 2
        hsv_color[..., 1] = 255
        hsv_color[..., 2] = normalized_flow_magnitude

        flow_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)


        alpha = 0.6
        overlayed_frame = cv2.addWeighted(next_frame, 1, flow_color, alpha, 0)

        resized_overlayed_frame = cv2.resize(overlayed_frame, (overlayed_frame.shape[1] // 2, overlayed_frame.shape[0] // 2))

        cv2.imshow('Dense Optical Flow on Original Frame', resized_overlayed_frame)
        previous_frame_gray = next_frame_gray

        pressed_key = cv2.waitKey(30) & 0xff
        if pressed_key == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()


"""
Sparse Optical Flow Algorithm
This algorithm tracks specific feature points between consecutive frames, providing a sparse but informative representation of motion. 
"""


def sparse_optical_flow(video_path, reset_interval=30, min_points=10):
    video_capture = cv2.VideoCapture(video_path)

    corner_detection_params = dict(maxCorners=100,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)

    lucas_kanade_params = dict(winSize=(15, 15),
                               maxLevel=2,
                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                         10, 0.03))
    random_colors = np.random.randint(0, 255, (100, 3))

    successful_frame_read, previous_frame = video_capture.read()
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    initial_points = cv2.goodFeaturesToTrack(previous_frame_gray, mask=None, **corner_detection_params)

    drawing_mask = np.zeros_like(previous_frame)

    frame_counter = 0

    while True:
        successful_frame_read, current_frame = video_capture.read()
        if not successful_frame_read:
            break

        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        new_points, st, err = cv2.calcOpticalFlowPyrLK(previous_frame_gray, current_frame_gray, initial_points, None, **lucas_kanade_params)

        good_new = new_points[st == 1] if new_points is not None else []
        good_old = initial_points[st == 1] if initial_points is not None else []

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            a, b, c, d = int(a), int(b), int(c), int(d)
            drawing_mask = cv2.line(drawing_mask, (a, b), (c, d), random_colors[i].tolist(), 2)
            current_frame = cv2.circle(current_frame, (a, b), 5, random_colors[i].tolist(), -1)

        combined_image = cv2.add(current_frame, drawing_mask)
        resized_combined_image = cv2.resize(combined_image, (combined_image.shape[1] // 2, combined_image.shape[0] // 2))
        cv2.imshow('Sparse Optical Flow', resized_combined_image)

        key_press = cv2.waitKey(30)
        if key_press == 27:
            break

        frame_counter += 1

        if len(good_new) < min_points or frame_counter % reset_interval == 0:
            initial_points = cv2.goodFeaturesToTrack(current_frame_gray, mask=None, **corner_detection_params)
            drawing_mask = np.zeros_like(previous_frame)
        else:
            initial_points = good_new.reshape(-1, 1, 2)

        previous_frame_gray = current_frame_gray.copy()

    video_capture.release()
    cv2.destroyAllWindows()


"""
Tracking algorithm that uses mouse click and edge detection to initialize and update the tracker on a video.
"""


def tracker_by_mouse_click(video_path):
    global click_coords

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_coords = (x, y)

    def initialize_tracker_with_edges(frame, click_coords):
        if click_coords is None:
            return None

        roi_size = 100
        half_roi = roi_size // 2
        x1, y1 = max(0, click_coords[0] - half_roi), max(0, click_coords[1] - half_roi)
        x2, y2 = min(frame.shape[1], click_coords[0] + half_roi), min(frame.shape[0], click_coords[1] + half_roi)

        roi = frame[y1:y2, x1:x2]
        edges_roi = cv2.Canny(roi, 100, 200)
        contours, _ = cv2.findContours(edges_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            padding = 20
            x = max(x - padding, 0)
            y = max(y - padding, 0)
            w = min(w + 2 * padding, roi.shape[1] - x)
            h = min(h + 2 * padding, roi.shape[0] - y)
            init_bb = (x1 + x, y1 + y, w, h)
        else:
            init_bb = (x1, y1, x2 - x1, y2 - y1)

        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, init_bb)
        return tracker

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', click_event)
    tracker = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if click_coords is not None:
            if tracker is None:
                tracker = initialize_tracker_with_edges(frame, click_coords)
            else:
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
                else:
                    tracker = None
                    click_coords = None

        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == 27:  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


# Without re-detecting the new points
# def sparse_optical_flow(video_path):
#     video_capture = cv2.VideoCapture(video_path)
#
#     # Parameters for corner detection
#     corner_detection_params = dict(maxCorners=100,
#                                    qualityLevel=0.3,
#                                    minDistance=7,
#                                    blockSize=7)
#
#     # Parameters for Lucas-Kanade optical flow
#     lucas_kanade_params = dict(winSize=(15, 15),
#                                maxLevel=2,
#                                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
#                                          10, 0.03))
#
#     # Create some random colors for visualizing tracks
#     random_colors = np.random.randint(0, 255, (100, 3))
#
#     # Take first frame and find corners in it
#     successful_frame_read, previous_frame = video_capture.read()
#     previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
#     initial_points = cv2.goodFeaturesToTrack(previous_frame_gray, mask=None, **corner_detection_params)
#
#     # Create a mask image for drawing purposes
#     drawing_mask = np.zeros_like(previous_frame)
#
#     while True:
#         successful_frame_read, current_frame = video_capture.read()
#         if not successful_frame_read:
#             break
#
#         current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
#
#         # Calculate optical flow
#         new_points, st, err = cv2.calcOpticalFlowPyrLK(previous_frame_gray, current_frame_gray, initial_points, None, **lucas_kanade_params)
#
#         # Select good points
#         good_new_points = new_points[st == 1]
#         good_old_points = initial_points[st == 1]
#
#         # Draw the tracks
#         for i, (new, old) in enumerate(zip(good_new_points, good_old_points)):
#             a, b = new.ravel()
#             c, d = old.ravel()
#
#             # Convert points to integers
#             a, b, c, d = int(a), int(b), int(c), int(d)
#
#             drawing_mask = cv2.line(drawing_mask, (a, b), (c, d), random_colors[i].tolist(), 2)
#             current_frame = cv2.circle(current_frame, (a, b), 5, random_colors[i].tolist(), -1)
#
#         # Combine the current frame with the drawing mask
#         combined_image = cv2.add(current_frame, drawing_mask)
#
#         # Resize for display
#         resized_combined_image = cv2.resize(combined_image, (combined_image.shape[1] // 2, combined_image.shape[0] // 2))
#         cv2.imshow('Sparse Optical Flow', resized_combined_image)
#
#         # Break loop on ESC key press
#         key_press = cv2.waitKey(30)
#         if key_press == 27:
#             break
#
#         # Update the previous frame and points
#         previous_frame_gray = current_frame_gray.copy()
#         initial_points = good_new_points.reshape(-1, 1, 2)
#
#     video_capture.release()
#     cv2.destroyAllWindows()
