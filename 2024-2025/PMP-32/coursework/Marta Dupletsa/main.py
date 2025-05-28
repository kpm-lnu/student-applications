from imageai.Detection import ObjectDetection, VideoObjectDetection
import os

execution_path = os.getcwd()

# === РОЗПІЗНАВАННЯ НА ФОТО ===
def detect_objects_on_image():
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
    detector.loadModel()

    detections = detector.detectObjectsFromImage(
        input_image=os.path.join(execution_path, "input.jpg"),
        output_image_path=os.path.join(execution_path, "output.jpg"),
        minimum_percentage_probability=50
    )

    print("=== Обʼєкти на фото ===")
    for obj in detections:
        print(f"{obj['name']} ({obj['percentage_probability']:.1f}%) -> {obj['box_points']}")

# === РОЗПІЗНАВАННЯ НА ВІДЕО ===
def detect_objects_on_video():
    video_detector = VideoObjectDetection()
    video_detector.setModelTypeAsYOLOv3()
    video_detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
    video_detector.loadModel()

    def for_frame(frame_number, output_array, output_count, returned_frame):
        print(f"[Кадр {frame_number}] Обʼєкти: {output_count}")

    video_detector.detectObjectsFromVideo(
        input_file_path=os.path.join(execution_path, "input_video.mp4"),
        output_file_path=os.path.join(execution_path, "output_video"),
        frames_per_second=20,
        log_progress=True,
        per_frame_function=for_frame,
        minimum_percentage_probability=50
    )

if __name__ == "__main__":
    detect_objects_on_image()
    detect_objects_on_video()
