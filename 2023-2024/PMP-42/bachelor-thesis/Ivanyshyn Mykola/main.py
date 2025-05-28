import tkinter as tk
import os
import shutil
import face_recognition
import numpy as np
import json
import cv2


from tkinter import filedialog, messagebox
from mtcnn import MTCNN

def getDataBase():
    for widget in root.winfo_children():
        widget.destroy()

def createNewFace():
    for widget in root.winfo_children():
        widget.destroy()
    root.columnconfigure(0, weight=1000)
    for i in range(5):
        root.rowconfigure(i, weight=1)

    label = tk.Label(root, text="Введіть назву особи:")
    label.grid(row = 0, column = 0, padx=10, pady=10, sticky="ew")

    name = tk.Entry(root)
    name.grid(row = 1, column = 0, padx=10, pady=10, sticky="ew")

    createButoon = tk.Button(root, text="Обрати фото", command=lambda: create(name))
    createButoon.grid(row = 2, column = 0, padx=10, pady=10, sticky="ew")

    returnButton = tk.Button(root, text="Назад", command=back)
    returnButton.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

    trainButton = tk.Button(root, text="Зберегти", command=train)
    trainButton.grid(row=4, column=0, padx=10, pady=10, sticky="ew")

def create(name):
    path = "Dataset/Face"

    folderName = name.get()

    if not folderName:
        messagebox.showerror("Помилка", "Будь ласка, введіть назву особи")
        return

    filePaths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg")])

    if not filePaths:
        return

    if not os.path.exists(path):
        os.makedirs(path)

    photoFolderPath = os.path.join(path, folderName)
    if not os.path.exists(photoFolderPath):
        os.makedirs(photoFolderPath)

    for filePath in filePaths:
        fileName = os.path.basename(filePath)
        destination_path = os.path.join(photoFolderPath, fileName)
        shutil.copy(filePath, destination_path)

    messagebox.showinfo("Виконано", "Фото успішно збережено")

def back():
    for widget in root.winfo_children():
        widget.destroy()
    mainpage()

def train():
    path = "Dataset/Face"
    pathSave = "Dataset"

    known_face_encodings = []
    known_face_names = []

    for root_, dirs, files in os.walk(path):
        for dir_name in dirs:
            print(f"Папка: {dir_name}")
            dir_path = os.path.join(root_, dir_name)
            temp = np.zeros(128)
            n = len(os.listdir(dir_path))
            print(n)
            for file_name in os.listdir(dir_path):
                full_path = os.path.join(path, dir_name)
                full_path = os.path.join(full_path, file_name)
                print(full_path)
                image = face_recognition.load_image_file(full_path)
                box = face_recognition.face_locations(image)
                encoding = face_recognition.face_encodings(image, box, model="large")[0]
                temp += encoding
            known_face_encodings.append(temp / n)
            known_face_names.append(dir_name)

    faces_data = {}
    for name, encoding in zip(known_face_names, known_face_encodings):
        faces_data[name] = encoding.tolist()

    with open('Dataset/faces_data1.json', 'w') as f:
        json.dump(faces_data, f)

    messagebox.showinfo("Виконано", "Дані зберезено у базі даних")

    for widget in root.winfo_children():
        widget.destroy()
    mainpage()

def loadFace():
    faceNames = []
    faceEncodings = []
    path = "Dataset/faces_data1.json"

    with open(path, 'r') as f:
        facesData = json.load(f)

    faceNames = list(facesData.keys())
    faceEncodings = [np.array(encoding) for encoding in facesData.values()]
    return  faceNames, faceEncodings
def launchCamera(model_):
    model = model_.get()
    if not model:
        messagebox.showerror("Помилка","Модель не обрано")
        return

    faceNames, faceEncodings = loadFace()

    cameraCapture = cv2.VideoCapture(0)
    if not cameraCapture.isOpened():
        messagebox.showerror("Помилка", "Камеру не знайдено")
        return

    while True:
        ret, frame = cameraCapture.read()
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = faceRecognition(rgbFrame,model, faceNames, faceEncodings)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cameraCapture.release()
            cv2.destroyAllWindows()
            break

def faceRecognition(frame, model, faceNames, faceEncodings):
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    faceLocations = []
    if(model == "HOG"):
        faceLocations = face_recognition.face_locations(frame)
    elif(model == "MTCNN"):
        faces = detector.detect_faces(frame)
        faceLocations = []
        for result in faces:
            x, y, width, height = result['box']
            faceLocations.append((y, x + width, y + height, x))

    faceEncodings_ = face_recognition.face_encodings(frame, faceLocations, model="large")

    faceNames_ = []
    for faceEncoding in faceEncodings_:
        matches = face_recognition.compare_faces(faceEncodings, faceEncoding)
        name = "Unknown"

        faceDistances = face_recognition.face_distance(faceEncodings, faceEncoding)
        bestIndex = np.argmin(faceDistances)

        if matches[bestIndex]:
            name = faceNames[bestIndex]

        faceNames_.append(name)

    for (top, right, bottom, left), name in zip(faceLocations, faceNames_):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        print(f"{name} {round(faceDistances[bestIndex], 3)}")
    return frame

def launchPhoto(model_):
    model = model_.get()
    if not model:
        messagebox.showerror("Помилка", "Модель не обрано")
        return

    faceNames, faceEncodings = loadFace()

    filePaths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg")])
    for filePath in filePaths:
        image = cv2.imread(filePath)
        rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame = faceRecognition(rgbImage, model, faceNames, faceEncodings)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def launchVideo(model_):
    model = model_.get()
    if not model:
        messagebox.showerror("Помилка", "Модель не обрано")
        return

    faceNames, faceEncodings = loadFace()

    filePaths = filedialog.askopenfilenames(filetypes=[("Image files", "*.mp4")])

    for filePath in filePaths:
        video = cv2.VideoCapture(filePath)
        if not video.isOpened():
            messagebox.showerror("Помилка", f"Не вдалося відкрити відеофайл: {filePath}")
            continue

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = faceRecognition(rgbFrame, model, faceNames, faceEncodings)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()


def mainpage():
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=10)
    for i in range(10):
        root.rowconfigure(i, weight=1)

    openFaceData = tk.Button(root, text="Відкрити базу даних", command=getDataBase)
    openFaceData.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    createFace = tk.Button(root, text="Додати лице", command=createNewFace)
    createFace.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

    model = tk.StringVar(value="")
    radioButton1 = tk.Radiobutton(root, text="Метод гістограми орієнтованих градієнтів", variable=model, value="HOG")
    radioButton2 = tk.Radiobutton(root, text="MTCNN нейромережа", variable=model, value="MTCNN")
    radioButton1.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
    radioButton2.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

    camera = tk.Button(root, text="Запустити розпізнавання з камери",command=lambda: launchCamera(model))
    camera.grid(row=4, column=0, padx=10, pady=10, sticky="nsew")

    photo = tk.Button(root, text="Запустити розпізнавання з фото",command=lambda: launchPhoto(model))
    photo.grid(row=5, column=0, padx=10, pady=10, sticky="nsew")

    video = tk.Button(root, text="Запустити розпізнавання з відео", command=lambda: launchVideo(model))
    video.grid(row=6, column=0, padx=10, pady=10, sticky="nsew")

detector = MTCNN()
root = tk.Tk()
root.title("Face Recognition")
root.geometry("1280x720")

mainpage()
root.mainloop()