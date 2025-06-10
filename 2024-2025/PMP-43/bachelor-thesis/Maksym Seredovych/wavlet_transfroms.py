import os
import cv2
import pywt
import numpy as np
from PIL import Image
from typing import Union

def get_wavelet_coefficients(
    img_input: Union[str, np.ndarray, Image.Image],
    wavelet: str = "db4",
    level: int = 3,
):
    """
    Обчислює дискретне вейвлет-перетворення (DWT) для зображення,
    заданого шляхом до файлу, numpy-масивом або об’єктом PIL.Image.

    Параметри
    ---------
    img_input : str | np.ndarray | PIL.Image.Image
        Шлях до зображення чи саме зображення.
    wavelet : str
        Назва базового вейвлету (напр., 'db1', 'db4', 'haar').
    level : int
        Рівень розкладання.

    Повертає
    --------
    coeffs : list[np.ndarray] | tuple
        Список (або кортеж) коефіцієнтів вейвлет-розкладу.
    """
    if isinstance(img_input, str):
        if not os.path.exists(img_input):
            raise FileNotFoundError(f"Файл не знайдено: {img_input}")

        img = cv2.imread(img_input, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Не вдалося завантажити зображення: {img_input}")

    elif isinstance(img_input, np.ndarray):
        img = img_input
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    elif isinstance(img_input, Image.Image):
        img = np.asarray(img_input.convert("L"))

    else:
        raise TypeError(
            "Аргумент img_input має бути str (шлях), numpy.ndarray або PIL.Image.Image"
        )
    img = img.astype(np.float32) / 255.0

    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
    return coeffs

if __name__ == "__main__":
    image_path = "data/image.png"
    wavelet = "db4"
    level = 3
    
    coeffs = get_wavelet_coefficients(image_path, wavelet=wavelet, level=level)

    print("Отримані ознаки:", coeffs)