import requests
from io import BytesIO
from PIL import Image
import os
import time
from triangulation import compress_image_with_delaunay

def load_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"\nПомилка при завантаженні зображення з URL: {e}")
        return None

def main():
    print("\n--- Стиснення зображення методом тріангуляції Делоне ---")
    
    source = input("Вставте посилання на зображення або шлях до локального файлу: ").strip()
    
    try:
        points = int(input("Скільки точок використовувати? ").strip())
        if points < 4:
            raise ValueError("Має бути щонайменше 4 точки.")
    except ValueError:
        print("Некоректне значення кількості точок.")
        return

    use_edges = input("Використовувати контурне виявлення (Canny)? [y/n]: ").strip().lower() == 'y'

    if source.startswith("http"):
        img = load_image_from_url(source)
        if img is None:
            return
        input_path = "temp_input_image.jpg"
        img.save(input_path)
    else:
        if not os.path.exists(source):
            print("Файл не знайдено.")
            return
        input_path = source

    
    os.makedirs("results", exist_ok=True)

    timestamp = int(time.time())
    output_path = f"results/compressed_{points}_pts_{timestamp}.jpeg"

    print("\nОбробка... Зачекайте.")
    start_time = time.time()

    compress_image_with_delaunay(input_path, output_path, num_points=points, use_edges=use_edges)
    
    end_time = time.time() 
    elapsed = end_time - start_time

    print(f"\nГотово! Результат збережено у файл: {output_path}")
    print(f" Час виконання: {elapsed:.2f} секунд")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрограму перервано користувачем.")
    except Exception as e:
        print(f"\nНевідома помилка: {e}")