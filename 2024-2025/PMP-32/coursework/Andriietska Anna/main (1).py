from diffusers import StableDiffusionPipeline
import torch
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from PIL import ImageTk, Image

# Завантаження моделі Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Змінні для збереження зображення та оцінки якості
generated_image = None
image_quality = None

def generate_image(prompt):
    global generated_image
    image = pipe(prompt).images[0]
    image.show()  # Показати зображення
    image.save("generated_image.png")  # Зберегти файл
    generated_image = image  # Зберігаємо зображення для подальшої оцінки
    return image

def on_generate_button_click():
    user_prompt = entry.get()  # Отримати введений опис
    if not user_prompt:
        messagebox.showwarning("Вхід", "Будь ласка, введіть опис зображення.")
        return
    
    # Генерація зображення
    image = generate_image(user_prompt)

    # Відображення зображення в GUI
    image.thumbnail((400, 400))  # Зменшити розмір для відображення
    img = ImageTk.PhotoImage(image)
    
    # Оновлення елемента зображення на GUI
    label_img.config(image=img)
    label_img.image = img  # Зберігаємо посилання на зображення

    # Очищення попередніх оцінок
    label_quality.config(text="Оцінка не проведена")

def on_save_button_click():
    # Вибір шляху для збереження зображення
    file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
    if file_path:
        image = Image.open("generated_image.png")
        image.save(file_path)
        messagebox.showinfo("Збережено", "Зображення збережено успішно!")

def on_good_quality_button_click():
    global image_quality
    image_quality = "Висока"
    label_quality.config(text=f"Оцінка якості: {image_quality}")

def on_poor_quality_button_click():
    global image_quality
    image_quality = "Низька"
    label_quality.config(text=f"Оцінка якості: {image_quality}")

# Створення основного вікна
root = tk.Tk()
root.title("Генерація зображень з тексту")
root.geometry("500x700")

# Створення елементів інтерфейсу
label = tk.Label(root, text="Введіть опис зображення:")
label.pack(pady=10)

entry = tk.Entry(root, width=50)
entry.pack(pady=10)

generate_button = tk.Button(root, text="Генерувати зображення", command=on_generate_button_click)
generate_button.pack(pady=10)

save_button = tk.Button(root, text="Зберегти зображення", command=on_save_button_click)
save_button.pack(pady=10)

label_img = tk.Label(root)
label_img.pack(pady=10)

label_quality = tk.Label(root, text="Оцінка не проведена", font=("Arial", 10))
label_quality.pack(pady=10)

# Кнопки для оцінки якості
good_quality_button = tk.Button(root, text="Оцінити високо", command=on_good_quality_button_click)
good_quality_button.pack(pady=5)

poor_quality_button = tk.Button(root, text="Оцінити низько", command=on_poor_quality_button_click)
poor_quality_button.pack(pady=5)

# Запуск GUI
root.mainloop()
