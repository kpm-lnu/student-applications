import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import math
import matplotlib.pyplot as plt
import os

class FractalDimension:
    def __init__(self):
        self.black_boundary = 100
        self.delta_max = 4
        self.q_min = 1
        self.q_max = 10
        self.mf_epsilon = 1e-6
        self.alpha_min = 0
        self.alpha_max = 0
        self.cd_points = []
        self.md_points = []
        self.sr_points = []
        self.mf_points = []
        self.alpha_groups = []
        self.local_density_images_directory = "LocalDensityImages"
        self.image = None
        self.total_saturation_sum = 0

    def calculate_capacitive_dimension(self, image_path):
        self.image = Image.open(image_path).convert('RGB')
        result = self._calculate_capacitive_dimension()
        return result

    def _calculate_capacitive_dimension(self):
        self.cd_points = []
        epsilon = min(self.image.width, self.image.height) // 2
        while epsilon > 1:
            self.cd_points.append(self._get_point(epsilon))
            epsilon //= 2
        return self._get_approximation_by_less_square_method(self.cd_points)

    def _get_point(self, epsilon):
        n_epsilon = 0
        for x in range(0, self.image.width + 1, epsilon):
            if self.image.width - x < epsilon:
                continue
            x1 = x + epsilon
            for y in range(0, self.image.height + 1, epsilon):
                if self.image.height - y < epsilon:
                    continue
                y1 = y + epsilon
                if self._is_fractal_in_cell(x, x1, y, y1):
                    n_epsilon += 1
        return math.log(1 / epsilon), math.log(n_epsilon)

    def _is_fractal_in_cell(self, x_start, x_end, y_start, y_end):
        for x in range(x_start, x_end):
            for y in range(y_start, y_end):
                pixel = self.image.getpixel((x, y))
                if self._is_black_pixel(pixel):
                    return True
        return False

    def _is_black_pixel(self, pixel):
        if pixel[0] <= self.black_boundary and pixel[1] <= self.black_boundary and pixel[2] <= self.black_boundary:
            return True
        return False

    def _get_approximation_by_less_square_method(self, points):
        k, b = self._less_square_get_coefficient(points)
        return k

    def calculate_minkowski_dimension_for_grayscale_images(self, image_path, epsilon=None):
        self.image = Image.open(image_path).convert('RGB')
        if epsilon is not None:
            return self._get_minkowski_dimension_for_grayscale_images(epsilon)
        else:
            self.md_points = []
            epsilon = min(self.image.width, self.image.height)
            while epsilon > 1:
                self.md_points.append((epsilon, self._get_minkowski_dimension_for_grayscale_images(epsilon)))
                epsilon //= 2

    def _get_minkowski_dimension_for_grayscale_images(self, epsilon):
        a_deltas = self._get_a_deltas(epsilon)
        return sum(a_deltas)

    def _get_a_deltas(self, epsilon):
        a_deltas = []
        for x in range(0, self.image.width + 1, epsilon):
            if self.image.width - x < epsilon:
                continue
            x1 = x + epsilon
            for y in range(0, self.image.height + 1, epsilon):
                if self.image.height - y < epsilon:
                    continue
                y1 = y + epsilon
                a_deltas.append(self._compute_a_delta_with_approximation(x, x1 - 1, y, y1 - 1, self.delta_max))
        return a_deltas

    def _compute_a_delta_with_approximation(self, x_start, x_end, y_start, y_end, delta_max):
        rows_count = x_end - x_start + 1
        cols_count = y_end - y_start + 1
        v_deltas = []
        u_delta_minus_1 = self._get_grayscale_matrix(rows_count, cols_count, x_start, y_start)
        b_delta_minus_1 = self._get_grayscale_matrix(rows_count, cols_count, x_start, y_start)
        for delta in range(1, delta_max + 1):
            u_delta = self._create_blanket(rows_count, cols_count, u_delta_minus_1, True)
            b_delta = self._create_blanket(rows_count, cols_count, b_delta_minus_1, False)
            v_deltas.append(self._calculate_v_delta(rows_count, cols_count, u_delta, b_delta))
            u_delta_minus_1 = u_delta
            b_delta_minus_1 = b_delta
        return (v_deltas[delta_max - 1] - v_deltas[delta_max - 2]) / 2

    def _get_grayscale_matrix(self, rows_count, cols_count, x_start, y_start):
        result = np.zeros((rows_count, cols_count), dtype=float)
        for x in range(rows_count):
            for y in range(cols_count):
                pixel = self.image.getpixel((x + x_start, y + y_start))
                result[x, y] = sum(pixel) / 3 / 255.0
        return result

    def _create_blanket(self, rows_count, cols_count, previous_blanket, is_upper):
        result = np.copy(previous_blanket)
        c = 0.01 if is_upper else -0.01
        for x in range(rows_count):
            for y in range(cols_count):
                element = previous_blanket[x, y] + c
                for i in range(-1, 2):
                    if x + i < 0 or x + i >= rows_count:
                        continue
                    for j in range(-1, 2):
                        if y + j < 0 or y + j >= cols_count:
                            continue
                        if i == 0 and j == 0:
                            continue
                        neighbour_element = previous_blanket[x + i, y + j]
                        element = max(element, neighbour_element) if is_upper else min(element, neighbour_element)
                result[x, y] = element
        return result

    def _calculate_v_delta(self, rows_count, cols_count, u_delta, b_delta):
        return np.sum(u_delta - b_delta)

    def calculate_renyi_spectre(self, image_path):
        self.sr_points = []
        for q in range(self.q_min, self.q_max + 1):
            self.image = Image.open(image_path).convert('RGB')
            if q == 0:
                self.black_boundary = 230
                self.sr_points.append((q, self._calculate_capacitive_dimension()))
                self.black_boundary = 100
            elif q == 1:
                self.sr_points.append((q, self._calculate_entropy()))
            else:
                self.sr_points.append((q, self._get_renyi_spectre(q)))

    def _get_renyi_spectre(self, q):
        points = []
        epsilon = min(self.image.width, self.image.height) // 2
        q_minus_1 = 1 if (q - 1) == 0 else (q - 1)
        while epsilon > 1:
            points.append((math.log(epsilon) * q_minus_1, math.log(self._get_sum_of_standardized_measures(q, epsilon))))
            epsilon //= 2
        return self._get_approximation_by_less_square_method(points)

    def _get_sum_of_standardized_measures(self, q, epsilon):
        saturation_sum_in_cells = []
        for x in range(0, self.image.width + 1, epsilon):
            if self.image.width - x < epsilon:
                continue
            x1 = x + epsilon
            for y in range(0, self.image.height + 1, epsilon):
                if self.image.height - y < epsilon:
                    continue
                y1 = y + epsilon
                saturation_sum_in_cells.append(self._get_saturation_sum_in_cell(x, x1, y, y1))
        saturation_sums = sum(saturation_sum_in_cells)
        result = 0
        for sum_val in saturation_sum_in_cells:
            pi = sum_val / saturation_sums
            result += pi ** q
        return result

    def _get_saturation_sum_in_cell(self, x_start, x_end, y_start, y_end):
        result = 0
        for x in range(x_start, x_end):
            for y in range(y_start, y_end):
                pixel = self.image.getpixel((x, y))
                result += sum(pixel) / 3
        return result

    def precalculate_alpha_min_alpha_max(self, image_path):
        self.image = Image.open(image_path).convert('RGB')
        self.total_saturation_sum = self._get_saturation_sum_in_cell(0, self.image.width, 0, self.image.height)
        self._calculate_alpha_min_alpha_max()

    def calculate_multifractal_with_local_density_function(self):
        self._prepare_folder_for_images()
        self._calculate_multifractal_dimension()

    def _prepare_folder_for_images(self):
        if not os.path.exists(self.local_density_images_directory):
            os.makedirs(self.local_density_images_directory)
        for file in os.listdir(self.local_density_images_directory):
            os.remove(os.path.join(self.local_density_images_directory, file))

    def _calculate_multifractal_dimension(self):
        self._calculate_alpha_groups()
        self.mf_points = []
        index = 0
        for file in os.listdir(self.local_density_images_directory):
            file_path = os.path.join(self.local_density_images_directory, file)
            cd = self.calculate_capacitive_dimension(file_path)
            self.mf_points.append((self.alpha_groups[index], cd))
            index += 1

    def _calculate_alpha_groups(self):
        self.alpha_groups = []
        image_index = 1
        for alpha in np.arange(self.alpha_min, self.alpha_max, self.mf_epsilon):
            layer_image = Image.new('RGB', (self.image.width, self.image.height), 'white')
            for x in range(self.image.width):
                for y in range(self.image.height):
                    current_alpha = self._calculate_alpha(x, y)
                    if alpha <= current_alpha < alpha + self.mf_epsilon:
                        layer_image.putpixel((x, y), (0, 0, 0))
            file_path = os.path.join(self.local_density_images_directory, f"{image_index}.bmp")
            layer_image.save(file_path)
            self.alpha_groups.append(((alpha, min(alpha + self.mf_epsilon, self.alpha_max))))
            image_index += 1

    def _calculate_alpha_min_alpha_max(self):
        result = set()
        for x in range(self.image.width):
            for y in range(self.image.height):
                result.add(self._calculate_alpha(x, y))
        self.alpha_min = min(result)
        self.alpha_max = max(result)

    def _calculate_alpha(self, x, y):
        pixel = self.image.getpixel((x, y))
        return (sum(pixel) / 3) / self.total_saturation_sum

    def _calculate_entropy(self):
        points = []
        epsilon = min(self.image.width, self.image.height) // 2
        while epsilon > 1:
            sum_p = self._get_sum_of_standardized_measures_for_entropy(epsilon)
            points.append((math.log(epsilon), sum_p))
            epsilon //= 2
        return self._get_approximation_by_less_square_method(points)

    def _get_sum_of_standardized_measures_for_entropy(self, epsilon):
        saturation_sum_in_cells = []
        for x in range(0, self.image.width + 1, epsilon):
            if self.image.width - x < epsilon:
                continue
            x1 = x + epsilon
            for y in range(0, self.image.height + 1, epsilon):
                if self.image.height - y < epsilon:
                    continue
                y1 = y + epsilon
                saturation_sum_in_cells.append(self._get_saturation_sum_in_cell(x, x1, y, y1))
        saturation_sums = sum(saturation_sum_in_cells)
        
        result = 0
        for sum_val in saturation_sum_in_cells:
            pi = sum_val / saturation_sums
            result += pi * math.log(pi)
        return result

    def _less_square_get_coefficient(self, points):
        x_values = [p[0] for p in points]
        y_values = [p[1] for p in points]
        n = len(points)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in points)
        sum_x_squared = sum(x ** 2 for x in x_values)
        k = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
        b = (sum_y - k * sum_x) / n
        return k, b

class FractalGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Розрахунок фрактальної розмірності")
        self.root.geometry("1100x800")
        self.fractal_dimension = FractalDimension()
        self.image_path = None
        self.image_tk = None

        self.create_widgets()

    def create_widgets(self):
        left_frame = tk.Frame(self.root)
        left_frame.grid(row=0, column=0, sticky="nsew")

        right_frame = tk.Frame(self.root)
        right_frame.grid(row=0, column=1, sticky="nsew")

        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.image_label = tk.Label(right_frame)
        self.image_label.pack(fill="both", expand=True)

        tk.Button(left_frame, text="Вибрати зображення", command=self.load_image).pack(pady=5)

        # Capacitive Dimension Panel
        cap_panel = tk.LabelFrame(left_frame, text="Ємнісна розмірність")
        cap_panel.pack(pady=5, fill="x")

        tk.Label(cap_panel, text="Поріг чорного кольору").pack()
        self.threshold_entry = tk.Entry(cap_panel)
        self.threshold_entry.pack()
        self.threshold_entry.insert(0, "100")

        tk.Button(cap_panel, text="Застосувати поріг", command=self.apply_threshold).pack(pady=5)
        tk.Button(cap_panel, text="Розрахувати", command=self.calculate_capacitive).pack(pady=5)
        self.cap_result_label = tk.Label(cap_panel, text="Відповідь:")
        self.cap_result_label.pack()

        # Minkowski Dimension Panel
        mink_panel = tk.LabelFrame(left_frame, text="Розмірність Мінковського")
        mink_panel.pack(pady=5, fill="x")

        tk.Label(mink_panel, text="Розмір комірки").pack()
        self.cell_size_entry = tk.Entry(mink_panel)
        self.cell_size_entry.pack()
        self.cell_size_entry.insert(0, "8")

        tk.Button(mink_panel, text="Розрахувати", command=self.calculate_minkowski).pack(pady=5)
        self.mink_result_label = tk.Label(mink_panel, text="Відповідь:")
        self.mink_result_label.pack()

        tk.Button(mink_panel, text="Залежність A delta від розміру комірки", command=self.dependence_delta).pack(pady=5)

        # Renyi Spectrum Panel
        renyi_panel = tk.LabelFrame(left_frame, text="Спектр Реньє")
        renyi_panel.pack(pady=5, fill="x")

        tk.Label(renyi_panel, text="q, min").pack()
        self.q_min_entry = tk.Entry(renyi_panel)
        self.q_min_entry.pack()
        self.q_min_entry.insert(0, "1")

        tk.Label(renyi_panel, text="q, max").pack()
        self.q_max_entry = tk.Entry(renyi_panel)
        self.q_max_entry.pack()
        self.q_max_entry.insert(0, "10")

        tk.Button(renyi_panel, text="Розрахувати", command=self.calculate_renyi).pack(pady=5)

        # Multifractal Spectrum Panel
        mf_panel = tk.LabelFrame(left_frame, text="Мультифрактальний спектр")
        mf_panel.pack(pady=5, fill="x")

        tk.Button(mf_panel, text="Розрахувати межі [alphaMin; alphaMax]", command=self.calculate_bounds).pack(pady=5)
        tk.Label(mf_panel, text="Min").pack()
        self.alpha_min_label = tk.Label(mf_panel, text="")
        self.alpha_min_label.pack()
        tk.Label(mf_panel, text="Max").pack()
        self.alpha_max_label = tk.Label(mf_panel, text="")
        self.alpha_max_label.pack()
        tk.Label(mf_panel, text="Крок розбиття").pack()
        self.split_step_entry = tk.Entry(mf_panel)
        self.split_step_entry.pack()
        self.split_step_entry.insert(0, "0.00001")
        tk.Button(mf_panel, text="Розрахувати", command=self.calculate_multifractal).pack(pady=5)

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.bmp;*.jpg;*.gif;*.png")])
        if self.image_path:
            image = Image.open(self.image_path)
            image.thumbnail((500, 500))
            self.image_tk = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.image_tk)
            self.image_label.image = self.image_tk

    def apply_threshold(self):
        if self.image_path:
            threshold = int(self.threshold_entry.get())
            self.fractal_dimension.black_boundary = threshold
            image = Image.open(self.image_path).convert('RGB')
            new_image = Image.new('RGB', image.size, 'white')
            for x in range(image.width):
                for y in range(image.height):
                    pixel = image.getpixel((x, y))
                    if pixel[0] <= threshold and pixel[1] <= threshold and pixel[2] <= threshold:
                        new_image.putpixel((x, y), (0, 0, 0))
            self.image_tk = ImageTk.PhotoImage(new_image)
            self.image_label.config(image=self.image_tk)
            self.image_label.image = self.image_tk

    def calculate_capacitive(self):
        if self.image_path:
            result = self.fractal_dimension.calculate_capacitive_dimension(self.image_path)
            self.cap_result_label.config(text=f"Відповідь: {result}")
            self.plot_graph(self.fractal_dimension.cd_points, "Графік апроксимації", "N(eps)", "eps")

    def calculate_minkowski(self):
        if self.image_path:
            epsilon = int(self.cell_size_entry.get())
            result = self.fractal_dimension.calculate_minkowski_dimension_for_grayscale_images(self.image_path, epsilon)
            self.mink_result_label.config(text=f"Відповідь: {result}")

    def dependence_delta(self):
        if self.image_path:
            self.fractal_dimension.calculate_minkowski_dimension_for_grayscale_images(self.image_path)
            self.plot_graph(self.fractal_dimension.md_points, "Графік залежності A delta від розміру комірки", "A delta", "eps")

    def calculate_renyi(self):
        if self.image_path:
            self.fractal_dimension.q_min = int(self.q_min_entry.get())
            self.fractal_dimension.q_max = int(self.q_max_entry.get())
            self.fractal_dimension.calculate_renyi_spectre(self.image_path)
            self.plot_graph(self.fractal_dimension.sr_points, "Графік залежності D(q) від значення q", "D(q)", "q")

    def calculate_bounds(self):
        if self.image_path:
            self.fractal_dimension.precalculate_alpha_min_alpha_max(self.image_path)
            self.alpha_min_label.config(text=f"{self.fractal_dimension.alpha_min * 1e5:.3f} * E-5")
            self.alpha_max_label.config(text=f"{self.fractal_dimension.alpha_max * 1e5:.3f} * E-5")

    def calculate_multifractal(self):
        if self.image_path:
            self.fractal_dimension.mf_epsilon = float(self.split_step_entry.get())
            self.fractal_dimension.calculate_multifractal_with_local_density_function()
            self.plot_bar_graph(self.fractal_dimension.mf_points, "Гістограма спектрів", "Ємнісна розмірність", "Діапазони alpha (*E-5)")

    def plot_graph(self, points, title, y_label, x_label):
        x_values = [p[0] for p in points]
        y_values = [p[1] for p in points]
        plt.figure()
        plt.plot(x_values, y_values, marker='o')
        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.show()

    def plot_bar_graph(self, points, title, y_label, x_label):
        x_labels = [f"від {p[0][0] * 1e5:.3f} до {p[0][1] * 1e5:.3f}" for p in points]
        y_values = [p[1] for p in points]
        plt.figure()
        plt.bar(x_labels, y_values)
        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = FractalGUI(root)
    root.mainloop()