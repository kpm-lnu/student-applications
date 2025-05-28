import math
import sys
import numpy as np
import cv2

def gaussian_smoothing(image):
    input_image = np.array(image)
    smoothed_image = np.array(image)
    pixel_sum = 0

    for row in range(3, image.shape[0] - 3):
        for col in range(3, image.shape[1] - 3):
            pixel_sum = apply_gaussian_filter_at_point(input_image, row, col)
            smoothed_image[row][col] = pixel_sum

    return smoothed_image


def apply_gaussian_filter_at_point(image_data, row, col):
    pixel_sum = 0
    for i in range(row - 3, row + 4):
        for j in range(col - 3, col + 4):
            pixel_sum += gaussian_filter[i - row + 3][j - col + 3] * image_data[i][j]

    return pixel_sum

def get_gradient_x(image_data, height, width):
    gradient_x = np.empty(shape=(height, width))
    for row in range(3, height - 5):
        for col in range(3, image_data[row].size - 5):
            if is_invalid_region(image_data, row, col):
                gradient_x[row + 1][col + 1] = None
            else:
                gradient_x[row + 1][col + 1] = prewitt_at_x(image_data, row, col)

    return abs(gradient_x)


def get_gradient_y(image_data, height, width):
    gradient_y = np.empty(shape=(height, width))
    for row in range(3, height - 5):
        for col in range(3, image_data[row].size - 5):
            if is_invalid_region(image_data, row, col):
                gradient_y[row + 1][col + 1] = None
            else:
                gradient_y[row + 1][col + 1] = prewitt_at_y(image_data, row, col)

    return abs(gradient_y)


def get_magnitude(gradient_x, gradient_y, height, width):
    magnitude = np.empty(shape=(height, width))
    for row in range(height):
        for col in range(width):
            magnitude[row][col] = ((gradient_x[row][col] ** 2 + gradient_y[row][col] ** 2) ** 0.5) / 1.4142
    return magnitude


def get_angle(gradient_x, gradient_y, height, width):
    angle_matrix = np.empty(shape=(height, width))
    current_angle = 0
    for row in range(height):
        for col in range(width):
            if gradient_x[row][col] == 0:
                if gradient_y[row][col] > 0:
                    current_angle = 90
                else:
                    current_angle = -90
            else:
                current_angle = math.degrees(math.atan(gradient_y[row][col] / gradient_x[row][col]))
            if current_angle < 0:
                current_angle += 360
            angle_matrix[row][col] = current_angle
    return angle_matrix


def local_maximization(gradient_magnitude, gradient_angle, height, width):
    maximized_gradient = np.empty(shape=(height, width))
    pixel_histogram = np.zeros(shape=(256))
    edge_pixel_count = 0

    for row in range(5, height - 5):
        for col in range(5, image[row].size - 5):
            current_angle = gradient_angle[row, col]
            current_magnitude = gradient_magnitude[row, col]
            pixel_value = 0

            if (0 <= current_angle <= 22.5 or 157.5 < current_angle <= 202.5 or 337.5 < current_angle <= 360):
                if current_magnitude > gradient_magnitude[row, col + 1] and current_magnitude > gradient_magnitude[row, col - 1]:
                    pixel_value = current_magnitude
                else:
                    pixel_value = 0

            elif (22.5 < current_angle <= 67.5 or 202.5 < current_angle <= 247.5):
                if current_magnitude > gradient_magnitude[row + 1, col - 1] and current_magnitude > gradient_magnitude[row - 1, col + 1]:
                    pixel_value = current_magnitude
                else:
                    pixel_value = 0

            elif (67.5 < current_angle <= 112.5 or 247.5 < current_angle <= 292.5):
                if current_magnitude > gradient_magnitude[row + 1, col] and current_magnitude > gradient_magnitude[row - 1, col]:
                    pixel_value = current_magnitude
                else:
                    pixel_value = 0

            elif 112.5 < current_angle <= 157.5 or 292.5 < current_angle <= 337.5:
                if current_magnitude > gradient_magnitude[row + 1, col + 1] and current_magnitude > gradient_magnitude[row - 1, col - 1]:
                    pixel_value = current_magnitude
                else:
                    pixel_value = 0

            maximized_gradient[row, col] = pixel_value

            if pixel_value > 0:
                edge_pixel_count += 1
                try:
                    pixel_histogram[int(pixel_value)] += 1
                except:
                    print('Помилка: значення рівня сірого поза діапазоном', pixel_value)

    print('Кількість пікселів краю:', edge_pixel_count)
    return [maximized_gradient, pixel_histogram, edge_pixel_count]


def p_tile(percent, image_data, pixel_histogram, edge_pixel_count, original_image):
    threshold = np.around(edge_pixel_count * percent / 100)
    pixel_sum, threshold_value = 0, 255
    for threshold_value in range(255, 0, -1):
        pixel_sum += pixel_histogram[threshold_value]
        if pixel_sum >= threshold:
            break

    for row in range(image_data.shape[0]):
        for col in range(image_data[row].size):
            if image_data[row, col] < threshold_value:
                image_data[row, col] = 0
            else:
                image_data[row, col] = 255

    print(f'Для {percent}% - результат:')
    print('Загальна кількість пікселів після порогування:', pixel_sum)
    print('Порогове значення рівня сірого:', threshold_value)
    
    cv2.imwrite('Outputs/' + str(percent) + "_percent.jpg", image_data)

def is_invalid_region(image_data, row, col):
    return image_data[row][col] == None or image_data[row][col + 1] == None or image_data[row][col - 1] == None or \
           image_data[row + 1][col] == None or image_data[row + 1][col + 1] == None or image_data[row + 1][col - 1] == None or \
           image_data[row - 1][col] == None or image_data[row - 1][col + 1] == None or image_data[row - 1][col - 1] == None

def prewitt_at_x(image_data, row, col):
    horizontal_gradient = 0
    for i in range(0, 3):
        for j in range(0, 3):
            horizontal_gradient += image_data[row + i, col + j] * prewittX[i, j]
    return horizontal_gradient

def prewitt_at_y(image_data, row, col):
    vertical_gradient = 0
    for i in range(0, 3):
        for j in range(0, 3):
            vertical_gradient += image_data[row + i, col + j] * prewittY[i, j]
    return vertical_gradient

if __name__ == "__main__":

    gaussian_filter = (1.0 / 140.0) * np.array([[1, 1, 2, 2, 2, 1, 1],
                                                [1, 2, 2, 4, 2, 2, 1],
                                                [2, 2, 4, 8, 4, 2, 2],
                                                [2, 4, 8, 16, 8, 4, 2],
                                                [2, 2, 4, 8, 4, 2, 2],
                                                [1, 2, 2, 4, 2, 2, 1],
                                                [1, 1, 2, 2, 2, 1, 1]])

    prewittX = (1.0 / 3.0) * np.array([[-1, 0, 1],
                                       [-1, 0, 1],
                                       [-1, 0, 1]])

    prewittY = (1.0 / 3.0) * np.array([[1, 1, 1],
                                       [0, 0, 0],
                                       [-1, -1, -1]])


    image = cv2.imread('Images/photo_cat.jpg', 0)
    if image is None:
        print('Помилка: Не вдалося завантажити зображення')
        sys.exit(1)

    height = image.shape[0]
    width = image.shape[1]

    print('Застосовуємо фільтр Гауса')
    gaussianData = gaussian_smoothing(image)
    print(gaussianData)
    cv2.imwrite('Outputs/filter_gauss.jpg', gaussianData)

    print('\nОбчислюємо горизонтальний градієнт')
    Gx = get_gradient_x(gaussianData, height, width)
    cv2.imwrite('Outputs/XGradient.jpg', Gx)

    print('\nОбчислюємо вертикальний градієнт')
    Gy = get_gradient_y(gaussianData, height, width)
    cv2.imwrite('Outputs/YGradient.jpg', Gy)

    print('\nОбчислюємо величину градієнта')
    gradient = get_magnitude(Gx, Gy, height, width)
    cv2.imwrite('Outputs/Gradient.jpg', gradient)

    print('\nОбчислюємо кут градієнта')
    gradientAngle = get_angle(Gx, Gy, height, width)

    print('\nЗастосовуємо немаксимальне придушення')
    localMaxSuppressed = local_maximization(gradient, gradientAngle, height, width)
    cv2.imwrite('Outputs/MaximizedImage.jpg', localMaxSuppressed[0])

    suppressedImage = localMaxSuppressed[0]
    numberOfPixels = localMaxSuppressed[1]
    edgePixels = localMaxSuppressed[2]

    print('\nЗастосовуємо порогування')
    p_tile10 = p_tile(10, np.copy(suppressedImage), numberOfPixels, edgePixels, image)
    p_tile20 = p_tile(30, np.copy(suppressedImage), numberOfPixels, edgePixels, image)
    p_tile30 = p_tile(50, np.copy(suppressedImage), numberOfPixels, edgePixels, image)

    print('\nШукаємо котиків на зображенні')
    cat_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
    if cat_cascade.empty():
        print('Помилка: Не вдалося завантажити каскадний класифікатор')
        sys.exit(1)

    cats = cat_cascade.detectMultiScale(gaussianData, scaleFactor=1.1, minNeighbors=3)

    for (x, y, w, h) in cats:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    print(f'Знайдено котиків: {len(cats)}')
    cv2.imwrite('Outputs/cat_detected.jpg', image)
    print('Обробку завершено. Результати збережено в папці Outputs')
