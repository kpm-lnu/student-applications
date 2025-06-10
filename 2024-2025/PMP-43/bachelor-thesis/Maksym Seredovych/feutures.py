import numpy as np
from wavlet_transfroms import get_wavelet_coefficients
import numpy as np
from scipy.stats import entropy as scipy_entropy

def compute_entropy(arr: np.ndarray) -> float:
    """
    Computes the Shannon entropy of the array based on its value distribution.
    """
    if arr.size == 0:
        return 0.0
    
    values, counts = np.unique(arr.flatten(), return_counts=True)
    probabilities = counts / counts.sum()
    
    return scipy_entropy(probabilities, base=2)


def compute_wavelet_features_from_coeffs_to_dict(coeffs, compute_entropy_function) -> list:
    """
    Обчислює статистичні характеристики (середнє, стандартне відхилення, дисперсію, ентропію)
    із коефіцієнтів вейвлет-перетворення та зберігає їх у списку словників.
    Кожен словник також містить інформацію про тип та рівень коефіцієнтів.

    Параметри:
        coeffs: Список коефіцієнтів, отриманих за допомогою pywt.wavedec2.
                Очікуваний формат: [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]
        compute_entropy_function: Функція, яка приймає np.ndarray та повертає значення ентропії (float).

    Повертає:
        list: Список словників. Кожен словник представляє один суб-діапазон (sub-band)
              та містить його статистичні характеристики та метадані.
              Приклад словника:
              {
                  'type': 'approximation',  # Тип коефіцієнта
                  'level': 2,               # Рівень розкладу
                  'mean': 0.123,            # Середнє значення
                  'std': 0.045,             # Стандартне відхилення
                  'var': 0.002,             # Дисперсія
                  'entropy': 1.234          # Ентропія
              }
    """
    features_list_of_dicts = []
    
    if not coeffs:
        return features_list_of_dicts
    
    decomposition_level_n = len(coeffs) - 1

    approx_coeffs_arr = coeffs[0]
    coeff_dict_approx = {
        'type': 'approximation',
        'level': decomposition_level_n,
        'mean': np.mean(approx_coeffs_arr),
        'std': np.std(approx_coeffs_arr),
        'var': np.var(approx_coeffs_arr),
        'entropy': compute_entropy_function(approx_coeffs_arr)
    }
    features_list_of_dicts.append(coeff_dict_approx)
    
    subband_types = ['horizontal_detail', 'vertical_detail', 'diagonal_detail']
    
    for i in range(1, len(coeffs)):
        detail_coeffs_tuple = coeffs[i]
        current_level = decomposition_level_n - (i - 1) 
        
        if isinstance(detail_coeffs_tuple, tuple) and len(detail_coeffs_tuple) == 3:
            for j, single_detail_arr in enumerate(detail_coeffs_tuple):
                coeff_dict_detail = {
                    'type': subband_types[j],
                    'level': current_level,
                    'mean': np.mean(single_detail_arr),
                    'std': np.std(single_detail_arr),
                    'var': np.var(single_detail_arr),
                    'entropy': compute_entropy_function(single_detail_arr)
                }
                features_list_of_dicts.append(coeff_dict_detail)
        else:
            print(f"Warning: Expected a tuple of 3 detail coefficients at coeffs index {i}, "
                  f"but found {type(detail_coeffs_tuple)}. Level: {current_level}")
    return features_list_of_dicts

def get_image_features(image_path: str, wavelet: str = 'db4', level: int = 3) -> np.ndarray:
    """
    Отримує вектор ознак з зображення шляхом обчислення вейвлет коефіцієнтів
    та подальшого розрахунку статистичних характеристик.
    
    Параметри:
        image_path (str): Шлях до зображення.
        wavelet (str): Тип вейвлету.
        level (int): Рівень розкладання.
    
    Повертає:
        np.ndarray: Вектор ознак.
    """
    coeffs = get_wavelet_coefficients(image_path, wavelet, level)
    features = compute_wavelet_features_from_coeffs_to_dict(coeffs)
    return features