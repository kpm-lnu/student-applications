import numpy as np


def l2_error(u_num, u_exact):
    """
    Обчислення похибки у нормі L² між чисельним та аналітичним розв’язком.

    :param u_num: чисельний розв’язок (вектор значень у вузлах)
    :param u_exact: точний (аналітичний) розв’язок у тих самих вузлах
    :return: похибка L²
    """
    return np.linalg.norm(u_num - u_exact) / np.linalg.norm(u_exact)


def max_abs_error(u_num, u_exact):
    """
    Максимальна абсолютна похибка у вузлах.

    :param u_num: чисельний розв’язок
    :param u_exact: точний розв’язок
    :return: max|u_num - u_exact|
    """
    return np.max(np.abs(u_num - u_exact))


def convergence_rate(errors, hs):
    """
    Оцінка порядку збіжності за методом найменших квадратів.
    Припускаємо: error ≈ Ch^p ⇒ log(error) = p*log(h) + log(C)

    :param errors: список похибок (L² або max), для кількох сіток
    :param hs: список відповідних кроків (макс. довжина сторони елемента)
    :return: оцінка порядку p
    """
    logs_h = np.log(hs)
    logs_e = np.log(errors)

    # МНК: p = (Σ(xy) - n·x̄·ȳ) / (Σ(x²) - n·x̄²)
    n = len(hs)
    x = logs_h
    y = logs_e
    p = (np.dot(x, y) - n * x.mean() * y.mean()) / (np.dot(x, x) - n * x.mean() ** 2)

    return p
