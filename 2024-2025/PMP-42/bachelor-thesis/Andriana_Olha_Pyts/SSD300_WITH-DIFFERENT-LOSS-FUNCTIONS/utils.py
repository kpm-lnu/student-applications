import json
import os
import torch
import random
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as FT
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Карта етикетки
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

# Колірна карта для обмежувальних рамок виявлених об’єктів з https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}


def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()
    for object in root.iter('object'):

        difficult = int(object.find('difficult').text == '1')

        label = object.find('name').text.lower().strip()
        if label not in label_map:
            continue

        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}


def create_data_lists(voc07_path, output_folder):
    """
    Створіть списки зображень, обмежувальні рамки та мітки об’єктів на цих зображеннях і збережіть їх у файлі.
    :param voc07_path: шлях до папки "VOC2007".
    :param voc12_path: шлях до папки "VOC2012".
    :param output_folder: папка, де потрібно зберегти файли JSON
    """
    voc07_path = os.path.abspath(voc07_path)
    #voc12_path = os.path.abspath(voc12_path)

    train_images = list()
    train_objects = list()
    n_objects = 0

    # Дані про навчання
    for path in [voc07_path]:

        # Знайдіть ідентифікатори зображень у навчальних даних
        with open(os.path.join(path, 'ImageSets/Main/trainval.txt')) as f:
            ids = f.read().splitlines()

        for id in ids:
            # Проаналізуйте XML-файл анотації
            objects = parse_annotation(os.path.join(path, 'Annotations', id + '.xml'))
            if len(objects['boxes']) == 0:
                continue
            n_objects += len(objects)
            train_objects.append(objects)
            train_images.append(os.path.join(path, 'JPEGImages', id + '.jpg'))

    assert len(train_objects) == len(train_images)

    # Зберегти у файл
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)  # save label map too

    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_images), n_objects, os.path.abspath(output_folder)))

    # Тестові дані
    test_images = list()
    test_objects = list()
    n_objects = 0

    # Знайдіть ідентифікатори зображень у тестових даних
    with open(os.path.join(voc07_path, 'ImageSets/Main/test.txt')) as f:
        ids = f.read().splitlines()

    for id in ids:
        # Проаналізуйте XML-файл анотації
        objects = parse_annotation(os.path.join(voc07_path, 'Annotations', id + '.xml'))
        if len(objects) == 0:
            continue
        test_objects.append(objects)
        n_objects += len(objects)
        test_images.append(os.path.join(voc07_path, 'JPEGImages', id + '.jpg'))

    assert len(test_objects) == len(test_images)

    # Зберегти у файл
    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print('\nThere are %d test images containing a total of %d objects. Files have been saved to %s.' % (
        len(test_images), n_objects, os.path.abspath(output_folder)))


def decimate(tensor, m):
    """
    Знищити тензор на коефіцієнт 'm', тобто зменшити дискретизацію, зберігаючи кожне 'm'-е значення.
    Це використовується, коли ми перетворюємо шари FC на еквівалентні згорточні шари, АЛЕ меншого розміру.
    :param tensor: тензор, який буде знищено
    :param m: список коефіцієнтів децимації для кожного розміру тензора; Жодного, якщо не знищувати вздовж виміру
    :return: знищений тензор
    """
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor


def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties):
    """
    Обчисліть середню точність (mAP) виявлених об’єктів.
    Перегляньте пояснення на сторінці https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173
    :param det_boxes: список тензорів, один тензор для кожного зображення, що містить обмежувальні рамки виявлених об’єктів
    :param det_labels: список тензорів, один тензор для кожного зображення, що містить мітки виявлених об'єктів
    :param det_scores: список тензорів, один тензор для кожного зображення, що містить бали міток виявлених об'єктів
    :param true_boxes: список тензорів, один тензор для кожного зображення, що містить обмежувальні прямокутники фактичних об’єктів
    :param true_labels: список тензорів, один тензор для кожного зображення, що містить мітки фактичних об’єктів
    :param true_difficulties: список тензорів, один тензор для кожного зображення, що містить реальні об’єкти (0 або 1)
    :return: список середньої точності для всіх класів, середня середня точність (mAP)
    """
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(
        true_labels) == len(
        true_difficulties)  # these are all lists of tensors of the same length, i.e. number of images
    n_classes = len(label_map)

    # Зберігайте всі (справжні) об’єкти в одному безперервному тензорі, одночасно відстежуючи зображення, з якого вони походять
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(
        device)  # (n_objects), n_objects is the total no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Зберігайте всі виявлення в одному безперервному тензорі, одночасно відстежуючи зображення, з якого воно походить
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Розрахувати AP для кожного класу (окрім фону)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    for c in range(1, n_classes):
        # Витягувати лише об’єкти з цим класом
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects

        # Слідкуйте за тим, які справжні об'єкти з цим класом вже були «виявлені»
        # Поки що жодного
        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(
            device)  # (n_class_objects)

        # Витягувати лише виявлення за допомогою цього класу
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        # Сортувати виявлення в порядку зменшення достовірності/балів
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # У порядку зменшення балів перевірте, чи вірний чи хибний позитивний результат
        true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Знайдіть об’єкти на одному зображенні з цим класом, їхні труднощі та чи були вони виявлені раніше
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # Якщо на цьому зображенні немає такого об’єкта, виявлення є помилковим
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Знайти максимальне перекриття цього виявлення з об’єктами на цьому зображенні цього класу
            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' — це індекс об'єкта в цих тензорах рівня зображення 'object_boxes', 'object_difficulties'
            # В оригінальних тензорах рівня класу 'true_class_boxes' тощо, 'ind' відповідає об'єкту з індексом...
            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
            # Нам потрібен 'original_ind' для оновлення 'true_class_boxes_detected'

            # Якщо максимальне перекриття перевищує порогове значення 0,5, це збіг
            if max_overlap.item() > 0.5:
                # Якщо об’єкт, з яким він зіставився, є «складним», ігноруйте його
                if object_difficulties[ind] == 0:
                    # Якщо цей об'єкт ще не виявлено, це справді позитивно
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                    # В іншому випадку це помилковий результат (оскільки цей об’єкт уже обліковується)
                    else:
                        false_positives[d] = 1
            # В іншому випадку виявлення відбувається в іншому місці, ніж фактичний об’єкт, і є хибним спрацьовуванням
            else:
                false_positives[d] = 1

        # Обчисліть кумулятивну точність і відкликання при кожному виявленні в порядку зменшення балів
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

        # Знайдіть середнє значення максимальної точності, що відповідає відкликанням вище порогового значення «t»
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    # Розрахувати середню середню точність (mAP)
    mean_average_precision = average_precisions.mean().item()

    # Зберігайте у словнику середні класові значення
    average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

    return average_precisions, mean_average_precision


def xy_to_cxcy(xy):
    """
    Перетворіть обмежувальні прямокутники з граничних координат (x_min, y_min, x_max, y_max) на координати центрального розміру (c_x, c_y, w, h).
    :param xy: обмежувальні рамки в граничних координатах, тензор розміру (n_boxes, 4)
    :return: обмежувальні рамки в центральних координатах розміру, тензор розміру (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy):
    """
    Перетворіть обмежувальні рамки з координат центру (c_x, c_y, w, h) на граничні координати (x_min, y_min, x_max, y_max).
    :param cxcy: обмежувальні рамки в центральних координатах розміру, тензор розміру (n_boxes, 4)
    :return: обмежувальні рамки в граничних координатах, тензор розміру (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Закодуйте обмежувальні прямокутники (які мають центральний розмір) з.р.т. відповідні попередні коробки (які мають центральний розмір).
    Для координат центру знайдіть зміщення відносно попереднього блоку та масштабуйте за розміром попереднього блоку.
    Для координат розміру масштабуйте за розміром попереднього блоку та перетворіть на простір журналу.
    У моделі ми передбачаємо координати обмежувальної рамки в цій закодованій формі.
    :param cxcy: обмежувальні рамки в координатах розміру центру, тензор розміру (n_priors, 4)
    :param priors_cxcy: попередні блоки, щодо яких потрібно виконати кодування, тензор розміру (n_priors, 4)
    :return: закодовані обмежувальні рамки, тензор розміру (n_priors, 4)
    """

    # 10 і 5 нижче називаються «варіаціями» в оригінальному репо Caffe, повністю емпіричними
    # Вони призначені для певного чисельного обумовлення, для «масштабування градієнта локалізації»
    # Перегляньте https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Декодуйте координати обмежувальної рамки, передбачені моделлю, оскільки вони закодовані у формі, згаданій вище.
    Вони декодуються в координати розміру центру.
    Це обернена функція вище.
    :param gcxgcy: закодовані обмежувальні рамки, тобто вихід моделі, тензор розміру (n_priors, 4)
    :param priors_cxcy: попередні блоки, щодо яких визначено кодування, тензор розміру (n_priors, 4)
    :return: декодовані обмежувальні рамки у формі центрального розміру, тензор розміру (n_priors, 4)
    """

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h


def find_intersection(set_1, set_2):
    """
    Знайдіть перетин кожної комбінації коробок між двома наборами коробок, які знаходяться в граничних координатах.
    :param set_1: набір 1, тензор розмірів (n1, 4)
    :param set_2: набір 2, тензор розмірів (n2, 4)
    :return: перетин кожного з ящиків у наборі 1 відносно кожного з ящиків у наборі 2, тензор розмірів (n1, n2)
    """

    # PyTorch автоматично транслює одиночні розміри
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Знайдіть перекриття Jaccard (IoU) кожної комбінації коробок між двома наборами коробок, які знаходяться в граничних координатах.
    :param set_1: набір 1, тензор розмірів (n1, 4)
    :param set_2: набір 2, тензор розмірів (n2, 4)
    :return: Жаккардове перекриття кожного з ящиків у наборі 1 щодо кожного з ящиків у наборі 2, тензор розмірів (n1, n2)
    """

    # Знайдіть перехрестя
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Знайдіть площі кожної коробки в обох наборах
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Знайдіть союз
    # PyTorch автоматично транслює одиночні розміри
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


# Деякі функції розширення, наведені нижче, були адаптовані з
# З https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
def expand(image, boxes, filler):
    """
    Виконайте операцію зменшення, розмістивши зображення на більшому полотні наповнювача.
    Допомагає навчитися виявляти менші предмети.
    :param image: зображення, тензор розмірів (3, original_h, original_w)
    :param boxes: обмежувальні рамки в граничних координатах, тензор розмірів (n_objects, 4)
    :param filler: значення RBG матеріалу наповнювача, список як [R, G, B]
    :return: розширене зображення, оновлені координати обмежувальної рамки
    """
    # Розрахувати розміри запропонованого розгорнутого (зменшеного) зображення
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    # Створіть такий образ за допомогою наповнювача
    filler = torch.FloatTensor(filler)  # (3)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)  # (3, new_h, new_w)
    # Примітка - не використовуйте expand() як new_image = filler.unsqueeze(1).unsqueeze(1).expand(3, new_h, new_w)
    # тому що всі розширені значення будуть використовувати ту саму пам’ять, тому зміна одного пікселя змінить усі

    # Розмістіть вихідне зображення у випадкових координатах цього нового зображення (початкова точка у верхньому лівому куті зображення)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image

    # Відповідно відкоригуйте координати обмежувальних рамок
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(
        0)  # (n_objects, 4), n_objects is the no. of objects in this image

    return new_image, new_boxes


def random_crop(image, boxes, labels, difficulties):
    """
    Виконує довільне обрізання, як зазначено в документі. Допомагає навчитися виявляти більші та часткові об'єкти.
    Зверніть увагу, що деякі об’єкти можуть бути вирізані повністю.
    Адаптовано з https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
    :param image: зображення, тензор розмірів (3, original_h, original_w)
    :param boxes: обмежувальні рамки в граничних координатах, тензор розмірів (n_objects, 4)
    :param labels: мітки об'єктів, тензор розмірів (n_objects)
    :param труднощі: труднощі виявлення цих об'єктів, тензор розмірності (n_objects)
    :return: обрізане зображення, оновлені координати обмежувальної рамки, оновлені мітки, оновлені труднощі
    """
    original_h = image.size(1)
    original_w = image.size(2)
    # Продовжуйте вибирати мінімальне перекриття, доки не буде отриманий врожай
    while True:
        # Довільно намалюйте значення для мінімального перекриття
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])  # 'None' refers to no cropping

        # Якщо не кадрування
        if min_overlap is None:
            return image, boxes, labels, difficulties

        # Спробуйте до 50 разів для цього вибору мінімального перекриття
        # Звичайно, це не згадується в статті, але 50 вибрано в оригінальному репо Caffe авторів статті
        max_trials = 50
        for _ in range(max_trials):
            # Розміри кадрування мають бути в [0,3, 1] від вихідних розмірів
            # Примітка - це [0.1, 1] у статті, але насправді [0.3, 1] у сховищі авторів
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            # Співвідношення сторін має бути [0,5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # Координати кадрування (початок у верхньому лівому куті зображення)
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])  # (4)

            # Обчисліть перекриття Jaccard між кадруванням і обмежувальними рамками
            overlap = find_jaccard_overlap(crop.unsqueeze(0),
                                           boxes)  # (1, n_objects), n_objects is the no. of objects in this image
            overlap = overlap.squeeze(0)  # (n_objects)

            # Якщо жоден обмежувальний прямокутник не має перекриття Jaccard більше, ніж мінімальне, повторіть спробу
            if overlap.max().item() < min_overlap:
                continue

            # Обрізати зображення
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)

            # Знайдіть центри оригінальних обмежувальних рамок
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # (n_objects, 2)

            # Знайдіть рамки, центри яких знаходяться в кадрі
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                    bb_centers[:, 1] < bottom)  # (n_objects), a Torch uInt8/Byte tensor, can be used as a boolean index

            # Якщо жодна обмежувальна рамка не має свого центру в кадрі, повторіть спробу
            if not centers_in_crop.any():
                continue

            # Відмовтеся від обмежувальних рамок, які не відповідають цьому критерію
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]

            # Обчисліть нові координати обмежувальних рамок у кадрі
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])  # crop[:2] is [left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])  # crop[2:] is [right, bottom]
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels, new_difficulties


def flip(image, boxes):
    """
    Перевернути зображення по горизонталі.
    :param image: зображення, зображення PIL
    :param boxes: обмежувальні рамки в граничних координатах, тензор розмірів (n_objects, 4)
    :return: перевернуте зображення, оновлені координати обмежувальної рамки
    """
    # Перевернути зображення
    new_image = FT.hflip(image)

    # Фліп коробки
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes


def resize(image, boxes, dims=(300, 300), return_percent_coords=True):
    """
    Змінити розмір зображення. Для SSD300 змініть розмір на (300, 300).
    Оскільки в цьому процесі відсоткові/дробні координати обчислюються для обмежувальних рамок (відносно розмірів зображення),
    ви можете зберегти їх.
    :param image: зображення, зображення PIL
    :param boxes: обмежувальні рамки в граничних координатах, тензор розмірів (n_objects, 4)
    :return: змінений розмір зображення, оновлені координати обмежувальної рамки (або дробові координати, у цьому випадку вони залишаються незмінними)
    """
    # Змінити розмір зображення
    new_image = FT.resize(image, dims)

    # Змініть розмір обмежувальних рамок
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes


def photometric_distort(image):
    """
    Викривляйте яскравість, контраст, насиченість і відтінок у довільному порядку з імовірністю 50%.
    :param image: зображення, зображення PIL
    :return: спотворене зображення
    """
    new_image = image

    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ is 'adjust_hue':
                # Caffe repo використовує 'hue_delta' 18 — ми ділимо на 255, оскільки PyTorch потребує нормалізованого значення
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                # Caffe repo використовує «нижні» та «верхні» значення 0,5 та 1,5 для яскравості, контрасту та насиченості
                adjust_factor = random.uniform(0.5, 1.5)

            # Застосуйте це спотворення
            new_image = d(new_image, adjust_factor)

    return new_image
  

def transform(image, boxes, labels, difficulties, split):
    """
    Застосуйте наведені вище перетворення.
    :param image: зображення, зображення PIL
    :param boxes: обмежувальні рамки в граничних координатах, тензор розмірів (n_objects, 4)
    :param labels: мітки об'єктів, тензор розмірів (n_objects)
    :param труднощі: труднощі виявлення цих об'єктів, тензор розмірності (n_objects)
    :param split: одне з «TRAIN» або «TEST», оскільки застосовуються різні набори перетворень
    :return: трансформоване зображення, трансформовані координати обмежувальної рамки, трансформовані мітки, трансформовані труднощі
    """
    assert split in {'TRAIN', 'TEST'}

    # Середнє значення та стандартне відхилення даних ImageNet, на яких навчався наш базовий VGG із torchvision
    # див.: https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties
    # Пропустіть наступні операції для оцінки/тестування
    if split == 'TRAIN':
        # Серія фотометричних спотворень у випадковому порядку, кожне з шансом появи 50%, як у Caffe repo
        new_image = photometric_distort(new_image)

        # Перетворення зображення PIL на тензор Torch
        new_image = FT.to_tensor(new_image)

        # Розгортання зображення (зменшення масштабу) з імовірністю 50% - корисно для навчання виявлення дрібних об'єктів
        # Заповнити навколишній простір даними ImageNet, на яких навчався наш базовий VGG
        if random.random() < 0.5:
            new_image, new_boxes = expand(new_image, boxes, filler=mean)

        # Довільне обрізання зображення (збільшення)
        new_image, new_boxes, new_labels, new_difficulties = random_crop(new_image, new_boxes, new_labels,
                                                                         new_difficulties)

        # Перетворення тензора Torch на зображення PIL
        new_image = FT.to_pil_image(new_image)

        # Перевернути зображення з імовірністю 50%.
        if random.random() < 0.5:
            new_image, new_boxes = flip(new_image, new_boxes)

    # Змінити розмір зображення до (300, 300) - це також перетворює абсолютні координати границь у їхню дробову форму
    new_image, new_boxes = resize(new_image, new_boxes, dims=(300, 300))

    # Перетворення зображення PIL на тензор Torch
    new_image = FT.to_tensor(new_image)

    # Нормалізація за середнім і стандартним відхиленням даних ImageNet, на яких навчався наш базовий VGG
    new_image = FT.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes, new_labels, new_difficulties


def adjust_learning_rate(optimizer, scale):
    """
    Шкала швидкості навчання за вказаним фактором.
    :param optimizer: оптимізатор, швидкість навчання якого має бути зменшена.
    :param scale: коефіцієнт для множення швидкості навчання.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("DECAYING learning rate.\n The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))


def accuracy(scores, targets, k):
    """
    Обчислює найвищу точність за прогнозованими та справжніми мітками.
    :param scores: оцінки з моделі
    :param targets: справжні мітки
    :param k: k із точністю top-k
    :return: найвища точність
    """
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def save_checkpoint(epoch, model, optimizer):
    """
    Зберегти контрольну точку моделі.
    :param epoch: номер епохи
    :param модель: модель
    :param optimizer: оптимізатор
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoint_ssd300.pth.tar'
    torch.save(state, filename)


class AverageMeter(object):
    """
    Відстежує останні, середні значення, суму та кількість показників.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    Відсікає градієнти, обчислені під час зворотного поширення, щоб уникнути вибуху градієнтів.
    :param optimizer: оптимізатор із градієнтами, які потрібно обрізати
    :param grad_clip: значення кліпу
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
