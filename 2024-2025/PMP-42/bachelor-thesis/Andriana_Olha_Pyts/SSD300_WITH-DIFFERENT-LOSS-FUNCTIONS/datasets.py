import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform


class PascalVOCDataset(Dataset):
    """
    Клас набору даних PyTorch для використання в завантажувачі даних PyTorch для створення пакетів.
    """
    def __init__(self, data_folder, split, keep_difficult=False):
        """
        :param data_folder: папка, де зберігаються файли даних
        :param split: розділити, одне з «TRAIN» або «TEST»
        :param keep_difficult: зберегти або відкинути об'єкти, які вважаються складними для виявлення?
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Читання файлів даних
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Прочитайте зображення
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # Прочитайте об’єкти на цьому зображенні (обмежувальні рамки, написи, труднощі)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Відмовтеся від складних предметів, якщо хочете
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Застосувати перетворення
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
       Оскільки кожне зображення може мати різну кількість об’єктів, нам потрібна функція сортування (яка буде передана в DataLoader).
        Тут описано, як поєднати ці тензори різних розмірів. Ми використовуємо списки.
        Примітка: це не потрібно визначати в цьому класі, може бути автономним.
        :param batch: ітерація з N наборів із __getitem__()
        :return: тензор зображень, списки тензорів різного розміру обмежувальних рамок, мітки та труднощі
        """
        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)
        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each