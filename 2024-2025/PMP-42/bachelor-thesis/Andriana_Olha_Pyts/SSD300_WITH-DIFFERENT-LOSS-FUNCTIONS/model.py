from torch import nn
from utils import *
import torch.nn.functional as F
from math import sqrt
from itertools import product as product
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGGBase(nn.Module):
    """
Базові згортки VGG для створення карт функцій нижчого рівня.
    """

    def __init__(self):
        super(VGGBase, self).__init__()

        # Стандартні згорточні шари у VGG16
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # stride = 1, by default
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # ceiling (not floor) here for even dims

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # retains size because stride is 1 (and padding)

        # Заміна для FC6 і FC7 у VGG16
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)  # atrous convolution

        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        # Завантажте попередньо підготовлені шари
        self.load_pretrained_layers()

    def forward(self, image):
        """
        Пряме поширення.

        :param image: зображення, тензор розмірів (N, 3, 300, 300)
        :return: карти функцій нижчого рівня conv4_3 і conv7
        """
        out = F.relu(self.conv1_1(image))  # (N, 64, 300, 300)
        out = F.relu(self.conv1_2(out))  # (N, 64, 300, 300)
        out = self.pool1(out)  # (N, 64, 150, 150)

        out = F.relu(self.conv2_1(out))  # (N, 128, 150, 150)
        out = F.relu(self.conv2_2(out))  # (N, 128, 150, 150)
        out = self.pool2(out)  # (N, 128, 75, 75)

        out = F.relu(self.conv3_1(out))  # (N, 256, 75, 75)
        out = F.relu(self.conv3_2(out))  # (N, 256, 75, 75)
        out = F.relu(self.conv3_3(out))  # (N, 256, 75, 75)
        out = self.pool3(out)  # (N, 256, 38, 38), it would have been 37 if not for ceil_mode = True

        out = F.relu(self.conv4_1(out))  # (N, 512, 38, 38)
        out = F.relu(self.conv4_2(out))  # (N, 512, 38, 38)
        out = F.relu(self.conv4_3(out))  # (N, 512, 38, 38)
        conv4_3_feats = out  # (N, 512, 38, 38)
        out = self.pool4(out)  # (N, 512, 19, 19)

        out = F.relu(self.conv5_1(out))  # (N, 512, 19, 19)
        out = F.relu(self.conv5_2(out))  # (N, 512, 19, 19)
        out = F.relu(self.conv5_3(out))  # (N, 512, 19, 19)
        out = self.pool5(out)  # (N, 512, 19, 19), pool5 does not reduce dimensions

        out = F.relu(self.conv6(out))  # (N, 1024, 19, 19)

        conv7_feats = F.relu(self.conv7(out))  # (N, 1024, 19, 19)

        # Lower-level feature maps
        return conv4_3_feats, conv7_feats

    def load_pretrained_layers(self):
        """
        Як і в статті, ми використовуємо VGG-16, попередньо навчений на завдання ImageNet, як базову мережу.
        Є один доступний у PyTorch, див. https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16
        Ми копіюємо ці параметри в нашу мережу. Це просто від conv1 до conv5.
        Однак оригінальний VGG-16 не містить рівнів conv6 і con7.
        Тому ми перетворюємо fc6 ​​і fc7 у згорткові шари та виконуємо підвибірку шляхом проріджування. Дивіться 'decimate' у utils.py.
        """
        # Поточний стан бази
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Попередньо підготовлена ​​база VGG
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        # Передача конв. параметри від попередньо навченої моделі до поточної моделі
        for i, param in enumerate(param_names[:-4]):  # excluding conv6 and conv7 parameters
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        # Перетворення fc6, fc7 на згорткові шари, а підвибірку (проріджуванням) на розміри conv6 і conv7
        # fc6
        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)  # (4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias']  # (4096)
        state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
        state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])  # (1024)
        # fc7
        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
        state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])  # (1024)

        # Примітка: шар FC розміром (K), що працює на розплющеній версії (C*H*W) двовимірного зображення розміром (C, H, W)...
        # ...еквівалентний згортковому шару з розміром ядра (H, W), вхідними каналами C, вихідними каналами K...
        # ...працює з двовимірним зображенням розміру (C, H, W) без заповнення

        self.load_state_dict(state_dict)

        print("\nLoaded base model.\n")


class AuxiliaryConvolutions(nn.Module):
    """
    Додаткові згортки для створення карт функцій вищого рівня.
    """

    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()

        # Допоміжні/додаткові звивини поверх основи VGG
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)  # stride = 1, by default
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # dim. reduction because padding = 0

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # dim. reduction because padding = 0

        # Ініціалізація параметрів згорток
        self.init_conv2d()

    def init_conv2d(self):
        """
        Ініціалізація параметрів згортки.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv7_feats):
        """
        Пряме поширення.

        :param conv7_feats: карта функцій нижчого рівня conv7, тензор розмірів (N, 1024, 19, 19)
        :return: карти функцій вищого рівня conv8_2, conv9_2, conv10_2 і conv11_2
        """
        out = F.relu(self.conv8_1(conv7_feats))  # (N, 256, 19, 19)
        out = F.relu(self.conv8_2(out))  # (N, 512, 10, 10)
        conv8_2_feats = out  # (N, 512, 10, 10)

        out = F.relu(self.conv9_1(out))  # (N, 128, 10, 10)
        out = F.relu(self.conv9_2(out))  # (N, 256, 5, 5)
        conv9_2_feats = out  # (N, 256, 5, 5)

        out = F.relu(self.conv10_1(out))  # (N, 128, 5, 5)
        out = F.relu(self.conv10_2(out))  # (N, 256, 3, 3)
        conv10_2_feats = out  # (N, 256, 3, 3)

        out = F.relu(self.conv11_1(out))  # (N, 128, 3, 3)
        conv11_2_feats = F.relu(self.conv11_2(out))  # (N, 256, 1, 1)

        # Карти функцій вищого рівня
        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats
  

class PredictionConvolutions(nn.Module):
    """
    Згортки для прогнозування результатів класу та обмежувальні рамки за допомогою карт функцій нижчого та вищого рівнів.

    Обмежувальні прямокутники (розташування) передбачені як закодовані зміщення відносно кожного з 8732 попередніх (за замовчуванням) полів.
    Перегляньте 'cxcy_to_gcxgcy' в utils.py для визначення кодування.

    Оцінки класів представляють бали кожного класу об’єктів у кожному з 8732 розташованих обмежувальних рамок.
    Високий бал для «фону» = відсутність об’єкта.
    """

    def __init__(self, n_classes):
        """
        :параметр n_classes: кількість різних типів об'єктів
        """
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        # Кількість попередніх блоків, які ми розглядаємо на позицію в кожній карті функцій
        n_boxes = {'conv4_3': 4,
                   'conv7': 6,
                   'conv8_2': 6,
                   'conv9_2': 6,
                   'conv10_2': 4,
                   'conv11_2': 4}
        # 4 попередні вікна означають, що ми використовуємо 4 різні співвідношення сторін тощо.

        # Згортки передбачення локалізації (передбачити зміщення відносно попередніх вікон)
        self.loc_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * 4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * 4, kernel_size=3, padding=1)
        self.loc_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * 4, kernel_size=3, padding=1)

        # Згортки передбачення класу (прогнозування класів у вікнах локалізації)
        self.cl_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * n_classes, kernel_size=3, padding=1)

        # Ініціалізація параметрів згорток
        self.init_conv2d()

    def init_conv2d(self):
        """
        Ініціалізація параметрів згортки.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):
        """
        Пряме поширення.

        :param conv4_3_feats: карта функцій conv4_3, тензор розмірів (N, 512, 38, 38)
        :param conv7_feats: карта функцій conv7, тензор розмірів (N, 1024, 19, 19)
        :param conv8_2_feats: conv8_2 карта функцій, тензор розмірів (N, 512, 10, 10)
        :param conv9_2_feats: conv9_2 карта функцій, тензор розмірів (N, 256, 5, 5)
        :param conv10_2_feats: conv10_2 карта функцій, тензор розмірів (N, 256, 3, 3)
        :param conv11_2_feats: conv11_2 карта функцій, тензор розмірів (N, 256, 1, 1)
        :return: 8732 локації та оцінки класів (тобто без попереднього вікна) для кожного зображення
        """
        batch_size = conv4_3_feats.size(0)

        # Передбачити межі блоків локалізації (як зміщення відносно попередніх блоків)
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)  # (N, 16, 38, 38)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3,
                                      1).contiguous()  # (N, 38, 38, 16), to match prior-box order (after .view())
        # (.contiguous() гарантує, що він зберігається в безперервній частині пам’яті, необхідної для .view() нижче)
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)  # (N, 5776, 4), there are a total 5776 boxes on this feature map

        l_conv7 = self.loc_conv7(conv7_feats)  # (N, 24, 19, 19)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 24)
        l_conv7 = l_conv7.view(batch_size, -1, 4)  # (N, 2166, 4), there are a total 2116 boxes on this feature map

        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)  # (N, 24, 10, 10)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 24)
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)  # (N, 600, 4)

        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)  # (N, 24, 5, 5)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 24)
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)  # (N, 150, 4)

        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)  # (N, 16, 3, 3)
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 16)
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)  # (N, 36, 4)

        l_conv11_2 = self.loc_conv11_2(conv11_2_feats)  # (N, 16, 1, 1)
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 16)
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)  # (N, 4, 4)

        # Передбачення класів у вікнах локалізації
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)  # (N, 4 * n_classes, 38, 38)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3,
                                      1).contiguous()  # (N, 38, 38, 4 * n_classes), to match prior-box order (after .view())
        c_conv4_3 = c_conv4_3.view(batch_size, -1,
                                   self.n_classes)  # (N, 5776, n_classes), there are a total 5776 boxes on this feature map

        c_conv7 = self.cl_conv7(conv7_feats)  # (N, 6 * n_classes, 19, 19)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 6 * n_classes)
        c_conv7 = c_conv7.view(batch_size, -1,
                               self.n_classes)  # (N, 2166, n_classes), there are a total 2116 boxes on this feature map

        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)  # (N, 6 * n_classes, 10, 10)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 6 * n_classes)
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)  # (N, 600, n_classes)

        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)  # (N, 6 * n_classes, 5, 5)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 6 * n_classes)
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)  # (N, 150, n_classes)

        c_conv10_2 = self.cl_conv10_2(conv10_2_feats)  # (N, 4 * n_classes, 3, 3)
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 4 * n_classes)
        c_conv10_2 = c_conv10_2.view(batch_size, -1, self.n_classes)  # (N, 36, n_classes)

        c_conv11_2 = self.cl_conv11_2(conv11_2_feats)  # (N, 4 * n_classes, 1, 1)
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 4 * n_classes)
        c_conv11_2 = c_conv11_2.view(batch_size, -1, self.n_classes)  # (N, 4, n_classes)

        # Всього 8732 коробки
        # Конкатенація в цьому конкретному порядку (тобто має відповідати порядку попередніх блоків)
        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1)  # (N, 8732, 4)
        classes_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2],
                                   dim=1)  # (N, 8732, n_classes)

        return locs, classes_scores


class SSD300(nn.Module):
    """
    Мережа SSD300 - інкапсулює базову мережу VGG, допоміжні та згортки прогнозування.
    """

    def __init__(self, n_classes):
        super(SSD300, self).__init__()

        self.n_classes = n_classes

        self.base = VGGBase()
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions(n_classes)

        # Оскільки функції нижчого рівня (conv4_3_feats) мають значно більші масштаби, ми беремо норму L2 і масштабуємо
        # Коефіцієнт масштабування спочатку встановлено на 20, але вивчається для кожного каналу під час резервної підтримки
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # у conv4_3_feats є 512 каналів
        nn.init.constant_(self.rescale_factors, 20)

        # Попередні коробки
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image):
        """
        Пряме поширення.

        :param image: зображення, тензор розмірів (N, 3, 300, 300)
        :return: 8732 локації та оцінки класів (тобто без попереднього вікна) для кожного зображення
        """
        # Виконайте згортки базової мережі VGG (генератори карти функцій нижчого рівня)
        conv4_3_feats, conv7_feats = self.base(image)  # (N, 512, 38, 38), (N, 1024, 19, 19)

        # Перемасштабувати conv4_3 після норми L2
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        conv4_3_feats = conv4_3_feats / norm  # (N, 512, 38, 38)
        conv4_3_feats = conv4_3_feats * self.rescale_factors  # (N, 512, 38, 38)
        # (PyTorch autobroadcasts singleton dimensions during arithmetic)

        # Виконайте допоміжні згортки (генератори карти функцій вищого рівня)
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = \
            self.aux_convs(conv7_feats)  # (N, 512, 10, 10),  (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)

        # Виконати згортки передбачення (передбачити зміщення відносно попередніх блоків і класів у кожному отриманому полі локалізації)
        locs, classes_scores = self.pred_convs(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats,
                                               conv11_2_feats)  # (N, 8732, 4), (N, 8732, n_classes)

        return locs, classes_scores

    def create_prior_boxes(self):
        """
        Створіть попередні (за замовчуванням) поля 8732 для SSD300, як визначено в статті.

        :return: попередні блоки в координатах розміру центру, тензор розмірів (8732, 4)
        """
        fmap_dims = {'conv4_3': 38,
                     'conv7': 19,
                     'conv8_2': 10,
                     'conv9_2': 5,
                     'conv10_2': 3,
                     'conv11_2': 1}

        obj_scales = {'conv4_3': 0.1,
                      'conv7': 0.2,
                      'conv8_2': 0.375,
                      'conv9_2': 0.55,
                      'conv10_2': 0.725,
                      'conv11_2': 0.9}

        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                        # Для співвідношення сторін 1 використовуйте додатковий пріоритет, масштаб якого є середнім геометричним значенням
                        # масштаб поточної карти об'єктів і масштаб наступної карти об'єктів
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            # Для останньої карти об’єктів не існує «наступної» карти об’єктів
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
        prior_boxes.clamp_(0, 1)  # (8732, 4)

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Розшифруйте 8732 місця та рейтинги класів (вихідні дані SSD300) для виявлення об’єктів.

        Для кожного класу виконайте немаксимальне придушення (NMS) на ящиках, які перевищують мінімальне порогове значення.

        :param predicted_locs: передбачувані розташування/блоки відносно 8732 попередніх блоків, тензор розмірів (N, 8732, 4)
        :param predicted_scores: оцінки класів для кожного із закодованих місць/коробок, тензор вимірів (N, 8732, n_classes)
        :param min_score: мінімальний поріг для того, щоб ящик вважався таким, що відповідає певному класу
        :param max_overlap: максимальне перекриття, яке можуть мати два блоки, щоб той із нижчим балом не пригнічувався через NMS
        :param top_k: якщо є багато результуючих виявлень у всіх класах, збережіть лише верхній 'k'
        :return: виявлення (коробки, мітки та оцінки), списки довжини batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        # Списки для зберігання остаточних прогнозованих блоків, міток і балів для всіх зображень
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Декодуйте координати об’єктів із форми, до якої ми регресували передбачені прямокутники
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (8732, 4), these are fractional pt. coordinates

            # Списки для зберігання коробок і балів для цього зображення
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            # Перевірте для кожного класу
            for c in range(1, self.n_classes):
                # Зберігайте лише передбачені поля та бали, якщо бали для цього класу перевищують мінімальний бал
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Сортуйте передбачені коробки та бали за балами
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Знайдіть перекриття між передбаченими квадратами
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Немаксимальне придушення (NMS)

                # Тензор torch.uint8 (байт) для відстеження того, які передбачені блоки потрібно придушити
                # 1 означає придушення, 0 означає не придушувати
                #suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)
                suppress = torch.zeros((n_above_min_score)).bool().to(device)  # (n_qualified)
                # Розгляньте кожну коробку в порядку зменшення балів
                for box in range(class_decoded_locs.size(0)):
                    # Якщо це поле вже позначено для придушення
                    if suppress[box] == 1:
                        continue

                    # Придушити блоки, перекриття яких (з цим блоком) перевищують максимальне перекриття
                    # Знайдіть такі ящики та оновіть індекси придушення
                    suppress = suppress | (overlap[box] > max_overlap)
                    #suppress = torch.max(suppress, overlap[box] > max_overlap)
                    # Операція max зберігає раніше закриті поля, як операція «АБО».

                    # Не пригнічуйте цей блок, навіть якщо він має перекриття 1 із собою
                    suppress[box] = 0

                # Зберігайте лише непригнічені ящики для цього класу
                image_boxes.append(class_decoded_locs[~suppress])
                image_labels.append(
                    torch.LongTensor(
                        (~suppress).sum().item() * [c]).to(device)
                )
                image_scores.append(class_scores[~suppress])

            # Якщо жодного об’єкта в жодному класі не знайдено, збережіть заповнювач для «фону»
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Об’єднати в один тензор
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Зберігати лише k верхніх об’єктів
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Додавання до списків, які зберігають передбачені рамки та бали для всіх зображень
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size



class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1 - alpha])
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = None
        self.reduction = reduction

    def forward(self, input, target):
        # input: (N, C), target: (N,)
        logpt = F.log_softmax(input, dim=1)  # (N, C)
        pt = torch.exp(logpt)  # (N, C)

        # Вибираємо logpt для правильних класів
        logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)  # (N,)
        pt = pt.gather(1, target.unsqueeze(1)).squeeze(1)       # (N,)

        if self.alpha is not None:
            if self.alpha.device != input.device:
                self.alpha = self.alpha.to(input.device)
            at = self.alpha.gather(0, target)
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

#тут використовуються стандартні функції втрат Cross-Entropy Loss та Smooth L1 Loss
class MultiBoxLoss(nn.Module):

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
       
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 8732)

        # Для кожного зображення
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i],
                                           self.priors_xy)  # (n_objects, 8732)

            # Для кожного попереднього знайдіть об’єкт, який має максимальне перекриття
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

            
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Потім призначте кожному об’єкту відповідний пріоритет максимального перекриття. (Це виправляє 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # Щоб переконатися, що ці пріоритети кваліфікуються, штучно дайте їм перекриття більше 0,5. (Це виправляє 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Мітки для кожного пріора
            label_for_each_prior = labels[i][object_for_each_prior]  # (8732)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

            # Магазин
            true_classes[i] = label_for_each_prior

            # Закодуйте координати об’єкта центрального розміру у форму, до якої ми регресували передбачені прямокутники
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)

        # Визначте позитивні попередні (об’єкт/нефон)
        positive_priors = true_classes != 0  # (N, 8732)

        # ВТРАТА ЛОКАЛІЗАЦІЇ
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # ВТРАТА ДОВІРИ
        # Кількість позитивних і негативних пріоритетів на зображення
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # Спочатку знайдіть втрату для всіх попередніх
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

        # Ми вже знаємо, які пріоритети позитивні
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # Як у документі, усереднено лише за позитивними попередніми даними, хоча обчислюється як за позитивними, так і за жорстко негативними попередніми даними
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # ПОВНА ВТРАТА

        return conf_loss + self.alpha * loc_loss

#тут використовуються стандартні функції втрат Focal Loss та Smooth L1 Loss
class FocalSmoothL1Loss(nn.Module):
    """
    Комбінація Focal Loss для класифікації та Smooth L1 Loss для локалізації.
    """
    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1., gamma=2.0):
        super(FocalSmoothL1Loss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        self.gamma = gamma
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        self.focal_loss = FocalLoss(gamma=gamma)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)

        for i in range(batch_size):
            n_objects = boxes[i].size(0)
            overlap = find_jaccard_overlap(boxes[i], self.priors_xy)
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)
            _, prior_for_each_object = overlap.max(dim=1)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)
            overlap_for_each_prior[prior_for_each_object] = 1.
            label_for_each_prior = labels[i][object_for_each_prior]
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0
            true_classes[i] = label_for_each_prior
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)

        positive_priors = true_classes != 0
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors]).sum()
        
        conf_loss_all = self.focal_loss(predicted_scores.view(-1, n_classes), true_classes.view(-1))
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)
        conf_loss_pos = conf_loss_all[positive_priors]
        conf_loss_neg = conf_loss_all.clone()
        conf_loss_neg[positive_priors] = 0.
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)
        hard_negatives = hardness_ranks < (self.neg_pos_ratio * positive_priors.sum(dim=1).unsqueeze(1))
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / positive_priors.sum().float()

        total_loss = conf_loss + self.alpha * loc_loss
        return total_loss

#тут використовуються стандартні функції втрат Cross-Entropy Loss та GIoU Loss    
class CEGIoULoss(nn.Module):
    """
    Комбінація Cross-Entropy Loss для класифікації та GIoU Loss для локалізації.
    """
    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(CEGIoULoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def giou_loss(self, pred_boxes, true_boxes):
        """
        Обчислює GIoU Loss між передбаченими та справжніми обмежувальними рамками.
        """
        pred_boxes = cxcy_to_xy(gcxgcy_to_cxcy(pred_boxes, self.priors_cxcy))
        true_boxes = cxcy_to_xy(gcxgcy_to_cxcy(true_boxes, self.priors_cxcy))
        
        inter = find_intersection(pred_boxes, true_boxes)
        area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        area_true = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])
        union = area_pred + area_true - inter
        iou = inter / union
        
        enclose_left = torch.min(pred_boxes[:, 0], true_boxes[:, 0])
        enclose_right = torch.max(pred_boxes[:, 2], true_boxes[:, 2])
        enclose_top = torch.min(pred_boxes[:, 1], true_boxes[:, 1])
        enclose_bottom = torch.max(pred_boxes[:, 3], true_boxes[:, 3])
        enclose_area = (enclose_right - enclose_left) * (enclose_bottom - enclose_top)
        
        giou = iou - (enclose_area - union) / enclose_area
        giou_loss = 1 - giou
        return giou_loss

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)

        for i in range(batch_size):
            n_objects = boxes[i].size(0)
            overlap = find_jaccard_overlap(boxes[i], self.priors_xy)
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)
            _, prior_for_each_object = overlap.max(dim=1)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)
            overlap_for_each_prior[prior_for_each_object] = 1.
            label_for_each_prior = labels[i][object_for_each_prior]
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0
            true_classes[i] = label_for_each_prior
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)

        positive_priors = true_classes != 0
        loc_loss = self.giou_loss(predicted_locs[positive_priors], true_locs[positive_priors]).sum()
        
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)
        conf_loss_pos = conf_loss_all[positive_priors]
        conf_loss_neg = conf_loss_all.clone()
        conf_loss_neg[positive_priors] = 0.
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)
        hard_negatives = hardness_ranks < (self.neg_pos_ratio * positive_priors.sum(dim=1).unsqueeze(1))
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / positive_priors.sum().float()

        total_loss = conf_loss + self.alpha * loc_loss
        return total_loss
    
#тут використовуються стандартні функції втрат Focal Loss та CIoU Loss    
class FocalCIoULoss(nn.Module):
    """
    Комбінація Focal Loss для класифікації та CIoU Loss для локалізації.
    """
    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1., gamma=2.0):
        super(FocalCIoULoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        self.gamma = gamma
        self.focal_loss = FocalLoss(gamma=gamma)

    def ciou_loss(self, pred_boxes, true_boxes):
        """
        Обчислює CIoU Loss між передбаченими та справжніми обмежувальними рамками.
        """
        pred_boxes = cxcy_to_xy(gcxgcy_to_cxcy(pred_boxes, self.priors_cxcy))
        true_boxes = cxcy_to_xy(gcxgcy_to_cxcy(true_boxes, self.priors_cxcy))
        
        inter = find_intersection(pred_boxes, true_boxes)
        area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        area_true = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])
        union = area_pred + area_true - inter
        iou = inter / union
        
        enclose_left = torch.min(pred_boxes[:, 0], true_boxes[:, 0])
        enclose_right = torch.max(pred_boxes[:, 2], true_boxes[:, 2])
        enclose_top = torch.min(pred_boxes[:, 1], true_boxes[:, 1])
        enclose_bottom = torch.max(pred_boxes[:, 3], true_boxes[:, 3])
        enclose_width = enclose_right - enclose_left
        enclose_height = enclose_bottom - enclose_top
        
        center_pred = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
        center_true = (true_boxes[:, :2] + true_boxes[:, 2:]) / 2
        center_distance = torch.sum((center_pred - center_true) ** 2, dim=1)
        enclose_diagonal = enclose_width ** 2 + enclose_height ** 2
        
        v = (4 / (math.pi ** 2)) * torch.pow(torch.atan((true_boxes[:, 2] - true_boxes[:, 0]) / (true_boxes[:, 3] - true_boxes[:, 1])) -
                                             torch.atan((pred_boxes[:, 2] - pred_boxes[:, 0]) / (pred_boxes[:, 3] - pred_boxes[:, 1])), 2)
        alpha = v / (1 - iou + v + 1e-7)
        
        ciou = iou - (center_distance / enclose_diagonal + alpha * v)
        ciou_loss = 1 - ciou
        return ciou_loss

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)

        for i in range(batch_size):
            n_objects = boxes[i].size(0)
            overlap = find_jaccard_overlap(boxes[i], self.priors_xy)
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)
            _, prior_for_each_object = overlap.max(dim=1)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)
            overlap_for_each_prior[prior_for_each_object] = 1.
            label_for_each_prior = labels[i][object_for_each_prior]
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0
            true_classes[i] = label_for_each_prior
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)

        positive_priors = true_classes != 0
        loc_loss = self.ciou_loss(predicted_locs[positive_priors], true_locs[positive_priors]).sum()
        
        conf_loss_all = self.focal_loss(predicted_scores.view(-1, n_classes), true_classes.view(-1))
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)
        conf_loss_pos = conf_loss_all[positive_priors]
        conf_loss_neg = conf_loss_all.clone()
        conf_loss_neg[positive_priors] = 0.
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)
        hard_negatives = hardness_ranks < (self.neg_pos_ratio * positive_priors.sum(dim=1).unsqueeze(1))
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / positive_priors.sum().float()

        total_loss = conf_loss + self.alpha * loc_loss
        return total_loss