from pathlib import Path
from glob import glob
import json

import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from functools import reduce

from src.image_ops import *


np.random.seed(0)
output_dir = '../../dataset/SROIE-single-line/train/syn_img/'

for p in tqdm(glob('../../dataset/SROIE-single-line/train/text/*/*.json')):

    with open(p, 'r') as f:
        gt_text = json.load(f)

    for i in range(5):
        font_src = np.random.choice(['fonts/arial.ttf',
                                     'fonts/MerchantCopy-GOXq.ttf',
                                     'fonts/times.ttf',
                                     'fonts/timesi.ttf',
                                     'fonts/Epson1.ttf'],
                                    p=[0.3, 0.3, 0.05, 0.05, 0.3])
        if font_src == 'fonts/arial.ttf':
            font_size = 30
        elif font_src == 'fonts/MerchantCopy-GOXq.ttf':
            font_size = 45
        elif font_src == 'fonts/times.ttf' or font_src == 'fonts/timesi.ttf':
            font_size = 30
        else:
            font_size = 50

        font = ImageFont.truetype(font=font_src, size=font_size, encoding='unicode')

        # Randomly generate uppercase or lowercase letters
        uppercase = bool(np.random.randint(0, 2))
        if uppercase:
            text = gt_text
        else:
            s = gt_text.lower().split(' ')
            c = []
            for x in s:
                if len(x) > 0:
                    c.append(x[0].upper() + x[1:])
                else:
                    c.append(x)
            text = ' '.join(c)

        word_width, word_height = font.getsize(text)

        img = Image.new(mode='RGB', size=(word_width + 10, int(word_height * 1.2)), color=(256, 256, 256))
        draw = ImageDraw.Draw(img)
        color = np.random.randint(0, 100)
        draw.text(xy=(5, int(word_height * 0.2) if font_src == 'fonts/Epson1.ttf' else int(word_height * 0.05)),
                  text=text, fill=(color, color, color), font=font)

        img = np.array(img).astype(float)
        if font_src == 'fonts/Epson1.ttf':
            img = MinFilterOperation(2)(img)

        text_img = img.copy()
        angle = np.random.normal(loc=0., scale=1.)

        gaussian_noise_std = np.random.choice([0, np.random.uniform(0., 32.)], p=[0.5, 0.5])
        speckle_std = np.random.choice([0, np.random.uniform(0., 0.7)], p=[0.5, 0.5])
        salt_amount = np.random.choice([0, np.random.uniform(0, 0.5)], p=[0.5, 0.5])
        gaussian_blur_radius = np.random.choice([0, 1], p=[0.75, 0.25])
        box_blur_radius = np.random.choice([0, 1], p=[0.9, 0.1])
        min_filter_radius = np.random.choice([1, 2, 3], p=[0.5, 0.4, 0.1])
        max_filter_radius = 1 if min_filter_radius == 1 and salt_amount > 0.1 else np.random.choice([1, 2], p=[0.9, 0.1])
        pepper_amount = np.random.choice([0., 0.005, 0.01], p=[0.8, 0.1, 0.1])
        shift = np.random.choice([0., np.random.normal(loc=0., scale=30.)], p=[0.5, 0.5])

        degr_pipeline = \
            [
                RotateOperation(angle=np.radians(angle), center=(img.shape[1] // 2, img.shape[0] // 2)),
                SpeckleOperation(mean=0, stddev=speckle_std),
                GaussianNoiseOperation(mean=0, stddev=gaussian_noise_std),
                SaltPepperOperation(salt_vs_pepper=1., amount=salt_amount),
                SaltPepperOperation(salt_vs_pepper=0., amount=pepper_amount),
                MinFilterOperation(radius=min_filter_radius),
                MaxFilterOperation(radius=max_filter_radius),
                GaussianBlurOperation(radius=gaussian_blur_radius),
                BoxBlurOperation(radius=box_blur_radius),
                ShiftOperation(shift=shift),
                ResizeOperation(width=int(img.shape[1] / img.shape[0] * 32), height=32, interpolation=cv2.INTER_CUBIC)
            ]

        degraded = reduce(lambda image, op: op(image), degr_pipeline, text_img)
        degraded = np.clip(degraded, 0, 255)
        degraded = np.uint8(degraded)

        p = Path(p)
        Path(output_dir + p.parts[-2]).mkdir(exist_ok=True)
        Image.fromarray(degraded).save(output_dir + p.parts[-2] + '/' + p.stem + f'_{i}.jpg')
