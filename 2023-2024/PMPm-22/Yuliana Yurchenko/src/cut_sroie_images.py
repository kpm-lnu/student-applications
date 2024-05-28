import json
from pathlib import Path
from glob import glob

import numpy as np
from tqdm import tqdm
import cv2


dataset_name = 'test'
input_dataset_dir = Path('../dataset/SROIE2019') / dataset_name

output_dataset_dir = Path('../dataset/SROIE-single-line') / dataset_name
save_images_dir = output_dataset_dir / 'img'
save_text_lines_dir = output_dataset_dir / 'text'

output_dataset_dir.mkdir()
save_images_dir.mkdir()
save_text_lines_dir.mkdir()

annotations_dir = input_dataset_dir / 'box'
images_dir = input_dataset_dir / 'img'

for ann_path, img_path in tqdm(zip(sorted(glob(str(annotations_dir / '*.txt'))),
                                   sorted(glob(str(images_dir / '*.jpg'))))):

    assert Path(ann_path).stem == Path(img_path).stem
    img = cv2.imread(img_path)

    with open(ann_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            s = line.split(',')

            try:
                bbox = [int(x) for x in s[:8]]
            except:  # test/X51006328967
                continue

            text = ','.join(s[8:])
            text = text[:-1]  # no \n in the end

            if not text == text.upper():
                print('%', Path(ann_path).stem, text)  # train/X51006466055, train/X51008142068, test/X51005447844

            text_line_img = img[bbox[1]: bbox[5], bbox[0]: bbox[4]]

            if np.any(np.asarray(text_line_img.shape) == 0):
                print('!', Path(ann_path).stem, bbox, text_line_img.shape)  # test/X51006008092
                continue

            if bbox[0] != bbox[6] or bbox[1] != bbox[3] or bbox[2] != bbox[4] or bbox[5] != bbox[7]:
                print('#', Path(ann_path).stem, bbox)  # Never
                continue

            d = save_images_dir / Path(ann_path).stem
            d.mkdir(exist_ok=True)
            cv2.imwrite(str(d / f'{i}.png'), text_line_img)

            d = save_text_lines_dir / Path(ann_path).stem
            d.mkdir(exist_ok=True)
            with open(d / f'{i}.json', 'w') as ff:
                json.dump(text, ff, indent=4)
