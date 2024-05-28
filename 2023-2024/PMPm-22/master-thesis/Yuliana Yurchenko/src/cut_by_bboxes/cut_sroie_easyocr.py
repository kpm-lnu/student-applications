import json
from pathlib import Path
from glob import glob

import numpy as np
from tqdm import tqdm
import cv2
import easyocr


dataset_name = 'test'
input_dataset_dir = Path('../../dataset/SROIE2019') / dataset_name

output_dataset_dir = Path('../../dataset/SROIE-single-line-easyocr-en-only') / dataset_name
save_images_dir = output_dataset_dir / 'img'
save_text_lines_dir = output_dataset_dir / 'text'
save_bboxes_dir = output_dataset_dir / 'box'

output_dataset_dir.mkdir()
save_images_dir.mkdir()
save_text_lines_dir.mkdir()
save_bboxes_dir.mkdir()

images_dir = input_dataset_dir / 'img'

reader = easyocr.Reader(lang_list=['en'])

for img_path in tqdm(sorted(glob(str(images_dir / '*.jpg')))):

    img = cv2.imread(img_path)
    result = reader.readtext(img_path)

    for i, t in enumerate(result):
        bbox = np.array(t[0][0] + t[0][2], dtype=int)
        text = t[1]
        text_line_img = img[bbox[1]: bbox[3], bbox[0]: bbox[2]]

        try:
            d = save_images_dir / Path(img_path).stem
            d.mkdir(exist_ok=True)
            cv2.imwrite(str(d / f'{i}.png'), text_line_img)

            d = save_text_lines_dir / Path(img_path).stem
            d.mkdir(exist_ok=True)
            with open(d / f'{i}.json', 'w') as ff:
                json.dump(text, ff, indent=4)

            d = save_bboxes_dir / Path(img_path).stem
            d.mkdir(exist_ok=True)
            with open(d / f'{i}.json', 'w') as ff:
                json.dump(bbox.tolist(), ff, indent=4)
        except:
            print(Path(img_path).stem, i)
