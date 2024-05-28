from pathlib import Path
from glob import glob
import json

from tqdm import tqdm
import numpy as np
import easyocr


reader = easyocr.Reader(lang_list=['en'], detector=False)

dataset_dir = Path('../../dataset/SROIE-single-line/test')
images_dir = dataset_dir / 'img'

predictions_dir = dataset_dir / 'easyocr'
predictions_dir.mkdir()

for img_path in tqdm(sorted(glob(str(images_dir / '*/*.png')))):

    result = reader.recognize(img_path)

    res = result[0][1]
    img_path = Path(img_path)
    d = predictions_dir / img_path.parts[-2]
    d.mkdir(exist_ok=True)
    with open(d / f'{img_path.stem}.json', 'w') as f:
        json.dump(res, f, indent=4)
