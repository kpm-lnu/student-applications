import json
from pathlib import Path
from glob import glob

from tqdm import tqdm
import numpy as np

from src.eval.metrics import edit_distance


def arrange_full_text_from_lines(bboxes: np.ndarray, texts: np.ndarray) -> str:

    full_text = ''

    j = 0
    while bboxes.shape[0] > 0:
        i = np.argmin(bboxes[:, 1])
        tline = (bboxes[i, 1] + bboxes[i, 3]) / 2
        line_indices = np.where(bboxes[:, 1] < tline)[0]

        s = np.argsort(bboxes[line_indices, 0])
        for c in s:
            full_text += ' ' + texts[line_indices][c]

        bboxes = np.delete(bboxes, line_indices, 0)
        texts = np.delete(texts, line_indices, 0)
        j += 1
        if j > 1000:
            print(bboxes, texts)
            break

    return full_text


mistakes = []

dataset_name = 'test'

annotations_dir = Path('../../dataset/SROIE2019') / dataset_name
predictions_dir = Path('../../dataset/SROIE-single-line-easyocr') / dataset_name

for ann_path in tqdm(sorted(glob(str(annotations_dir / 'box' / '*.txt')))):

    bboxes = []
    texts = []
    with open(ann_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            s = line.split(',')

            try:
                bbox = [int(x) for x in s[:8]]
            except:
                continue

            text = ','.join(s[8:])
            text = text[:-1]  # no \n in the end

            bboxes.append([bbox[0], bbox[1], bbox[4], bbox[5]])
            texts.append(text)

    bboxes = np.array(bboxes)
    texts = np.array(texts)
    expected_full_text = arrange_full_text_from_lines(bboxes, texts)

    bboxes = []
    texts = []
    stem = Path(ann_path).stem
    bboxes_path = predictions_dir / 'box' / stem
    texts_path = predictions_dir / 'sroie-synthetic-data-epoch-30' / stem
    for b, t in zip(sorted(glob(str(bboxes_path / '*.json'))),
                    sorted(glob(str(texts_path / '*.json')))):
        with open(b, 'r') as f:
            box = json.load(f)
            bboxes.append(box)

        with open(t, 'r') as f:
            text = json.load(f)
            text = text.upper()
            texts.append(text)

    bboxes = np.array(bboxes)
    texts = np.array(texts)
    actual_full_text = arrange_full_text_from_lines(bboxes, texts)

    relative_mistake = int(edit_distance(actual_full_text, expected_full_text)) / len(expected_full_text)
    mistakes.append(relative_mistake)

m = np.mean(mistakes)
print(m)
