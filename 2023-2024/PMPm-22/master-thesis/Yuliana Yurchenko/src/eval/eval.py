from pathlib import Path
from glob import glob
import json

from tqdm import tqdm
import numpy as np

from src.eval.metrics import edit_distance


gt_text_path = Path('../../dataset/SROIE-single-line/test/text')
pred_text_path = Path('../../dataset/SROIE-single-line/test/sroie_model_pred')


gt_texts = []
pred_texts = []
mistakes = []

for p in tqdm(glob('../../dataset/SROIE-single-line/test/text/*/*.json')):
    p = Path(p)
    subfolder = p.parts[-2]
    name = p.name

    with open(gt_text_path / subfolder / name, 'r') as f:
        gt_text = json.load(f)

    with open(pred_text_path / subfolder / name, 'r') as f:
        pred_text = json.load(f)

    gt_texts.append(gt_text)
    pred_texts.append(pred_text)

    relative_mistake = int(edit_distance(pred_text, gt_text)) / len(gt_text)
    mistakes.append(relative_mistake)

m = np.mean(mistakes)
print(m)
