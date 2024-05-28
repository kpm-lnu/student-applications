from pathlib import Path
from glob import glob
import json

from tqdm import tqdm
import cv2

from tensorflow import keras
from tensorflow.keras.models import load_model

from src.ctc_layer import CTCLayer
from src.predict_sroie_single_line.preprocessing_tools import preprocess_image, predict_text

model = load_model('../../models/sroie-single-line-both-10 (1).h5', custom_objects={'CTCLayer': CTCLayer})
keras_model = keras.models.Model(model.get_layer(name="image").input, model.get_layer(name="dense2").output)

dataset_dir = Path('../../dataset/SROIE-single-line/test')
images_dir = dataset_dir / 'img'

predictions_dir = dataset_dir / 'sroie_both10_upd_model_pred'
predictions_dir.mkdir()

for p in tqdm(glob(str(images_dir / '*/*.png'))):
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    text = predict_text(preprocess_image(img), keras_model)

    p = Path(p)
    d = predictions_dir / p.parts[-2]
    d.mkdir(exist_ok=True)
    with open(d / f'{p.stem}.json', 'w') as f:
        json.dump(text, f, indent=4)
