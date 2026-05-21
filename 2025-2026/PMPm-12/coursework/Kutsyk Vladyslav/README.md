# UAV Detection & Tracking
## YOLOv8s-P2-SimAM + ByteTrack-IMM-KF

Система виявлення та траєкторного супроводу безпілотних
літальних апаратів як малих рухомих цілей у відеопотоці.

---

## Архітектура системи

Пайплайн складається з двох незалежних модулів:

**Модуль виявлення (YOLOv8s-P2-SimAM)**
- Базова архітектура: YOLOv8s (anchor-free, CSPDarknet backbone)
- **Модифікація 1:** Четверта детекційна голова P2 (stride=4) для
  об'єктів площею < 32×32 пікселів — основна зміна для малих цілей
- **Модифікація 2:** SimAM (Simple, Parameter-free Attention Module)
  як forward-hook на шарах neck — без нових навчуваних параметрів
- **Модифікація 3:** Функція втрат Wise-IoU v3 замість стандартної
  CIoU — динамічне зважування складних малих прикладів

**Модуль трекінгу (ByteTrack + IMM-KF)**
- Базовий алгоритм: ByteTrack із двопрохідною асоціацією
- **Модифікація:** стандартний CV-фільтр Калмана замінено на
  Interacting Multiple Model (IMM-KF) із трьома паралельними
  моделями руху: CV (стала швидкість), CA (стале прискорення),
  CT (координований розворот)
- Функція вартості асоціації: комбінація відстані Махаланобіса
  та IoU з адаптивними воротами

---

## Структура файлів

```
uav_project/
│
├── config.py            # усі гіперпараметри і шляхи
│
├── simam.py             # SimAM attention module (nn.Module)
├── wiseiou.py           # Wise-IoU v3 + кастомний WiseIoUTrainer
├── yolov8s_p2.yaml      # YAML архітектури з 4 detection heads
├── model_builder.py     # збирає модель + SimAM hooks
│
├── imm_kalman.py        # IMM-KF: CV / CA / CT моделі
├── tracker.py           # ByteTrack + IMM-KF (proposed)
├── tracker_baseline.py  # ByteTrack + CV-KF (baseline B1)
│
├── detector.py          # обгортка детектора для інференсу
├── dataset_utils.py     # читання DUT Anti-UAV (detection + tracking)
│
├── train.py             # навчання: одна конфігурація або всі
├── evaluate.py          # метрики детекції та трекінгу
├── ablation.py          # порівняльний аналіз всіх конфігурацій
├── pipeline.py          # демонстрація на відео або папці кадрів
│
└── requirements.txt
```

---

## Встановлення

```bash
pip install -r requirements.txt
```

Потрібен Python 3.11, PyTorch 2.2+, CUDA 11.8+.

---

## Структура датасету DUT Anti-UAV

Датасет складається з двох незалежних частин.

**Detection split** (для навчання детектора):
```
data/DUT-Anti-UAV/
  detection/
    train/
      images/  *.jpg          # 5 200 зображень
      labels/  *.txt          # YOLO format: cls cx cy w h (normalized)
    val/
      images/  val/labels/    # 2 600 зображень
    test/
      images/  test/labels/   # 2 200 зображень
```

**Tracking split** (для оцінки трекера):
```
  tracking/
    train/  val/  test/
      <seq_name>/
        imgs/  *.jpg                  # кадри послідовності
        groundtruth_rect.txt          # анотації треку
```

Формат `groundtruth_rect.txt` — один рядок на кадр:
```
x_topleft  y_topleft  width  height
929 559 243 142
932 556 243 147
...
```
Якщо ціль невидима у кадрі — рядок містить `0 0 0 0`.

Перед навчанням `data.yaml` генерується автоматично.
Якщо потрібно згенерувати вручну:
```python
from dataset_utils import make_data_yaml
from pathlib import Path
make_data_yaml(Path("data/DUT-Anti-UAV"), Path("data/DUT-Anti-UAV/data.yaml"))
```

---

## Конфігурації (ablation study)

| ID | Детектор | Трекер | Опис |
|----|----------|--------|------|
| B1 | YOLOv8s стандартний | CV-KF | Повний baseline |
| B2 | YOLOv8s + P2 | CV-KF | + голова P2 |
| B3 | YOLOv8s + P2 + SimAM | CV-KF | + SimAM attention |
| B4 | YOLOv8s + P2 + SimAM + WIoU | CV-KF | + Wise-IoU v3 |
| **P** | **те саме що B4** | **IMM-KF** | **повна система** |

B4 і P мають однаковий детектор — різниця лише у трекері.
Тому для повного порівняння достатньо натренувати B1 і B4.

---

## Навчання

### Одна конфігурація
```bash
# Baseline (B1) — стандартний YOLOv8s
python train.py --config b1

# Запропонована система (B4/P) — з усіма модифікаціями
python train.py --config b4

# Лише Етап I (без дофайнтюнінгу, швидше)
python train.py --config b4 --stage 1
```

### Всі конфігурації послідовно
```bash
python train.py --config all
```
Це запускає B1 → B2 → B3 → B4 по черзі (кожна ~3–4 год).

### Два етапи навчання

**Етап I** (80 епох, AdamW, backbone не заморожений):
- Повне навчання на DUT Anti-UAV detection split
- Агресивна аугментація: Mosaic, motion blur, MixUp
- Ініціалізація backbone з `yolov8s.pt` (pretrained на COCO)
- Зупинка за early stopping (patience=15 епох)

**Етап II** (20 епох, SGD, backbone заморожений):
- Дофайнтюнінг: навчаються лише neck та detection heads
- Менший LR (1e-4), легша аугментація
- Завантажує ваги з Етапу I

Після навчання ваги зберігаються у `checkpoints/<config>_best.pt`.

---

## Оцінювання

```bash
# Детекція + трекінг разом
python evaluate.py --weights checkpoints/b4_best.pt

# Лише детекція (mAP, AP_S, Recall)
python evaluate.py --weights checkpoints/b4_best.pt --task detect

# Лише трекінг (MOTA, IDF1, HOTA, IDSW)
python evaluate.py --weights checkpoints/b4_best.pt --task track

# Вимір FPS на синтетичних кадрах
python evaluate.py --weights checkpoints/b4_best.pt --task fps
```

Результати та графіки зберігаються у `results/`.
Графік динаміки IMM-ймовірностей (μ_CV, μ_CA, μ_CT) будується
автоматично для кожної тестової послідовності.

---

## Ablation study (порівняльний аналіз)

### Варіант 1 — порівняти лише трекери (швидко)
Використовує одні ваги з двома різними трекерами:
```bash
python ablation.py \
  --weights_p checkpoints/b4_best.pt \
  --tracker_only
```

### Варіант 2 — повне ablation (B1 → B2 → B3 → B4 → P)
```bash
python ablation.py \
  --weights_b1 checkpoints/b1_best.pt \
  --weights_b2 checkpoints/b2_best.pt \
  --weights_b3 checkpoints/b3_best.pt \
  --weights_b4 checkpoints/b4_best.pt \
  --weights_p  checkpoints/b4_best.pt
```

Скрипт автоматично генерує у `results/ablation/plots/`:
- `ablation_mota.png` — bar chart по MOTA
- `ablation_idf1.png` — bar chart по IDF1
- `ablation_hota.png` — bar chart по HOTA
- `ablation_idsw.png` — bar chart по IDSW (менше = краще)
- `imm_probs_<seq>.png` — динаміка μ_CV / μ_CA / μ_CT для конфіг. P
- `ablation_results.csv` — готова таблиця для вставки у LaTeX
- `ablation_results.json` — всі числа у JSON

---

## Демонстрація пайплайну

```bash
# На папці з кадрами (tracking test sequence)
python pipeline.py \
  --weights checkpoints/b4_best.pt \
  --source  data/DUT-Anti-UAV/tracking/test/seq1/imgs \
  --show --save_video --show_imm

# На відеофайлі
python pipeline.py \
  --weights checkpoints/b4_best.pt \
  --source  video.mp4 \
  --show --save_video

# На вебкамері (ID камери = 0)
python pipeline.py \
  --weights checkpoints/b4_best.pt \
  --source 0 --show
```

Прапори:
- `--show` — виводити вікно OpenCV у реальному часі
- `--save_video` — зберегти `results/output.mp4`
- `--show_imm` — зберегти графік ймовірностей IMM моделей

На кожному кадрі відображається:
- Сірі рамки — сирі детекції (до трекінгу)
- Кольорові рамки з ID — підтверджені треки
- Поточна домінуюча модель IMM (`[CV 0.81]`, `[CT 0.64]` тощо)
- FPS та кількість активних треків

---

## Рекомендований порядок запуску

```bash
# 1. Встановити залежності
pip install -r requirements.txt

# 2. Перевірити датасет
python -c "
from dataset_utils import load_tracking_split
from pathlib import Path
seqs = load_tracking_split(Path('data/DUT-Anti-UAV/tracking/train'))
print(f'Послідовностей: {len(seqs)}')
print(f'Перша: {seqs[0].name}, кадрів: {len(seqs[0])}')
print(f'Перший bbox: {seqs[0].gts[0]}')
"

# 3. Натренувати baseline і proposed
python train.py --config b1 --stage 1
python train.py --config b4 --stage 1

# 4. Порівняти детектори
python ablation.py \
  --weights_b1 checkpoints/b1_best.pt \
  --weights_b4 checkpoints/b4_best.pt \  
  --weights_p  checkpoints/b4_best.pt

# 5. Порівняти трекери (на вагах b4)
python ablation.py \
  --weights_p checkpoints/b4_best.pt \
  --tracker_only

# 6. Демонстрація на тестовій послідовності
python pipeline.py \
  --weights checkpoints/b4_best.pt \
  --source  data/DUT-Anti-UAV/tracking/test/seq1/imgs \
  --show --show_imm
```
