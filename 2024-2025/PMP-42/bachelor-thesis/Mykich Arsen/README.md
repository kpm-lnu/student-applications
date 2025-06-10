# SEIRV Model Visualization & Analysis

## Опис

Цей пакет містить Python-скрипти для моделювання епідемічного процесу з урахуванням вакцинації (SEIRV-модель), а також для аналізу чутливості параметрів. Скрипти генерують графіки та таблиці, які зберігаються у відповідних папках.

## Вимоги

- Python 3.8+
- Встановлені бібліотеки:
  - numpy
  - matplotlib
  - pandas
  - scipy
  - seaborn

Встановити залежності можна командою:
```bash
pip install numpy matplotlib pandas scipy seaborn
```

## Який скрипт генерує яке зображення

### seirv_parameter_comparison.py
- `seirv_base_scenario.png` — базовий сценарій SEIRV
- `seirv_v0_comparison.png` — вплив початкового рівня вакцинації
- `seirv_efficacy_comparison.png` — вплив ефективності вакцини
- `seirv_nu_comparison.png` — вплив швидкості вакцинації
- `seirv_Re_vs_vaccination.png` — залежність ефективного репродуктивного числа від рівня вакцинації

### seirv_model.py
- `seirv_model.png` — базова динаміка SEIRV (один сценарій)

### phase_portrait.py
- `seirv_phase_portrait.png` — фазовий портрет SEIRV
- `seir_phase_portrait.png` — фазовий портрет SEIR
- `sir_phase_portrait.png` — фазовий портрет SIR

### sensitivity_analysis.py
- `sensitivity_coefficients.png` — аналіз чутливості параметрів

## Як отримати візуалізації та таблиці

### 1. Основні графіки SEIRV-моделі та порівняння параметрів

**Генеруються скриптом:**  
`seirv_parameter_comparison.py`

**Щоб згенерувати ці графіки, запустіть:**
```bash
python seirv_parameter_comparison.py
```
> Всі графіки будуть збережені у папці `figures/`. Таблиці з результатами виводяться у консоль.

### 2. Базова динаміка SEIRV

**Генерується скриптом:**  
`seirv_model.py`

**Щоб згенерувати графік, запустіть:**
```bash
python seirv_model.py
```
> Графік буде збережено у поточній папці як `seirv_model.png`.

### 3. Фазові портрети

**Генеруються скриптом:**  
`phase_portrait.py`

**Щоб згенерувати всі фазові портрети, запустіть:**
```bash
python phase_portrait.py
```
> Будуть збережені файли: `seirv_phase_portrait.png`, `seir_phase_portrait.png`, `sir_phase_portrait.png` у поточній папці.

### 4. Аналіз чутливості параметрів

**Генерується скриптом:**  
`sensitivity_analysis.py`

**Щоб отримати ці результати, запустіть:**
```bash
python sensitivity_analysis.py
```
> Графік буде збережено як `sensitivity_coefficients.png`, а таблиця коефіцієнтів чутливості зʼявиться у консолі.

---

## Додатково

- Всі результати (графіки) зберігаються у поточній директорії.


---

**Зауваження:**  
Якщо ви перейменували або перемістили скрипти, змініть відповідні команди запуску.
