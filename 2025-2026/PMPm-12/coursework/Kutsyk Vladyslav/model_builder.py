"""
model_builder.py
Будує модифіковану модель YOLOv8s-P2-SimAM.

Кроки:
  1. Реєструє SimAM у просторі імен Ultralytics.
  2. Завантажує архітектуру з yolov8s_p2.yaml.
  3. Вставляє SimAM як forward-hooks на шари 15, 18, 21, 24
     (neck C2f виходи) — без зміни YAML і без нових ваг.
"""

from pathlib import Path
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import ultralytics.nn.modules as nnm

from simam import SimAM
from config import MODEL_YAML, PRETRAINED, SIMAM_LAMBDA


# ── 1. Реєстрація SimAM в Ultralytics ─────────────────────────
nnm.SimAM = SimAM


# ── 2. Індекси шарів neck, куди вставляємо SimAM hooks ────────
# Відповідають коментарям у yolov8s_p2.yaml:
# 15 = neck-P3, 18 = neck-P2, 21 = final-P3, 24 = final-P4
SIMAM_HOOK_LAYERS = {15, 18, 21, 24}


def _make_simam_hook(simam_module: SimAM):
    """Повертає forward-hook, що застосовує SimAM до виходу шару."""
    def hook(module, input, output):
        return simam_module(output)
    return hook


def _simam_forward_hook(module, input, output):
    """Top-level forward hook: applies registered `simam_module` if present.

    Placing this at module scope makes it picklable for `torch.save`.
    """
    sim = getattr(module, "simam_module", None)
    return sim(output) if sim is not None else output


def build_model(
    yaml_path: Path = MODEL_YAML,
    pretrained: str  = PRETRAINED,
    nc: int          = 1,
    freeze_backbone: bool = False,
) -> YOLO:
    """
    Повертає YOLO-обгортку з:
      - архітектурою yolov8s_p2.yaml (4 detection heads)
      - SimAM hooks на neck-шарах
      - (опційно) замороженим backbone для 2-го етапу навчання

    Args:
        yaml_path:        шлях до YAML конфігурації
        pretrained:       шлях до .pt ваг для трансферного навчання
        nc:               кількість класів
        freeze_backbone:  True = заморозити backbone (Етап II)
    """
    # Завантажуємо модель із YAML; якщо pretrained вказано —
    # переносимо ваги backbone (ignore mismatch for new heads)
    model = YOLO(str(yaml_path))

    if pretrained:
        base = YOLO(pretrained)
        # Скопіювати backbone ваги (шари 0–9)
        _transfer_backbone_weights(model.model, base.model)
        del base

    # Змінити кількість класів якщо потрібно
    if nc != model.model.nc:
        model.model.nc = nc

    # ── 3. Вставка SimAM hooks ────────────────────────────────
    _attach_simam_hooks(model.model)

    # ── 4. Заморожування backbone (Етап II) ───────────────────
    if freeze_backbone:
        _freeze_backbone(model.model)
        print("[ModelBuilder] Backbone заморожено (Етап II).")

    total_params = sum(p.numel() for p in model.model.parameters())
    trainable    = sum(p.numel() for p in model.model.parameters()
                       if p.requires_grad)
    print(f"[ModelBuilder] Параметри: {total_params/1e6:.2f}M total, "
          f"{trainable/1e6:.2f}M trainable")

    return model


# ── Допоміжні функції ─────────────────────────────────────────

def _attach_simam_hooks(det_model: nn.Module) -> None:
    """Прикріплює SimAM як forward-hook до обраних шарів neck."""
    hooked = 0
    for i, layer in enumerate(det_model.model):
        if i in SIMAM_HOOK_LAYERS:
            simam_mod = SimAM(e_lambda=SIMAM_LAMBDA)
            # register SimAM as a submodule of the layer so its
            # parameters/state_dict are tracked and picklable
            layer.add_module("simam_module", simam_mod)
            # register the forward hook (module-level function)
            layer.register_forward_hook(_simam_forward_hook)
            hooked += 1

    print(f"[ModelBuilder] SimAM hooks встановлено на {hooked} шарів: "
          f"{sorted(SIMAM_HOOK_LAYERS)}")


def _freeze_backbone(det_model: nn.Module) -> None:
    """Заморожує перші 10 шарів (backbone, індекси 0–9)."""
    for i, layer in enumerate(det_model.model):
        if i < 10:
            for p in layer.parameters():
                p.requires_grad = False


def _transfer_backbone_weights(
    target: nn.Module, source: nn.Module
) -> None:
    """
    Копіює ваги backbone (шари 0–9) з source у target.
    Ігнорує невідповідності розмірів (нові шари залишаються random init).
    """
    target_state = target.state_dict()
    source_state = source.state_dict()
    transferred  = 0

    for key, val in source_state.items():
        # Визначаємо індекс шару за ключем виду "model.X...."
        parts = key.split(".")
        if len(parts) < 2:
            continue
        try:
            layer_idx = int(parts[1])
        except ValueError:
            continue

        if layer_idx >= 10:          # пропускаємо neck та head
            continue
        if key in target_state and target_state[key].shape == val.shape:
            target_state[key] = val
            transferred += 1

    target.load_state_dict(target_state)
    print(f"[ModelBuilder] Перенесено {transferred} backbone-тензорів.")
