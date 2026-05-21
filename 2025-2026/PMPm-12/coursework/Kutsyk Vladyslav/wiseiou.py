"""
wiseiou.py
Wise-IoU v3 — loss function з динамічним фокусуючим зважуванням.
Tong et al., arXiv:2301.10051, 2023.

Інтегрується через підклас DetectionTrainer Ultralytics.
"""

import torch
import torch.nn as nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.metrics import bbox_iou
from config import WISEIOU_BETA, WISEIOU_DELTA


# ── Wise-IoU v3 обчислення ────────────────────────────────────

class WiseIoUBboxLoss(BboxLoss):
    """
    Замінює стандартний CIoU на Wise-IoU v3 у bbox-гілці loss.
    Наслідує BboxLoss для сумісності з Ultralytics pipeline.
    """

    def __init__(self, reg_max: int = 16, use_dfl: bool = True,
                 beta: float = WISEIOU_BETA,
                 delta: float = WISEIOU_DELTA):
        super().__init__(reg_max, use_dfl)
        self.beta  = beta
        self.delta = delta
        # Ковзне середнє IoU-похибки (оновлюється per-batch)
        self.register_buffer("u_bar",
                             torch.tensor(0.5, dtype=torch.float32))

    def iou_loss(self, pred_bboxes: torch.Tensor,
                 target_bboxes: torch.Tensor) -> torch.Tensor:
        """
        Wise-IoU v3 для однієї пари (pred, target) bounding boxes.
        Формат: xyxy, normalized.
        """
        # Базове IoU (використовуємо вбудований обчислювач)
        iou = bbox_iou(pred_bboxes, target_bboxes,
                       xywh=False, CIoU=False).squeeze(-1)
        iou = iou.clamp(0.0, 1.0)

        # Фокус-змінна u = 1 - IoU
        u = 1.0 - iou

        # Оновлення ковзного середнього (детач, не через граф)
        with torch.no_grad():
            u_mean = u.detach().mean()
            self.u_bar = 0.9 * self.u_bar + 0.1 * u_mean

        # Фокусуючий ваговий коефіцієнт r
        eps = 1e-7
        r = (self.beta * self.u_bar) / (
            self.delta * u + self.u_bar + eps
        )

        # Фінальний loss
        loss = r * u
        return loss.mean()

    def forward(self, pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask):
        """
        Повністю перевизначає forward() батьківського BboxLoss,
        замінюючи лише частину IoU-втрати.
        """
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        # Wise-IoU замість CIoU
        iou_loss = self.iou_loss(pred_bboxes[fg_mask],
                                 target_bboxes[fg_mask])

        # DFL лишається стандартним із батьківського класу
        if self.use_dfl:
            target_ltrb = self.bbox2dist(anchor_points,
                                         target_bboxes, self.reg_max - 1)
            dfl_loss = self._df_loss(pred_dist[fg_mask].view(-1,
                                     self.reg_max), target_ltrb[fg_mask])
        else:
            dfl_loss = torch.zeros(1, device=pred_dist.device)

        return iou_loss, dfl_loss


# ── Кастомний тренер ──────────────────────────────────────────

class WiseIoUTrainer(DetectionTrainer):
    """
    Підклас DetectionTrainer з Wise-IoU v3 в bbox loss.
    Використання:
        trainer = WiseIoUTrainer(overrides={...})
        trainer.train()
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        # Замінити BboxLoss на WiseIoUBboxLoss
        if hasattr(model, "criterion") and hasattr(
                model.criterion, "bbox_loss"):
            reg_max = model.criterion.bbox_loss.reg_max
            use_dfl = model.criterion.bbox_loss.use_dfl
            model.criterion.bbox_loss = WiseIoUBboxLoss(
                reg_max=reg_max, use_dfl=use_dfl
            ).to(next(model.parameters()).device)
            if verbose:
                print("[WiseIoU] BboxLoss замінено на WiseIoUBboxLoss v3")
        return model
