import math, torch
from torch.nn import functional as F

class Balanced_Sampler():
    # Витягнує якомога більшу вибірку batch_size_per_image із заданим додатним дробом.
    def __init__(self, batch_size_per_image, positive_fraction):
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
    # Приклади позитивних і негативних прикладів з етикеток.
    def __call__(self, labels):
        sampled_positive_masks = []
        sampled_negative_masks = []
        for labels_per_image in labels:
            # Знайти індекси позитивних та негативних міток.
            positive_idx = torch.where(labels_per_image >= 1)[0]
            negative_idx = torch.where(labels_per_image == 0)[0]
            # Визначте кількість позитивних і негативних зразків.
            num_positive = int(self.batch_size_per_image * self.positive_fraction)
            num_positive = min(positive_idx.numel(), num_positive)
            
            num_negative = self.batch_size_per_image - num_positive
            num_negative = min(negative_idx.numel(), num_negative)
            # Випадкова вибірка позитивних і негативних індексів.
            sampled_positive = torch.randperm(positive_idx.numel(), device=positive_idx.device)[:num_positive]
            sampled_negative = torch.randperm(negative_idx.numel(), device=negative_idx.device)[:num_negative]
            
            sampled_positive_idx = positive_idx[sampled_positive]
            sampled_negative_idx = negative_idx[sampled_negative]
            # Створіть маски для вибраних позитивних і негативних прикладів
            sampled_positive_mask = torch.zeros_like(labels_per_image, device=labels_per_image.device).bool()
            sampled_negative_mask = torch.zeros_like(labels_per_image, device=labels_per_image.device).bool()
            
            sampled_positive_mask[sampled_positive_idx] = True
            sampled_negative_mask[sampled_negative_idx] = True
            
            sampled_positive_masks.append(sampled_positive_mask)
            sampled_negative_masks.append(sampled_negative_mask)

        return torch.stack(sampled_positive_masks, dim=0).bool(), torch.stack(sampled_negative_masks, dim=0).bool()

class BoxCoder():
    def __init__(self, weights=(1., 1., 1., 1.), bbox_clip=math.log(1000. / 16)):
        self.weights = weights
        self.bbox_clip = bbox_clip
    # Декодує дельти обмежувальних рамок, щоб створити пропозиції.    
    def decode(self, bbox_deltas, proposals):
        widths = proposals[:, :, 2] - proposals[:, :, 0]
        heights = proposals[:, :, 3] - proposals[:, :, 1]
        cx = (proposals[:, :, 0] + proposals[:, :, 2]) / 2
        cy = (proposals[:, :, 1] + proposals[:, :, 3]) / 2
        
        wx, wy, ww, wh = self.weights
        dx = bbox_deltas[:, :, 0] / wx
        dy = bbox_deltas[:, :, 1] / wy
        dw = bbox_deltas[:, :, 2] / ww
        dh = bbox_deltas[:, :, 3] / wh

        dw = torch.clamp(dw, max=self.bbox_clip)
        dh = torch.clamp(dh, max=self.bbox_clip)

        pred_cx = cx + dx * widths
        pred_cy = cy + dy * heights
        pred_w = widths * torch.exp(dw)
        pred_h = heights * torch.exp(dh)

        pred_x1 = pred_cx - pred_w / 2
        pred_y1 = pred_cy - pred_h / 2
        pred_x2 = pred_cx + pred_w / 2
        pred_y2 = pred_cy + pred_h / 2
        pred_bboxs = torch.stack((pred_x1, pred_y1, pred_x2, pred_y2), dim=2)
        return pred_bboxs

    # Кодує пропозиції та відповідні базові обмежувальні рамки, щоб створити дельти обмежувальних рамок.
    def encode(self, matched_gt_bboxs, proposals):
        wx, wy, ww, wh = self.weights
        
        proposals_x1 = proposals[:, :, 0]
        proposals_y1 = proposals[:, :, 1]
        proposals_x2 = proposals[:, :, 2]
        proposals_y2 = proposals[:, :, 3]
        
        matched_gt_bboxs_x1 = matched_gt_bboxs[:, :, 0]
        matched_gt_bboxs_y1 = matched_gt_bboxs[:, :, 1]
        matched_gt_bboxs_x2 = matched_gt_bboxs[:, :, 2]
        matched_gt_bboxs_y2 = matched_gt_bboxs[:, :, 3]
        
        proposals_widths = proposals_x2 - proposals_x1
        proposals_heights = proposals_y2 - proposals_y1
        proposals_cx = (proposals_x1 + proposals_x2) / 2
        proposals_cy = (proposals_y1 + proposals_y2) / 2

        matched_gt_bboxs_widths = matched_gt_bboxs_x2 - matched_gt_bboxs_x1
        matched_gt_bboxs_heights = matched_gt_bboxs_y2 - matched_gt_bboxs_y1
        matched_gt_bboxs_cx = (matched_gt_bboxs_x1 + matched_gt_bboxs_x2) / 2
        matched_gt_bboxs_cy = (matched_gt_bboxs_y1 + matched_gt_bboxs_y2) / 2

        targets_dx = wx * (matched_gt_bboxs_cx - proposals_cx) / proposals_widths
        targets_dy = wy * (matched_gt_bboxs_cy - proposals_cy) / proposals_heights
        targets_dw = ww * torch.log(matched_gt_bboxs_widths / proposals_widths)
        targets_dh = wh * torch.log(matched_gt_bboxs_heights / proposals_heights)

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=2)
        return targets

class Matcher(object):
    # Зіставляє пропозиції з блоками базової істини, використовуючи матрицю якості (IoU-матрицю).
    def __init__(self, iou_positive_thresh, iou_negative_high, iou_negative_low, low_quality_match):
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2

        self.iou_positive_thresh = iou_positive_thresh
        self.iou_negative_high = iou_negative_high
        self.iou_negative_low = iou_negative_low
        self.low_quality_match = low_quality_match

    # Зіставляє пропозиції зі скриньками базової істини.
    def __call__(self, match_quality_matrix):
        proposals_max_iou_val, proposals_max_iou_idx = match_quality_matrix.max(dim=0)
        proposals_match = proposals_max_iou_idx.clone()
        
        # Mark proposals as negative
        negative_mask = (self.iou_negative_low <= proposals_max_iou_val) & (proposals_max_iou_val < self.iou_negative_high)
        
        # Mark proposals as between thresholds
        between_mask = (self.iou_negative_high <= proposals_max_iou_val) & (proposals_max_iou_val < self.iou_positive_thresh)
        between_mask = between_mask | (proposals_max_iou_val < self.iou_negative_low)
        
        proposals_max_iou_idx[negative_mask] = self.BELOW_LOW_THRESHOLD
        proposals_max_iou_idx[between_mask] = self.BETWEEN_THRESHOLDS
        
        if self.low_quality_match:
            gt_max_iou_val, _ = match_quality_matrix.max(dim=1)
            positive_idx = torch.where(match_quality_matrix == gt_max_iou_val[:, None])[1]
            proposals_max_iou_idx[positive_idx] = proposals_match[positive_idx]
        return proposals_max_iou_idx
    
def giou_loss(pred_boxes, target_boxes, eps=1e-7):
    """
    Generalized Intersection over Union Loss
    Args:
        pred_boxes: [N,4] predicted bounding boxes
        target_boxes: [N,4] target bounding boxes
        eps: epsilon to avoid divide by zero
    Returns:
        GIoU loss
    """
    # Calculate intersection area
    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    
    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    
    # Calculate union area
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    union = pred_area + target_area - intersection + eps
    
    # Calculate IoU
    iou = intersection / union
    
    # Calculate enclosing box
    x1_enc = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    y1_enc = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    x2_enc = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    y2_enc = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    
    # Calculate area of enclosing box
    enc_area = (x2_enc - x1_enc) * (y2_enc - y1_enc) + eps
    
    # Calculate GIoU
    giou = iou - (enc_area - union) / enc_area
    
    # GIoU loss
    return 1 - giou

def ciou_loss(pred_boxes, target_boxes, eps=1e-7):
    """
    Complete Intersection over Union Loss
    Args:
        pred_boxes: [N,4] predicted bounding boxes
        target_boxes: [N,4] target bounding boxes
        eps: epsilon to avoid divide by zero
    Returns:
        CIoU loss
    """
    # Calculate intersection area
    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    
    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    
    # Calculate union area
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    union = pred_area + target_area - intersection + eps
    
    # Calculate IoU
    iou = intersection / union
    
    # Calculate center distance
    pred_center_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_center_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    target_center_x = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    target_center_y = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
    
    center_dist = (pred_center_x - target_center_x).pow(2) + (pred_center_y - target_center_y).pow(2)
    
    # Calculate enclosing diagonal distance
    x1_enc = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    y1_enc = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    x2_enc = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    y2_enc = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    enc_diag = (x2_enc - x1_enc).pow(2) + (y2_enc - y1_enc).pow(2) + eps
    
    # Calculate aspect ratio
    pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
    pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
    target_w = target_boxes[:, 2] - target_boxes[:, 0]
    target_h = target_boxes[:, 3] - target_boxes[:, 1]
    
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h), 2)
    alpha = v / (1 - iou + v + eps)
    
    # Calculate CIoU
    ciou = iou - (center_dist / enc_diag) - (alpha * v)
    
    # CIoU loss
    return 1 - ciou

def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    Focal Loss
    Args:
        inputs: prediction logits
        targets: target labels
        alpha: weighting factor
        gamma: focusing parameter
        reduction: reduction method
    Returns:
        Focal loss
    """
    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1-pt)**gamma * BCE_loss
    
    if reduction == 'mean':
        return torch.mean(F_loss)
    elif reduction == 'sum':
        return torch.sum(F_loss)
    else:
        return F_loss