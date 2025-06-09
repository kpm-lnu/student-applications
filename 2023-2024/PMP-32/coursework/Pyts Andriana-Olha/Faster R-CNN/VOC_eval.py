import torch, torchvision
# Обчислює середню точність (AP) для заданих кривих відкликання та точності.
def voc_ap(recall, precision, use_07_metric):
    if use_07_metric:
        ap = 0.
        for thresh in torch.arange(0., 1.1, 0.1):
            if torch.sum(recall >= thresh) == 0: p = 0
            else: p = float(torch.max(precision[recall >= thresh]).cpu())
            ap = ap + p / 11.
    else:
        rec = torch.cat((torch.tensor([0]), recall, torch.tensor([1])))
        pre = torch.cat((torch.tensor([0]), precision, torch.tensor([0])))

        # Зробіть криву точності незростаючою
        for i in range(pre.shape[0]-1, 0, -1):
            pre[i-1] = torch.max(pre[i-1], pre[i])

        # Обчисліть площу під кривою точного відкликання
        i = torch.where(rec[1:] != rec[:-1])[0]
        ap = float(torch.sum((rec[i+1] - rec[i]) * pre[i+1]).cpu())
    return ap

# Оцінює результати виявлення за допомогою метрики VOC
def voc_eval(pred_bboxs, pred_labels, pred_scores, gt_bboxs, gt_labels, num_classes, iou_thresh, use_07_metric):
    # Об'єднує всі передбачення та обгрунтовані істини на зображеннях
    pred_bboxs_concat = torch.cat(pred_bboxs)
    pred_labels_concat = torch.cat(pred_labels)
    pred_scores_concat = torch.cat(pred_scores)
    
    gt_bboxs_concat = torch.cat(gt_bboxs).cuda(pred_bboxs_concat.device)
    gt_labels_concat = torch.cat(gt_labels).cuda(pred_labels_concat.device)
    
    # Створює списки ідентифікаторів зображень для кожної базової правди та прогнозу
    gt_image_ids, pred_image_ids = [], []
    for image_id in range(len(gt_labels)): gt_image_ids += [image_id] * gt_labels[image_id].shape[0]
    for image_id in range(len(pred_labels)): pred_image_ids += [image_id] * pred_labels[image_id].shape[0]
    
    gt_image_ids = torch.tensor(gt_image_ids, device=gt_labels_concat.device)
    pred_image_ids = torch.tensor(pred_image_ids, device=pred_labels_concat.device)
    
    aps = []
    for class_idx in range(1, num_classes):
        # Отримайте базову правдивість і прогнозовані дані для поточного класу
        gt_masks_per_class = gt_labels_concat == class_idx
        gt_bboxs_per_class = gt_bboxs_concat[gt_masks_per_class]
        gt_image_ids_per_class = gt_image_ids[gt_masks_per_class]
        
        check = torch.zeros_like(gt_image_ids_per_class, dtype=bool)
        gt_num = torch.sum(gt_masks_per_class)
        
        pred_masks_per_class = pred_labels_concat == class_idx
        pred_bboxs_per_class = pred_bboxs_concat[pred_masks_per_class]
        pred_scores_per_class = pred_scores_concat[pred_masks_per_class]
        pred_image_ids_per_class = pred_image_ids[pred_masks_per_class]
        
        # Сортує прогнози за кількістю балів у порядку спадання
        sort_idx_per_class = torch.argsort(-pred_scores_per_class)
        sort_pred_bboxs_per_class = pred_bboxs_per_class[sort_idx_per_class]
        sort_pred_image_ids_per_class = pred_image_ids_per_class[sort_idx_per_class]
        
        pred_num = torch.sum(pred_masks_per_class)
        tp, fp = torch.zeros(pred_num, device=pred_num.device), torch.zeros(pred_num, device=pred_num.device)
        
        # Оцінює кожен прогноз
        for i in range(pred_num):
            match_idx = torch.where(gt_image_ids_per_class == sort_pred_image_ids_per_class[i])[0]
            if match_idx.nelement() != 0:
                gt_bboxs_target = gt_bboxs_per_class[match_idx].clone()
                pred_bboxs_target = sort_pred_bboxs_per_class[i].clone().view(-1, 4)
                
                # Відрегулюйте 1-піксельну невідповідність між базовою істинністю та обмежувальними рамками передбачення
                gt_bboxs_target[:, 2:] += 1
                pred_bboxs_target[:, 2:] += 1
                
                # Обчислює IU між передбаченням і базовою правдою
                IoUs = torchvision.ops.box_iou(gt_bboxs_target, pred_bboxs_target).view(-1)
                IoU_idx, IoU = torch.argmax(IoUs), torch.max(IoUs)
                
                # Визначає, чи є прогноз справді позитивним чи хибним
                if (IoU > iou_thresh) and (check[match_idx][IoU_idx] == False):
                    tp[i] = 1.
                    check[match_idx][IoU_idx] = True
                else:
                    fp[i] = 1.
            else:
                fp[i] = 1.
        
        # Обчислює кумулятивні істинні та хибні спрацьовування        
        tp, fp = torch.cumsum(tp, dim=0), torch.cumsum(fp, dim=0)
        
        # Обчислює криві відкликання та точності
        recall = tp / gt_num
        precision = tp / (tp + fp)
        
        # Обчислює середню точність для поточного класу
        ap = voc_ap(recall, precision, use_07_metric)
        aps.append(ap)
    # Обчислити середню точність (mAP)
    return sum(aps) / len(aps)