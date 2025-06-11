import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss, FocalCIoULoss, CEGIoULoss, FocalSmoothL1Loss
from datasets import PascalVOCDataset
from utils import *

# Параметри даних
data_folder = './'  
keep_difficult = True  

# Параметри моделі
# Тут не надто багато, оскільки SSD300 має дуже специфічну структуру
n_classes = len(label_map) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Параметри навчання
checkpoint = None  # path to model checkpoint, None if none
batch_size = 8  # batch size
iterations = 120000  # number of iterations to train
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 200  # print training status every __ batches
lr = 1e-3  # learning rate
decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True


def main():
    """
    Навчання.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Ініціалізація моделі або контрольна точка завантаження
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Перейти до пристрою за умовчанням
    model = model.to(device)
    #criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)
    #criterion = FocalSmoothL1Loss(priors_cxcy=model.priors_cxcy).to(device)
    #criterion = CEGIoULoss(priors_cxcy=model.priors_cxcy).to(device)
    criterion = FocalCIoULoss(priors_cxcy=model.priors_cxcy).to(device)

    # Спеціальні завантажувачі даних
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    # Обчислити загальну кількість епох для навчання та епох для зниження швидкості навчання (тобто перетворити ітерації в епохи)
    epochs = iterations // (len(train_dataset) // 32)
    decay_lr_at = [it // (len(train_dataset) // 32) for it in decay_lr_at]

    # Епохи
    for epoch in range(start_epoch, epochs):

        # Загасання швидкості навчання в окремі епохи
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # Навчання однієї епохи
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # Зберегти КПП
        save_checkpoint(epoch, model, optimizer)


def train(train_loader, model, criterion, optimizer, epoch):
    """
    Навчання однієї епохи.

    :param train_loader: DataLoader для навчальних даних
    :param модель: модель
    :param критерій: втрата MultiBox
    :param optimizer: оптимізатор
    :param epoch: номер епохи
    """
    model.train()  
    batch_time = AverageMeter()  
    data_time = AverageMeter()  
    losses = AverageMeter()  

    start = time.time()

    # Партії
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Перейти до пристрою за умовчанням
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Передній проп.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Втрата
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Задня опора.
        optimizer.zero_grad()
        loss.backward()

        # Виріжте градієнти, якщо необхідно
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Оновити модель
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Статус друку
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored


def average_iou(det_boxes, true_boxes):
    all_ious = []
    for db, tb in zip(det_boxes, true_boxes):
        if db.size(0) == 0 or tb.size(0) == 0:
            continue
        iou = find_jaccard_overlap(db, tb)
        max_iou, _ = iou.max(dim=1)
        all_ious.extend(max_iou.tolist())
    return sum(all_ious) / len(all_ious) if all_ious else 0.



def evaluate(test_loader, model):
    model.eval()

    det_boxes = []
    det_labels = []
    det_scores = []
    true_boxes = []
    true_labels = []
    true_difficulties = []

    with torch.no_grad():
        for images, boxes, labels, difficulties in test_loader:
            images = images.to(device)
            predicted_locs, predicted_scores = model(images)
            boxes_pred, labels_pred, scores_pred = model.detect_objects(
                predicted_locs, predicted_scores,
                min_score=0.01, max_overlap=0.45, top_k=200)

            det_boxes.extend(boxes_pred)
            det_labels.extend(labels_pred)
            det_scores.extend(scores_pred)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

    # mAP
    APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores,
                             true_boxes, true_labels, true_difficulties)

    print('\nEvaluation Metrics:')
    print(f'mAP: {mAP:.4f}')
    for cls, ap in APs.items():
        print(f'{cls}: AP={ap:.4f}')

    # Recall and Precision
    total_tp = sum([l.size(0) for l in true_labels])
    total_det = sum([l.size(0) for l in det_labels])
    recall = total_tp / (total_tp + 1e-10)
    precision = total_tp / (total_det + 1e-10)
    print(f'Recall: {recall:.4f}, Precision: {precision:.4f}')

    # Average IoU
    avg_iou = average_iou(det_boxes, true_boxes)
    print(f'Average IoU: {avg_iou:.4f}')


if __name__ == '__main__':
    main()