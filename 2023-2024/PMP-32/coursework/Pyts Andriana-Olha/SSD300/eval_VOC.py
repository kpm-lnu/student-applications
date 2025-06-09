from utils import *
from data_set import PascalVOCDataset
from tqdm import tqdm
from pprint import PrettyPrinter

# Хороше форматування під час друку AP для кожного класу та mAP
pp = PrettyPrinter()

# Параметри
data_folder = './'
keep_difficult = True  
batch_size = 32
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = './checkpoint_ssd300.pth.tar'

# Завантажте контрольну точку моделі, яку потрібно оцінити
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)

# Перейти в режим оцінювання   
model.eval()

# Завантажити тестові дані
test_dataset = PascalVOCDataset(data_folder,
                                split='test',
                                keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)

def evaluate(test_loader, model):
    # Переконайтеся, що він у режимі оцінки
    model.eval()
    # Списки для зберігання виявлених і справжніх ящиків, міток, балів
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  

    with torch.no_grad():
        # Партії
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)
            labels = [l.to(device) for l in labels]
            boxes = [b.to(device) for b in boxes]
            predicted_locs, predicted_scores = model(images)
            # Виявлення об'єктів у вихідних даних SSD
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                        min_score=0.01, max_overlap=0.45,
                                                                                        top_k=200)
            # Збережіть результати цієї партії для розрахунку mAP
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

        # Обчислити mAP
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)
    # Роздрукуйте AP для кожного класу
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)


if __name__ == '__main__':
    evaluate(test_loader, model)