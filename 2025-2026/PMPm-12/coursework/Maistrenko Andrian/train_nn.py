import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import random
import time

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# ==========================================
# 1. ЗАВАНТАЖЕННЯ ТА ПІДГОТОВКА ДАНИХ
# ==========================================
print("Завантаження датасету 'fem_stress_dataset_method_b.csv'...")

try:
    # df = pd.read_csv('fem_stress_dataset_method_b.csv')
    df = pd.read_csv('fem_stress_dataset_method_b_2000.csv')
except FileNotFoundError:
    print("Помилка: Файл 'fem_stress_dataset_method_b.csv' не знайдено! Спочатку згенеруй дані.")
    exit()

print(f"Загальна кількість зібраних вузлів: {len(df)}")

# Визначаємо цільові змінні (Y)
y_cols = [col for col in df.columns if col.startswith('TARGET_')]

# Визначаємо вхідні ознаки (X)
ignore_cols = ['sample_id', 'r', 'z', 'p'] + y_cols
x_cols = [col for col in df.columns if col not in ignore_cols]

print(f"Кількість вхідних ознак (X): {len(x_cols)}")
print(f"Кількість цільових змінних (Y): {len(y_cols)}")

X = df[x_cols].values
Y = df[y_cols].values

scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

# ==========================================
# 2. ПІДГОТОВКА DATASET ДЛЯ PYTORCH
# ==========================================
class FEMDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.float32)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

train_dataset = FEMDataset(X_train, Y_train)
test_dataset = FEMDataset(X_test, Y_test)

batch_size = 128 if len(X_train) > 5000 else 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==========================================
# 3. АРХІТЕКТУРА НЕЙРОННОЇ МЕРЕЖІ
# ==========================================
class StressPredictorNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(StressPredictorNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.2), 
            nn.Linear(512,512),
            nn.LeakyReLU(),
            nn.Dropout(0.1), 
            nn.Linear(512,512),
            nn.LeakyReLU(),
            nn.Linear(512, output_size)
        )

    def forward(self, x):
        return self.network(x)
model = StressPredictorNN(input_size=len(x_cols), output_size=len(y_cols))

# ==========================================
# 4. НАЛАШТУВАННЯ НАВЧАННЯ
# ==========================================
criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

epochs = 200
train_losses = []
test_losses = []

print("\nПочаток тренування...")
# class EarlyStopping:
#     def __init__(self, patience=20, min_delta=1e-4):
#         self.patience = patience      # how many epochs to wait after last improvement
#         self.min_delta = min_delta    # minimum change to count as improvement
#         self.best_loss = float('inf')
#         self.counter = 0
#         self.best_weights = None

#     def step(self, val_loss, model):
#         if val_loss < self.best_loss - self.min_delta:
#             self.best_loss = val_loss
#             self.counter = 0
#             self.best_weights = copy.deepcopy(model.state_dict())  # save best
#         else:
#             self.counter += 1

#         return self.counter >= self.patience  # returns True when should stop
# ==========================================
# 5. ЦИКЛ ТРЕНУВАННЯ
# ==========================================
# early_stopping = EarlyStopping(patience=40, min_delta=5e-5)
start = time.time()
for epoch in range(epochs):
    model.train()
    running_train_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * batch_x.size(0)
    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    model.eval()
    running_test_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            running_test_loss += loss.item() * batch_x.size(0)
    epoch_test_loss = running_test_loss / len(test_loader.dataset)
    test_losses.append(epoch_test_loss)

    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(f"Епоха [{epoch+1}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Test Loss: {epoch_test_loss:.4f}")

    # --- early stopping ---
    # if early_stopping.step(epoch_test_loss, model):
    #     print(f"⏹ Early stopping на епосі {epoch+1} | Найкращий Test Loss: {early_stopping.best_loss:.4f}")
    #     break

# restore best weights
# model.load_state_dict(early_stopping.best_weights)
print("Навчання завершено!")

# ==========================================
# 6. ОЦІНКА ЯКОСТІ ТА ВІЗУАЛІЗАЦІЯ
# ==========================================
# Збереження моделі
torch.save(model.state_dict(), "fem_stress_model_method_b.pth")
print("\nМодель збережено у 'fem_stress_model_method_b.pth'")

# Візуалізація Loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Епохи')
plt.ylabel('MSE Loss (Нормалізований)')
plt.title('Процес навчання нейромережі')
plt.legend()
plt.grid(True)
plt.savefig('training_loss_method_b.png', dpi=300)
print("Графік навчання збережено як 'training_loss_method_b.png'")

model.eval()
with torch.no_grad():
    test_inputs = torch.tensor(X_test, dtype=torch.float32)
    test_targets_scaled = torch.tensor(Y_test, dtype=torch.float32)
    
    predictions_scaled = model(test_inputs)
    
    # Повертаємо справжні фізичні значення
    predictions_real = scaler_Y.inverse_transform(predictions_scaled.numpy())
    targets_real = scaler_Y.inverse_transform(test_targets_scaled.numpy())

    mae = np.mean(np.abs(predictions_real - targets_real), axis=0)
    
    print("\nОцінка точності (Середня абсолютна похибка в Паскалях):")
    for idx, col in enumerate(y_cols):
        print(f"  {col}: {mae[idx]:.2e}")
end=time.time()
print(f"It took {end-start} seconds")
