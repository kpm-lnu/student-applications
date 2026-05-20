import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==========================================
# 1. ЗАВАНТАЖЕННЯ ДАНИХ
# ==========================================
df = pd.read_csv('fem_stress_dataset.csv')

y_cols = [col for col in df.columns if col.startswith('TARGET_')]
ignore_cols = ['sample_id', 'r', 'z', 'p'] + y_cols
x_cols = [col for col in df.columns if col not in ignore_cols]

X = df[x_cols].values
Y = df[y_cols].values

scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

# ==========================================
# 2. ЗАВАНТАЖЕННЯ МОДЕЛІ
# ==========================================
class StressPredictorNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(StressPredictorNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, output_size)
        )
    def forward(self, x):
        return self.network(x)

model = StressPredictorNN(input_size=len(x_cols), output_size=len(y_cols))
model.load_state_dict(torch.load("fem_stress_model.pth"))
model.eval()

# ==========================================
# 3. РОЗРАХУНОК ПОХИБОК
# ==========================================
with torch.no_grad():
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    predictions_scaled = model(X_tensor)
    # Повертаємо у фізичні величини (Паскалі)
    predictions_real = scaler_Y.inverse_transform(predictions_scaled.numpy())

error_coarse = Y 
error_nn = Y - predictions_real 

# ==========================================
# 4. ВІЗУАЛІЗАЦІЯ 
# ==========================================
components = ['srr', 'szz', 'srz', 'stt']
titles = [r'Radial Stress $\sigma_{rr}$', r'Axial Stress $\sigma_{zz}$', 
          r'Shear Stress $\sigma_{rz}$', r'Hoop Stress $\sigma_{\theta\theta}$']

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i in range(4):
    ax = axs[i]
    
    min_val = np.percentile(error_coarse[:, i], 1)
    max_val = np.percentile(error_coarse[:, i], 99)
    limit = max(abs(min_val), abs(max_val))
    if limit == 0: limit = 1e-5
    bins = np.linspace(-limit, limit, 100)
    
    ax.hist(error_coarse[:, i], bins=bins, color='red', alpha=0.6, 
            histtype='step', linestyle='--', linewidth=2, label=r'Error Coarse ($\sigma^F - \sigma^C$)')
    
    ax.hist(error_nn[:, i], bins=bins, color='blue', alpha=0.8, 
            histtype='step', linestyle='-', linewidth=2, label=r'Error NN ($\sigma^F - \sigma^{NN}$)')
    
    ax.set_yscale('log')
    ax.set_title(f'Error Distribution: {titles[i]}')
    ax.set_xlabel('Error (Pa)')
    ax.set_ylabel('Number of Patterns (log scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('error_distribution.png', dpi=300)
print("Графіки розподілу похибок збережено у 'error_distribution.png'")
