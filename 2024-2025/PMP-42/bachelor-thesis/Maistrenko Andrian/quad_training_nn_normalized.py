import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

# === 1. Модель ===
class QuadPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 12)
        )

    def forward(self, x):
        return self.model(x)

# === 2. Парсинг даних ===
def load_dataset(filename):
    X, Y = [], []

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found!")
        return np.array([]), np.array([])

    print(f"Total lines in file: {len(lines)}")
    
    for line_num, line in enumerate(lines, 1):
        if not line.startswith("Quad"):
            continue

        try:
            content = line.strip().split(":")[1].strip()
            
            import re
            pattern = r'\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*w=([-+]?\d*\.?\d+)\)'
            matches = re.findall(pattern, content)
            
            if len(matches) != 4:
                continue
            
            coords = []
            weights = []
            
            for match in matches:
                x = float(match[0])
                y = float(match[1]) 
                w = float(match[2])
                coords.extend([x, y])
                weights.append(w)
            
            free_input = []
            for i in range(4):
                x, y = coords[2 * i], coords[2 * i + 1]
                is_fixed = ((abs(x) < 1e-6 and abs(y) < 1e-6) or  # (0,0)
                           (abs(x - 1.0) < 1e-6 and abs(y) < 1e-6) or  # (1,0)
                           (abs(x - 2.0) < 1e-6 and abs(y) < 1e-6))    # (2,0)
                
                if not is_fixed:
                    free_input.extend([x, y])
            
            if len(free_input) == 4:
                X.append(free_input)
                Y.append(coords + weights)
                
        except Exception as e:
            print(f"Error processing line {line_num}: {e}")
            continue

    print(f"Successfully loaded {len(X)} samples")
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# === 3. Нормалізація даних ===
def create_scalers(X_train, Y_train, normalization_type='standard'):

    if normalization_type == 'standard':
        X_scaler = StandardScaler()
        Y_scaler = StandardScaler()
    elif normalization_type == 'minmax':
        X_scaler = MinMaxScaler()
        Y_scaler = MinMaxScaler()
    else:
        raise ValueError("normalization_type must be 'standard' or 'minmax'")
    
    X_scaler.fit(X_train)
    Y_scaler.fit(Y_train)
    
    return X_scaler, Y_scaler

def normalize_data(X_train, X_val, Y_train, Y_val, X_scaler, Y_scaler):
    X_train_norm = X_scaler.transform(X_train)
    X_val_norm = X_scaler.transform(X_val)
    Y_train_norm = Y_scaler.transform(Y_train)
    Y_val_norm = Y_scaler.transform(Y_val)
    
    return X_train_norm, X_val_norm, Y_train_norm, Y_val_norm

# === 4. Тренування з нормалізацією ===
def train_model_normalized(X_train, Y_train, X_val, Y_val, 
                          normalization_type='standard', epochs=2000, lr=1e-3):
    
    X_scaler, Y_scaler = create_scalers(X_train, Y_train, normalization_type)
    
    X_train_norm, X_val_norm, Y_train_norm, Y_val_norm = normalize_data(
        X_train, X_val, Y_train, Y_val, X_scaler, Y_scaler)
    
    print(f"Data normalization: {normalization_type}")
    print(f"X_train range: [{X_train_norm.min():.3f}, {X_train_norm.max():.3f}]")
    print(f"Y_train range: [{Y_train_norm.min():.3f}, {Y_train_norm.max():.3f}]")
    
    model = QuadPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    X_train_tensor = torch.from_numpy(X_train_norm.astype(np.float32))
    Y_train_tensor = torch.from_numpy(Y_train_norm.astype(np.float32))
    X_val_tensor = torch.from_numpy(X_val_norm.astype(np.float32))
    Y_val_tensor = torch.from_numpy(Y_val_norm.astype(np.float32))

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, Y_train_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(X_val_tensor)
            val_loss = criterion(val_output, Y_val_tensor)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict()

        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")

    print(f"Final - Train Loss: {loss.item():.6f}, Val Loss: {best_val_loss:.6f}")
    model.load_state_dict(best_model_state)
    
    return model, X_scaler, Y_scaler

# === 5. Збереження моделі та скейлерів ===
def save_model_with_scalers(model, X_scaler, Y_scaler, model_path="quad_predictor_normalized.pth", 
                           scaler_path="scalers.pkl"):
    torch.save(model.state_dict(), model_path)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump({'X_scaler': X_scaler, 'Y_scaler': Y_scaler}, f)
    
    print(f"Model saved to {model_path}")
    print(f"Scalers saved to {scaler_path}")

# === 6. Завантаження моделі та скейлерів ===
def load_model_with_scalers(model_path="quad_predictor_normalized.pth", 
                           scaler_path="scalers.pkl"):
    model = QuadPredictor()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    
    return model, scalers['X_scaler'], scalers['Y_scaler']

# === 7. Прогнозування з нормалізацією ===
def predict_normalized(model, X_scaler, Y_scaler, X_new):
    model.eval()
    with torch.no_grad():
        X_new_norm = X_scaler.transform(X_new)
        X_tensor = torch.from_numpy(X_new_norm.astype(np.float32))
        
        Y_pred_norm = model(X_tensor).numpy()
        
        Y_pred = Y_scaler.inverse_transform(Y_pred_norm)
        
    return Y_pred

# === 8. Виконання ===
if __name__ == "__main__":
    X, Y = load_dataset("PSO/all_optimized_combined.txt")
    
    if len(X) == 0:
        print("No data loaded!")
        exit(1)
    
    print(f"Loaded {len(X)} samples.")
    print(f"Input range: X min={X.min():.3f}, max={X.max():.3f}")
    print(f"Output range: Y min={Y.min():.3f}, max={Y.max():.3f}")

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    print("\n=== Training with Standard Normalization ===")
    model_std, X_scaler_std, Y_scaler_std = train_model_normalized(
        X_train, Y_train, X_val, Y_val, normalization_type='standard', epochs=2000)
    save_model_with_scalers(model_std, X_scaler_std, Y_scaler_std, 
                           "quad_predictor_standard.pth", "scalers_standard.pkl")

    print("\n=== Training with MinMax Normalization ===")
    model_minmax, X_scaler_minmax, Y_scaler_minmax = train_model_normalized(
        X_train, Y_train, X_val, Y_val, normalization_type='minmax', epochs=2000)
    save_model_with_scalers(model_minmax, X_scaler_minmax, Y_scaler_minmax, 
                           "quad_predictor_minmax.pth", "scalers_minmax.pkl")
    
    print("\n=== Example Usage ===")
    X_test = X_val[:5]  # Перші 5 зразків з валідації
    
    Y_pred_std = predict_normalized(model_std, X_scaler_std, Y_scaler_std, X_test)
    print(f"Standard normalization predictions shape: {Y_pred_std.shape}")
    
    Y_pred_minmax = predict_normalized(model_minmax, X_scaler_minmax, Y_scaler_minmax, X_test)
    print(f"MinMax normalization predictions shape: {Y_pred_minmax.shape}")