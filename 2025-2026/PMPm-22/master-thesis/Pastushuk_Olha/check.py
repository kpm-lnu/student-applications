import torch

print(f"--- Перевірка системи ---")
print(f"Версія PyTorch: {torch.__version__}")
print(f"CUDA доступна: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Версія CUDA у PyTorch: {torch.version.cuda}")
    print(f"Поточна відеокарта: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
else:
    print("⚠️ CUDA не знайдена. Система використовує CPU.")