import math

def calculate_distance(tx_power, signal_strength):
    """
    Розрахунок відстані до точки доступу Wi-Fi на основі потужності передавача та сили сигналу.
    Формула: d = 10 ^ ((TxPower - RSSI) / (10 * n))
    """
    n = 2  # емпіричний коефіцієнт затухання сигналу
    return 10 ** ((tx_power - signal_strength) / (10 * n))

# Параметри точки доступу Wi-Fi
tx_power = -40  # Потужність передавача в dBm

# Введення сили сигналу користувачем
signal_strength = float(input("Введіть силу сигналу (в dBm): "))

# Розрахунок відстані на основі виміряного RSS
calculated_distance = calculate_distance(tx_power, signal_strength)
print(f"Розрахована відстань: {calculated_distance} метрів")
