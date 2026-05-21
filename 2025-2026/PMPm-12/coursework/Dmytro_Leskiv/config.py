from dataclasses import dataclass, field
from typing import Dict, List


BASE_THREATS: List[Dict[str, object]] = [
    {"name": "Перехоплення трафіку",        "probability": 0.24, "damage": 0.65, "target_layer": "network"},
    {"name": "Компрометація ключів",         "probability": 0.18, "damage": 0.85, "target_layer": "crypto"},
    {"name": "Підміна даних",                "probability": 0.20, "damage": 0.75, "target_layer": "network"},
    {"name": "Несанкціонований доступ",      "probability": 0.16, "damage": 0.90, "target_layer": "device"},
    {"name": "Соціальна інженерія",          "probability": 0.22, "damage": 0.70, "target_layer": "social"},
    {"name": "DoS / перевантаження",         "probability": 0.14, "damage": 0.60, "target_layer": "network"},
    {"name": "Компрометація хмарного акаунта","probability": 0.12, "damage": 0.95, "target_layer": "cloud"},
]

SCENARIOS: Dict[str, Dict[str, float]] = {
    "base": {
        "attack_intensity": 0.60,
        "human_factor":     0.35,
        "channel_instability": 0.25,
        "node_count":  7,
        "data_volume": 35.0,
    },
    "high_attack": {
        "attack_intensity": 0.90,
        "human_factor":     0.35,
        "channel_instability": 0.35,
        "node_count":  7,
        "data_volume": 35.0,
    },
    "high_human_factor": {
        "attack_intensity": 0.60,
        "human_factor":     0.75,
        "channel_instability": 0.25,
        "node_count":  7,
        "data_volume": 35.0,
    },
    "large_system": {
        "attack_intensity": 0.65,
        "human_factor":     0.40,
        "channel_instability": 0.30,
        "node_count": 14,
        "data_volume": 60.0,
    },
}

# Людські читабельні назви сценаріїв для графіків
SCENARIO_LABELS: Dict[str, str] = {
    "base":              "Базовий",
    "high_attack":       "Висока інтенсивність атак",
    "high_human_factor": "Високий людський чинник",
    "large_system":      "Велика система",
}

METHOD_LABELS: Dict[str, str] = {
    "M1_Класичний_одноконтурний":    "M1 — одноконтурний",
    "M2_Класичний_багатоконтурний":  "M2 — багатоконтурний",
    "M3_Гібридний_постквантовий":    "M3 — постквантовий",
}


@dataclass
class ModelConfig:
    seed: int = 42
    monte_carlo_runs: int = 300
    output_dir: str = "results"

    weights: Dict[str, float] = field(default_factory=lambda: {
        "security_score":      0.22,
        "post_quantum_score":  0.15,
        "risk":                0.18,
        "delay":               0.12,
        "computational_cost":  0.11,
        "energy_cost":         0.08,
        "human_factor_score":  0.09,
        "overhead":            0.05,
    })

    attack_sweep_values: List[float] = field(
        default_factory=lambda: [0.20, 0.35, 0.50, 0.65, 0.80, 0.95]
    )
    human_sweep_values: List[float] = field(
        default_factory=lambda: [0.10, 0.20, 0.35, 0.50, 0.65, 0.80]
    )

    def __post_init__(self) -> None:
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("Сума ваг повинна бути додатною.")
        self.weights = {k: v / total for k, v in self.weights.items()}