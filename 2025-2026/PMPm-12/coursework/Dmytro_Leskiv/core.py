from dataclasses import dataclass
from math import exp
from random import Random
from typing import Dict, List


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


@dataclass
class Threat:
    name: str
    probability: float
    damage: float
    target_layer: str


@dataclass
class EnvironmentParams:
    attack_intensity: float
    human_factor: float
    channel_instability: float
    node_count: int
    data_volume: float


@dataclass
class ProtectionMethod:
    name: str
    security_score: float
    post_quantum_score: float  # 0..1, де 1 = повна постквантова стійкість
    cpu_cost: float
    memory_cost: float
    energy_cost: float
    base_delay: float
    human_sensitivity: float
    overhead: float
    protection_efficiency: float
    risk_modifier: float


def build_threats(threat_data: List[Dict[str, object]]) -> List[Threat]:
    return [
        Threat(
            name=str(item["name"]),
            probability=float(item["probability"]),
            damage=float(item["damage"]),
            target_layer=str(item["target_layer"]),
        )
        for item in threat_data
    ]


def get_default_methods() -> List[ProtectionMethod]:
    """
    M1 — класичний одноконтурний (AES-128 + RSA-2048).
    M2 — класичний багатоконтурний (AES-256 + ECC-256, два контури).
    M3 — гібридний постквантовий (AES-256 + ССС на LDPC-кодах).
    """
    return [
        ProtectionMethod(
            name="M1_Класичний_одноконтурний",
            security_score=0.62,
            post_quantum_score=0.15,   # RSA-2048 → ~0 у PQ-сценарії
            cpu_cost=0.35,
            memory_cost=0.30,
            energy_cost=0.32,
            base_delay=0.28,
            human_sensitivity=0.85,
            overhead=0.20,
            protection_efficiency=0.75,
            risk_modifier=1.00,
        ),
        ProtectionMethod(
            name="M2_Класичний_багатоконтурний",
            security_score=0.78,
            post_quantum_score=0.25,   # AES-256 частково стійкий, ECC-256 — ні
            cpu_cost=0.48,
            memory_cost=0.42,
            energy_cost=0.44,
            base_delay=0.38,
            human_sensitivity=0.55,
            overhead=0.30,
            protection_efficiency=1.05,
            risk_modifier=0.82,
        ),
        ProtectionMethod(
            name="M3_Гібридний_постквантовий",
            security_score=0.90,
            post_quantum_score=0.92,   # ССС на LDPC: S_pq ≈ S_cl
            cpu_cost=0.72,
            memory_cost=0.68,
            energy_cost=0.70,
            base_delay=0.62,
            human_sensitivity=0.45,
            overhead=0.48,
            protection_efficiency=1.30,
            risk_modifier=0.68,
        ),
    ]


def compute_human_factor_score(method: ProtectionMethod, env: EnvironmentParams) -> float:
    return clamp(env.human_factor * method.human_sensitivity, 0.0, 1.0)


def compute_attack_probability(
    method: ProtectionMethod,
    env: EnvironmentParams,
    human_factor_score: float,
) -> float:
    """
    При високій інтенсивності атак (λ > 0.7) методи з низькою
    постквантовою стійкістю отримують додатковий штраф — це відображає
    можливість квантових атак у майбутньому.
    """
    alpha = max(0.15, method.protection_efficiency * method.security_score)

    # Постквантовий штраф: актуальний при λ > 0.7, максимум +30% до агресивності
    pq_penalty = 0.0
    if env.attack_intensity > 0.70:
        pq_deficit = 1.0 - method.post_quantum_score   # 0 для M3, ~0.85 для M1
        pq_penalty = 0.30 * pq_deficit * (env.attack_intensity - 0.70) / 0.30

    effective_intensity = env.attack_intensity * (1.0 + pq_penalty)

    probability = 1.0 - exp(
        -(effective_intensity * (1.0 + env.channel_instability) * (1.0 + human_factor_score))
        / alpha
    )
    return clamp(probability, 0.0, 1.0)


def compute_delay(method: ProtectionMethod, env: EnvironmentParams) -> float:
    d_s = env.data_volume / 50.0
    d_n = env.node_count / 10.0
    delay = method.base_delay * (1.0 + 0.55 * d_s + 0.35 * d_n + 0.40 * env.channel_instability)
    return max(delay, 0.001)


def compute_computational_cost(method: ProtectionMethod, env: EnvironmentParams) -> float:
    d_s = env.data_volume / 50.0
    d_n = env.node_count / 10.0
    cost = (method.cpu_cost + 0.65 * method.memory_cost) * (1.0 + 0.50 * d_n + 0.60 * d_s)
    return max(cost, 0.001)


def compute_energy_cost(method: ProtectionMethod, env: EnvironmentParams) -> float:
    d_s = env.data_volume / 50.0
    d_n = env.node_count / 10.0
    energy = method.energy_cost * (1.0 + 0.45 * d_s + 0.30 * d_n + 0.30 * env.channel_instability)
    return max(energy, 0.001)


def compute_overhead(method: ProtectionMethod, env: EnvironmentParams) -> float:
    d_s = env.data_volume / 50.0
    overhead = method.overhead * (1.0 + 0.25 * d_s)
    return max(overhead, 0.001)


def compute_effective_security(
    method: ProtectionMethod,
    env: EnvironmentParams,
    human_factor_score: float,
    attack_probability: float,
) -> float:
    sec = method.security_score
    sec *= (1.0 - 0.25 * env.channel_instability)
    sec *= (1.0 - 0.20 * human_factor_score)
    sec *= (1.0 - 0.15 * attack_probability)
    return clamp(sec, 0.05, 0.99)


def compute_risk(
    method: ProtectionMethod,
    env: EnvironmentParams,
    threats: List[Threat],
    human_factor_score: float,
    rng: Random,
) -> float:
    total_damage = sum(t.damage for t in threats)
    total_risk = 0.0

    for threat in threats:
        prob = threat.probability * env.attack_intensity

        layer_mult = 1.0
        if threat.target_layer == "network":
            layer_mult += 0.60 * env.channel_instability
        elif threat.target_layer == "social":
            layer_mult += 0.80 * env.human_factor
        elif threat.target_layer == "cloud":
            layer_mult += 0.30 * env.channel_instability + 0.20 * env.attack_intensity
        elif threat.target_layer == "device":
            layer_mult += 0.20 * (env.node_count / 10.0)
        elif threat.target_layer == "crypto":
            # Криптографічні загрози посилюються при низькій PQ-стійкості
            pq_factor = 1.0 + 0.40 * (1.0 - method.post_quantum_score)
            layer_mult += 0.15 * env.attack_intensity * pq_factor

        sys_mult    = 1.0 + 0.08 * (env.node_count / 10.0) + 0.05 * (env.data_volume / 50.0)
        human_mult  = (1.0 + human_factor_score) if threat.target_layer == "social" else 1.0
        prot_mult   = method.risk_modifier / max(
            0.20, method.protection_efficiency * (0.8 + 0.4 * method.security_score)
        )
        noise = rng.uniform(0.95, 1.05)

        eff_prob = clamp(prob * layer_mult * sys_mult * human_mult * prot_mult * noise, 0.0, 0.98)
        total_risk += eff_prob * threat.damage

    return total_risk / max(total_damage, 1e-9)


def _safe_norm(value: float, mn: float, mx: float) -> float:
    """Нормування з захистом від ділення на нуль."""
    return 1.0 if abs(mx - mn) < 1e-12 else (value - mn) / (mx - mn)


def evaluate_method(
    method: ProtectionMethod,
    env: EnvironmentParams,
    threats: List[Threat],
    rng: Random,
) -> Dict[str, float]:
    hf_score       = compute_human_factor_score(method, env)
    attack_prob    = compute_attack_probability(method, env, hf_score)
    delay          = compute_delay(method, env)
    comp_cost      = compute_computational_cost(method, env)
    energy         = compute_energy_cost(method, env)
    overhead       = compute_overhead(method, env)
    eff_security   = compute_effective_security(method, env, hf_score, attack_prob)
    risk           = compute_risk(method, env, threats, hf_score, rng)

    return {
        "method":              method.name,
        "security_score":      eff_security,
        "post_quantum_score":  method.post_quantum_score,
        "risk":                risk,
        "attack_probability":  attack_prob,
        "delay":               delay,
        "computational_cost":  comp_cost,
        "energy_cost":         energy,
        "human_factor_score":  hf_score,
        "overhead":            overhead,
    }


def compute_integral_scores(
    raw_results: List[Dict[str, float]],
    weights: Dict[str, float],
) -> List[Dict[str, float]]:
    benefit_metrics = ["security_score", "post_quantum_score"]
    cost_metrics    = ["risk", "delay", "computational_cost",
                       "energy_cost", "human_factor_score", "overhead"]
    all_metrics     = benefit_metrics + cost_metrics

    ranges: Dict[str, Dict[str, float]] = {}
    for m in all_metrics:
        vals = [float(row[m]) for row in raw_results]
        ranges[m] = {"min": min(vals), "max": max(vals)}

    scored: List[Dict[str, float]] = []
    for row in raw_results:
        r = dict(row)
        for m in benefit_metrics:
            r[f"{m}_norm"] = _safe_norm(float(row[m]), ranges[m]["min"], ranges[m]["max"])
        for m in cost_metrics:
            r[f"{m}_norm"] = _safe_norm(ranges[m]["max"], float(row[m]), ranges[m]["max"]) \
                             if abs(ranges[m]["max"] - ranges[m]["min"]) < 1e-12 else \
                             (ranges[m]["max"] - float(row[m])) / (ranges[m]["max"] - ranges[m]["min"])

        j = sum(weights[m] * r[f"{m}_norm"] for m in all_metrics if m in weights)
        r["integral_score"] = j
        scored.append(r)

    return scored