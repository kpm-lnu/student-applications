import pandas as pd
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tabulate import tabulate

# --- Завантаження даних ---
full_df = pd.read_excel('Dataset.xlsx')
full_df['dateRep'] = pd.to_datetime(full_df['dateRep'])
ukraine_df = full_df[full_df['countriesAndTerritories'] == 'Ukraine'].copy()
ukraine_df = ukraine_df.sort_values('dateRep')

print(f"\nДані з {ukraine_df['dateRep'].min().date()} по {ukraine_df['dateRep'].max().date()}\n")

periods = [
    ('2020-03-27', '2020-05-31', 'Весна 2020'),
    ('2020-06-01', '2020-08-31', 'Літо 2020'),
    ('2020-09-01', '2020-11-15', 'Осіння хвиля 2020'),
    ('2020-11-16', '2020-12-14', 'Зима 2020'),
]

N = 43993643  # Населення України

# --- Модель SEIR ---
def seir_model(y, t, beta, sigma, gamma):
    S, E, I, R = y
    N_total = S + E + I + R
    dSdt = -beta * S * I / N_total
    dEdt = beta * S * I / N_total - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

# --- Допоміжні функції ---
def calculate_metrics(y_true, y_pred):
    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    
    def mape(y_true, y_pred):
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else np.nan
    
    return {
        'RMSE': rmse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
        'MAPE (%)': mape(y_true, y_pred)
    }

# --- Основна логіка ---
results = []
all_predictions = []

for idx, (start_date, end_date, label) in enumerate(periods):
    print(f"\n{'='*50}\nОбробка періоду: {label}\n{'='*50}")
    
    # Підготовка даних
    period_data = ukraine_df[(ukraine_df['dateRep'] >= start_date) & (ukraine_df['dateRep'] <= end_date)]
    if len(period_data) < 10:
        print(f"⚠️ Недостатньо даних ({len(period_data)} днів)")
        continue
    
    dates = period_data['dateRep']
    cases = period_data['cases'].values
    deaths = period_data['deaths'].values
    
    # Коефіцієнт летальності (ковзне середнє за 14 днів)
    window = min(14, len(cases))
    f = np.sum(deaths[-window:]) / np.sum(cases[-window:]) if np.sum(cases[-window:]) > 0 else 0
    
    # Початкові умови
    if idx == 0:
        E0, I0, R0 = cases[0]*3, cases[0], 0
    else:
        E0, I0, R0 = results[-1]['E'][-1], results[-1]['I'][-1], results[-1]['R'][-1]
    
    # Оптимізація параметрів
    bounds = [(0.1, 0.6), (1/7, 1/3), (1/14, 1/10)]
    initial_guess = [0.3, 1/5.2, 1/12.39]
    
    res = minimize(
        lambda params: np.sum((cases - params[1]*odeint(
            seir_model, 
            [N-E0-I0-R0, E0, I0, R0], 
            np.arange(len(cases)), 
            args=tuple(params))[:,1])**2),
        initial_guess,
        bounds=bounds
    )
    
    if not res.success:
        print(f"❌ Оптимізація не вдалася для періоду {label}")
        continue
    
    beta, sigma, gamma = res.x
    
    # Розв'язання моделі
    solution = odeint(seir_model, [N-E0-I0-R0, E0, I0, R0], np.arange(len(dates)), args=(beta, sigma, gamma))
    S, E, I, R = solution.T
    
    # Прогнозовані значення
    new_cases_pred = sigma * E
    new_deaths_pred = f * new_cases_pred
    
    # Збереження результатів
    results.append({
        'Період': label,
        'Дата': dates,
        'S': S, 'E': E, 'I': I, 'R': R,
        'Справжні випадки': cases,
        'Прогноз випадків': new_cases_pred,
        'Справжні смерті': deaths,
        'Прогноз смертей': new_deaths_pred,
        'Параметри': {'β': beta, 'σ': sigma, 'γ': gamma, 'f': f}
    })
    
    all_predictions.append({
        'Період': label,
        'Дата': dates,
        'Тип': ['Випадки']*len(dates) + ['Смерті']*len(dates),
        'Справжні': np.concatenate([cases, deaths]),
        'Прогноз': np.concatenate([new_cases_pred, new_deaths_pred])
    })

# --- Візуалізація ---

# 1. Графік параметрів по періодах
plt.figure(figsize=(12, 6))
params_df = pd.DataFrame([{**{'Період': r['Період']}, **r['Параметри']} for r in results])
params_df.plot(x='Період', y=['β', 'σ', 'γ'], kind='bar', ax=plt.gca())
plt.title('Оптимальні параметри моделі по періодах')
plt.ylabel('Значення параметру')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Графік прогнозу vs реальних значень для всіх періодів
plt.figure(figsize=(12, 6))
for r in results:
    plt.plot(r['Дата'], r['Справжні випадки'], 'o-', label=f"{r['Період']} - спостереження")
    plt.plot(r['Дата'], r['Прогноз випадків'], 'x--', label=f"{r['Період']} - прогноз")
plt.title('Порівняння спостережуваних та прогнозованих випадків')
plt.xlabel('Дата')
plt.ylabel('Кількість випадків')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Графік смертей
plt.figure(figsize=(12, 6))
for r in results:
    plt.plot(r['Дата'], r['Справжні смерті'], 'o-', label=f"{r['Період']} - спостереження")
    plt.plot(r['Дата'], r['Прогноз смертей'], 'x--', label=f"{r['Період']} - прогноз")
plt.title('Порівняння спостережуваних та прогнозованих смертей')
plt.xlabel('Дата')
plt.ylabel('Кількість смертей')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Красиві таблиці ---
print("\n\n" + "="*80)
print("ОПТИМАЛЬНІ ПАРАМЕТРИ МОДЕЛІ".center(80))
print("="*80)
params_table = []
for r in results:
    params_table.append([
        r['Період'],
        f"{r['Параметри']['β']:.4f}",
        f"{r['Параметри']['σ']:.4f}",
        f"{r['Параметри']['γ']:.4f}",
        f"{r['Параметри']['f']:.4f}"
    ])
print(tabulate(params_table, 
               headers=['Період', 'β (контактна швидкість)', 'σ (інкубаційна)', 'γ (одужання)', 'f (летальність)'], 
               tablefmt='grid'))

print("\n\n" + "="*80)
print("МЕТРИКИ ЯКОСТІ ПРОГНОЗУ ВИПАДКІВ".center(80))
print("="*80)
cases_metrics = []
for r in results:
    metrics = calculate_metrics(r['Справжні випадки'], r['Прогноз випадків'])
    cases_metrics.append([
        r['Період'],
        f"{metrics['RMSE']:.1f}",
        f"{metrics['MAE']:.1f}",
        f"{metrics['MAPE (%)']:.1f}%"
    ])
print(tabulate(cases_metrics, 
               headers=['Період', 'RMSE', 'MAE', 'MAPE (%)'], 
               tablefmt='grid'))

print("\n\n" + "="*80)
print("МЕТРИКИ ЯКОСТІ ПРОГНОЗУ СМЕРТЕЙ".center(80))
print("="*80)
deaths_metrics = []
for r in results:
    metrics = calculate_metrics(r['Справжні смерті'], r['Прогноз смертей'])
    deaths_metrics.append([
        r['Період'],
        f"{metrics['RMSE']:.1f}",
        f"{metrics['MAE']:.1f}",
        f"{metrics['MAPE (%)']:.1f}%"
    ])
print(tabulate(deaths_metrics, 
               headers=['Період', 'RMSE', 'MAE', 'MAPE (%)'], 
               tablefmt='grid'))