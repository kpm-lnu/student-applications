import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

N_VARS = 10 
(i_S_H, i_I_H, i_R_H, i_I_L, i_R_L, i_S_R, i_I_R, i_S_A, i_I_A, i_W) = range(N_VARS)

DEFAULT_PARAMS = {
    # --- Параметри Людей (H) ---
    'gamma_H': 1 / 14,      # Швидкість одужання від хантавірусу 
    'gamma_L': 1 / 10,      # Швидкість одужання від лептоспірозу 
    
    'omega_H': 1 / 180, # Швидкість втрати імунітету до хантавірусу 
    'omega_L': 1 / 180, # Швидкість втрати імунітету до лептоспірозу 

    # --- Параметри Резервуару Хантавірусу (Гризуни, R) ---
    'base_beta_R': 0.008,   # Базовий коеф. передачі 
    'base_beta_H': 0.001,   # Базовий коеф. передачі 
    'base_Lambda_R': 2.0,   # Базова народжуваність гризунів 
    'base_mu_R': 0.08,      # Базова природна смертність гризунів 

    # --- Параметри Резервуару Лептоспірозу (Тварини A, Середовище W) ---
    'base_beta_A': 0.01,    # Базовий коеф. передачі 
    'base_beta_L': 0.005,   # Базовий коеф. передачі 
    'base_beta_AL': 0.0001, # Базовий коеф. передачі 
    'base_Lambda_A': 5.0,  # Базова народжуваність тварин 
    'base_mu_A': 0.08,      # Базова природна смертність тварин
    'gamma_A': 1 / 90,      # Швидкість одужання тварин 
    'alpha_A': 0.1,         # Швидкість виділення бактерій у середовище інфікованими тваринами
    'base_delta_W': 0.03,   # Базова швидкість загибелі бактерій у середовищі 
    
    # --- Параметри Міграції ---
    'm_R': 0.00001,           # Коеф. міграції гризунів (S_R, I_R) між областями
    'm_A': 0.000001,          # Коеф. міграції тварин (S_A, I_A) між областями
}


def _model_equations(t, y, climate_data, params, neighbors_indexed, oblast_order, n_regions):
    
    state = y.reshape((n_regions, N_VARS))
    dydt = np.zeros_like(state)

    current_day_of_year = int(t % 365) + 1 
    
    try:
        climate_today = climate_data[climate_data['day_of_year'] == current_day_of_year]
    except Exception as e:
        print(f"Помилка отримання клімату: t={t}, day={current_day_of_year}, {e}")
        return np.zeros_like(y) 

    for i in range(n_regions):
        
        S_H, I_H, R_H, I_L, R_L, S_R, I_R, S_A, I_A, W = state[i]
        
        region_name = oblast_order[i]
        climate_region = climate_today[climate_today['oblast'] == region_name]
        
        if climate_region.empty:
            climate_region = climate_today.iloc[0]
            
        temp = climate_region['temperature_avg'].values[0]
        humidity = climate_region['humidity_avg'].values[0]
        precip = climate_region['precipitation_sum'].values[0]

        # Розрахунок динамічних (кліматичних) параметрів
        Lambda_R = params['base_Lambda_R'] * max(0.5, 1 + (temp - 10) / 30) 
        mu_R = params['base_mu_R'] * max(0.8, 1 - (temp - 10) / 40)      
        Lambda_A = params['base_Lambda_A'] * max(0.5, 1 + (temp - 10) / 30) 
        mu_A = params['base_mu_A'] * max(0.8, 1 - (temp - 10) / 40)         
        
        delta_W = params['base_delta_W'] * max(0.1, 1 + (temp - 15) / 20)
        beta_L = params['base_beta_L'] * (1 + precip / 10) 
        beta_A = params['base_beta_A'] * (1 + precip / 10) 
        beta_H = params['base_beta_H'] * (1 + humidity / 100) 
        beta_R = params['base_beta_R'] * (1 + humidity / 100) 
        beta_AL = params['base_beta_AL'] * (1 + humidity / 100) 

        # Розрахунок сил інфекції
        lambda_H = beta_H * I_R
        lambda_L = (beta_L * W) + (beta_AL * I_A)
        
        mig_S_R, mig_I_R, mig_S_A, mig_I_A = 0, 0, 0, 0
        for j in neighbors_indexed[i]:
            mig_S_R += params['m_R'] * state[j, i_S_R]
            mig_I_R += params['m_R'] * state[j, i_I_R]
            mig_S_A += params['m_A'] * state[j, i_S_A]
            mig_I_A += params['m_A'] * state[j, i_I_A]
        num_neighbors = len(neighbors_indexed[i])
        mig_S_R -= num_neighbors * params['m_R'] * S_R
        mig_I_R -= num_neighbors * params['m_R'] * I_R
        mig_S_A -= num_neighbors * params['m_A'] * S_A
        mig_I_A -= num_neighbors * params['m_A'] * I_A
        
        S_H_norm = max(0, 1 - I_H - R_H - I_L - R_L) 
        
        dS_H = -lambda_H * S_H_norm - lambda_L * S_H_norm + params['omega_H'] * R_H + params['omega_L'] * R_L
        
        dI_H = lambda_H * S_H_norm - params['gamma_H'] * I_H
        
        dR_H = params['gamma_H'] * I_H - params['omega_H'] * R_H
        
        dI_L = lambda_L * S_H_norm - params['gamma_L'] * I_L
        
        dR_L = params['gamma_L'] * I_L - params['omega_L'] * R_L
        
        dS_R = Lambda_R - beta_R * S_R * I_R - mu_R * S_R + mig_S_R
        dI_R = beta_R * S_R * I_R - mu_R * I_R + mig_I_R
        dS_A = Lambda_A - beta_A * S_A * W - mu_A * S_A + mig_S_A
        dI_A = beta_A * S_A * W - (mu_A + params['gamma_A']) * I_A + mig_I_A
        dW = params['alpha_A'] * I_A - delta_W * W
        
        dydt[i] = [dS_H, dI_H, dR_H, dI_L, dR_L, dS_R, dI_R, dS_A, dI_A, dW]

    return dydt.flatten()

def load_simulation_data():

    print("[SimEngine] Завантаження даних...")
    
    try:
        climate_df = pd.read_csv('climate_data_2022_v2.csv')
        climate_df['date'] = pd.to_datetime(climate_df['date'])
        climate_df['day_of_year'] = climate_df['date'].dt.dayofyear
        print("  > Клімат OK")
    except FileNotFoundError:
        print("[ПОМИЛКА] Файл 'climate_data_2022_v2.csv' не знайдено.")
        return None

    population_data = {
        'Вінницька область': 1509515, 'Волинська область': 1021356,
        'Дніпропетровська область': 3096485, 'Донецька область': 4059372,
        'Житомирська область': 1179032, 'Закарпатська область': 1244476,
        'Запорізька область': 1638462, 'Івано-Франківська область': 1351822,
        'Київська область': 1795079, 'Кіровоградська область': 903712,
        'Луганська область': 2102921, 'Львівська область': 2478133,
        'Миколаївська область': 1091821, 'Одеська область': 2351392,
        'Полтавська область': 1352283, 'Рівненська область': 1141784,
        'Сумська область': 1035772, 'Тернопільська область': 1021713,
        'Харківська область': 2598961, 'Херсонська область': 1001598,
        'Хмельницька область': 1228829, 'Черкаська область': 1160744,
        'Чернівецька область': 890457, 'Чернігівська область': 959315,
        'Автономна Республіка Крим': 1963770, 'м. Севастополь': 509992,
        'м. Київ': 2952301
    }
    print("  > Демографія OK")

    neighbors_data = {
        'Вінницька область': ['Житомирська область', 'Київська область', 'Черкаська область', 'Кіровоградська область', 'Одеська область', 'Хмельницька область'],
        'Волинська область': ['Львівська область', 'Рівненська область'],
        'Дніпропетровська область': ['Полтавська область', 'Харківська область', 'Донецька область', 'Запорізька область', 'Херсонська область', 'Миколаївська область', 'Кіровоградська область'],
        'Донецька область': ['Дніпропетровська область', 'Запорізька область', 'Луганська область'],
        'Житомирська область': ['Київська область', 'Вінницька область', 'Хмельницька область', 'Рівненська область'],
        'Закарпатська область': ['Львівська область', 'Івано-Франківська область'],
        'Запорізька область': ['Дніпропетровська область', 'Донецька область', 'Херсонська область'],
        'Івано-Франківська область': ['Львівська область', 'Тернопільська область', 'Чернівецька область', 'Закарпатська область'],
        'Київська область': ['Житомирська область', 'Вінницька область', 'Черкаська область', 'Полтавська область', 'Чернігівська область', 'м. Київ'],
        'Кіровоградська область': ['Вінницька область', 'Черкаська область', 'Полтавська область', 'Дніпропетровська область', 'Миколаївська область', 'Одеська область'],
        'Луганська область': ['Харківська область', 'Донецька область'],
        'Львівська область': ['Волинська область', 'Рівненська область', 'Тернопільська область', 'Івано-Франківська область', 'Закарпатська область'],
        'Миколаївська область': ['Одеська область', 'Кіровоградська область', 'Дніпропетровська область', 'Херсонська область'],
        'Одеська область': ['Вінницька область', 'Кіровоградська область', 'Миколаївська область'],
        'Полтавська область': ['Київська область', 'Черкаська область', 'Кіровоградська область', 'Дніпропетровська область', 'Харківська область', 'Сумська область'],
        'Рівненська область': ['Волинська область', 'Львівська область', 'Тернопільська область', 'Хмельницька область', 'Житомирська область'],
        'Сумська область': ['Чернігівська область', 'Полтавська область', 'Харківська область'],
        'Тернопільська область': ['Львівська область', 'Рівненська область', 'Хмельницька область', 'Чернівецька область', 'Івано-Франківська область'],
        'Харківська область': ['Сумська область', 'Полтавська область', 'Дніпропетровська область', 'Луганська область'],
        'Херсонська область': ['Миколаївська область', 'Дніпропетровська область', 'Запорізька область', 'Автономна Республіка Крим'],
        'Хмельницька область': ['Житомирська область', 'Вінницька область', 'Черкаська область', 'Тернопільська область', 'Рівненська область'],
        'Черкаська область': ['Київська область', 'Полтавська область', 'Кіровоградська область', 'Вінницька область', 'Хмельницька область'],
        'Чернівецька область': ['Івано-Франківська область', 'Тернопільська область'],
        'Чернігівська область': ['Київська область', 'Сумська область'],
        'Автономна Республіка Крим': ['Херсонська область', 'м. Севастополь'],
        'м. Севастополь': ['Автономна Республіка Крим'],
        'м. Київ': ['Київська область']
    }
    print("  > Сусіди OK")

    oblast_order = sorted(list(population_data.keys()))
    n_regions = len(oblast_order)
    oblast_to_index = {name: i for i, name in enumerate(oblast_order)}
    
    neighbors_indexed = []
    for oblast_name in oblast_order:
        neighbor_names = neighbors_data.get(oblast_name, [])
        neighbor_indices = [oblast_to_index[name] for name in neighbor_names]
        neighbors_indexed.append(neighbor_indices)
    print("  > Індексація OK")

    return {
        "climate_df": climate_df,
        "population_data": population_data,
        "oblast_order": oblast_order,
        "n_regions": n_regions,
        "oblast_to_index": oblast_to_index,
        "neighbors_indexed": neighbors_indexed
    }

def create_initial_conditions(n_regions):

    y0_2d = np.zeros((n_regions, N_VARS))
    y0_2d[:, i_S_H] = 1.0  
    
    y0_2d[:, i_S_R] = DEFAULT_PARAMS['base_Lambda_R'] / DEFAULT_PARAMS['base_mu_R'] # ~100
    y0_2d[:, i_S_A] = DEFAULT_PARAMS['base_Lambda_A'] / DEFAULT_PARAMS['base_mu_A'] # ~500
    
    return y0_2d

def run_simulation(start_day, simulation_days, y0_2d, simulation_data, params):

    print(f"[SimEngine] Запуск симуляції на {simulation_days} днів...")
    
    t_start = start_day
    t_end = simulation_days + start_day
    t_eval_points = np.arange(t_start, t_end + 1, 1) 
    
    y0_1d = y0_2d.flatten()
    
    climate_df = simulation_data['climate_df']
    neighbors_indexed = simulation_data['neighbors_indexed']
    oblast_order = simulation_data['oblast_order']
    n_regions = simulation_data['n_regions']
    
    sol = solve_ivp(
        fun=_model_equations,
        t_span=(t_start, t_end),
        y0=y0_1d,
        t_eval=t_eval_points,
        args=(climate_df, params, neighbors_indexed, oblast_order, n_regions),
        method='RK45' 
    )

    if not sol.success:
        print(f"[SimEngine ПОМИЛКА] Симуляція не вдалася: {sol.message}")
        return None, None

    print(f"[SimEngine] Симуляція завершена успішно.")
    
    result_3d = sol.y.reshape((n_regions, N_VARS, len(t_eval_points)))
    
    return result_3d, t_eval_points