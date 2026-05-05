import pandas as pd
from datetime import datetime
from meteostat import Hourly, Stations
import time

start_date = datetime(2022, 1, 1)
end_date = datetime(2022, 12, 31)

output_filename = 'climate_data_2022_v2.csv'

OBLAST_COORDINATES = {
    'Вінницька область': (49.2331, 28.4682),
    'Волинська область': (50.7472, 25.3254),
    'Дніпропетровська область': (48.4647, 35.0462),
    'Донецька область': (48.0159, 37.8028),
    'Житомирська область': (50.2547, 28.6587),
    'Закарпатська область': (48.6208, 22.2879),
    'Запорізька область': (47.8388, 35.1396),
    'Івано-Франківська область': (48.9226, 24.7111),
    'Київська область': (50.4501, 30.5234),
    'Кіровоградська область': (48.5079, 32.2623),
    'Луганська область': (48.5740, 39.3078),
    'Львівська область': (49.8397, 24.0297),
    'Миколаївська область': (46.9750, 31.9946),
    'Одеська область': (46.4825, 30.7233),
    'Полтавська область': (49.5883, 34.5514),
    'Рівненська область': (50.6199, 26.2516),
    'Сумська область': (50.9077, 34.7981),
    'Тернопільська область': (49.5535, 25.5948),
    'Харківська область': (49.9935, 36.2304),
    'Херсонська область': (46.6354, 32.6169),
    'Хмельницька область': (49.4230, 26.9871),
    'Черкаська область': (49.4444, 32.0598),
    'Чернівецька область': (48.2917, 25.9358),
    'Чернігівська область': (51.4982, 31.2893),
    'Автономна Республіка Крим': (44.9521, 34.1024),
    'м. Севастополь': (44.6167, 33.5254),
    'м. Київ': (50.4501, 30.5234)
}

all_data_frames = []
MAX_STATIONS_TO_TRY = 10 

print(f"Починаю збір даних за період з {start_date.date()} по {end_date.date()}...")

for oblast_name, (lat, lon) in OBLAST_COORDINATES.items():
    print(f"\n--- Обробка: '{oblast_name}' ---")
    try:
        stations = Stations().nearby(lat, lon)
        stations_to_try = stations.fetch(MAX_STATIONS_TO_TRY)
        
        if stations_to_try.empty:
            print(f"[ПОМИЛКА] Для '{oblast_name}' не знайдено ЖОДНОЇ метеостанції.")
            continue

        data_hourly = None
        
        for station_id in stations_to_try.index:
            print(f"  > Спроба зі станцією {station_id}...")
            
            data_hourly_check = Hourly(station_id, start_date, end_date)
            data_hourly_check = data_hourly_check.fetch()
            
            if not data_hourly_check.empty and len(data_hourly_check) > (365 * 24 / 2):
                print(f"[УСПІХ] Знайдено дані на станції: {station_id}")
                data_hourly = data_hourly_check
                break 
            else:
                print(f"  > Станція {station_id} порожня або має замало даних.")
        
        if data_hourly is None or data_hourly.empty:
            print(f"[ПОМИЛКА] Не вдалося знайти дані для '{oblast_name}' на жодній з {MAX_STATIONS_TO_TRY} станцій.")
            continue

        data_daily = data_hourly.resample('D').agg({
            'temp': 'mean',
            'rhum': 'mean',
            'prcp': 'sum'
        })
        
        data_daily['oblast'] = oblast_name
        all_data_frames.append(data_daily)
        
        time.sleep(0.5) 

    except Exception as e:
        print(f"[КРИТИЧНА ПОМИЛКА] Не вдалося обробити '{oblast_name}': {e}")


if all_data_frames:
    print("\nОб'єднання всіх даних в один файл...")
    
    final_df = pd.concat(all_data_frames)
    
    final_df.reset_index(names='date', inplace=True)
    
    final_df.rename(columns={
        'temp': 'temperature_avg',
        'rhum': 'humidity_avg',
        'prcp': 'precipitation_sum'
    }, inplace=True)
    
    print("Очищення даних (заповнення пропусків всередині груп)...")
    final_df['temperature_avg'] = final_df.groupby('oblast')['temperature_avg'].ffill().bfill()
    final_df['humidity_avg'] = final_df.groupby('oblast')['humidity_avg'].ffill().bfill()
    final_df['precipitation_sum'] = final_df.groupby('oblast')['precipitation_sum'].ffill().bfill()

    final_df.fillna(0, inplace=True)
    
    final_df.to_csv(output_filename, index=False, encoding='utf-8')
    print(f"\n[УСПІХ] Всі дані успішно збережено у файл: {output_filename}")
    print(f"Кінцева кількість рядків: {len(final_df)} (має бути 9855)")
else:
    print("\n[ПОМИЛКА] Не вдалося зібрати жодних даних.")