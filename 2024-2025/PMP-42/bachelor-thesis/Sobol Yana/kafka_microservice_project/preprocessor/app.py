import json
import time
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers='kafka:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def send_from_file(filename: str):
    try:
        with open(filename, 'r') as f:
            data_list = json.load(f)
    except Exception as e:
        print(f" Помилка зчитування JSON-файлу: {e}")
        return

    for item in data_list:
        try:
            producer.send('fem_input', value=item)
            producer.flush()
            print(" Надіслано:", item)
            time.sleep(2) 
        except Exception as e:
            print(f" Помилка надсилання: {e}")

if __name__ == "__main__":
    print("Preprocessor: надсилаємо дані з JSON-файлу...")
    send_from_file('input_data.json')
