from kafka import KafkaConsumer
import json
import matplotlib.pyplot as plt
import numpy as np
import threading
import time


params_map = {}
solution_map = {}

def compute_exact_solution(x, L, EA, b, F, u0):
    return -(b / (2 * EA)) * x**2 + ((F + b * L) / EA) * x + u0

def maybe_plot(request_id):
    
    if request_id in params_map and request_id in solution_map:
        params = params_map[request_id]
        u = solution_map[request_id]

        print(f"Postprocessor: Побудова графіку для id={request_id}...")

        # Отримання параметрів
        L = params['L']
        EA = params['EA']
        b = params['b']
        F = params['F']
        u0 = params['u0']

        # Побудова графіка
        x_dense = np.linspace(0, L, 200)
        x = np.linspace(0, L, len(u))
        exact = compute_exact_solution(x_dense, L, EA, b, F, u0)

        plt.figure()
        plt.plot(x, u, 'o-', label='FEM')
        plt.plot(x_dense, exact, 'r--', label='Exact')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.title(f'FEM vs Exact Solution (id={request_id})')
        plt.grid(True)
        plt.legend()
        filename = f"result_{request_id}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Postprocessor: Графік збережено як {filename}")

def listen_input():
    print("Postprocessor: Слухаємо топік 'fem_input'...")
    consumer = KafkaConsumer(
        'fem_input',
        bootstrap_servers='kafka:9092',
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest',
        enable_auto_commit=True
    )
    for msg in consumer:
        data = msg.value
        request_id = data["id"]
        params_map[request_id] = data
        print(f"Postprocessor: Отримано параметри для id={request_id}")
        maybe_plot(request_id)

def listen_output():
    print("Postprocessor: Слухаємо топік 'fem_output'...")
    consumer = KafkaConsumer(
        'fem_output',
        bootstrap_servers='kafka:9092',
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest',
        enable_auto_commit=True
    )
    for msg in consumer:
        data = msg.value
        request_id = data["id"]
        u = np.array(data["u"])
        solution_map[request_id] = u
        print(f"Postprocessor: Отримано FEM-розв’язок для id={request_id}")
        maybe_plot(request_id)

if __name__ == "__main__":
    print("Postprocessor: Запуск...")

    # Запуск потоків слухання Kafka
    threading.Thread(target=listen_input, daemon=True).start()
    threading.Thread(target=listen_output, daemon=True).start()

    
    while True:
        time.sleep(1)
