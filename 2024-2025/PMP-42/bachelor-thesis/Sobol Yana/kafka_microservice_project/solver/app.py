from kafka import KafkaConsumer, KafkaProducer
import numpy as np
import json

print("Solver: Ініціалізація Kafka Consumer та Producer...")

consumer = KafkaConsumer(
    'fem_input',
    bootstrap_servers='kafka:9092',
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest',  
    enable_auto_commit=True
)

producer = KafkaProducer(
    bootstrap_servers='kafka:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def solve_fem(id, L, EA, b, u0, F, n_elements):
    print("Solver: Обчислюємо FEM розв'язок...")
    n_nodes = n_elements + 1
    h = L / n_elements
    K = np.zeros((n_nodes, n_nodes))
    F_vec = np.zeros(n_nodes)

    for e in range(n_elements):
        ke = (EA / h) * np.array([[1, -1], [-1, 1]])
        fe = (b * h / 2) * np.array([1, 1])
        K[e:e+2, e:e+2] += ke
        F_vec[e:e+2] += fe

    K[0, :] = 0
    K[0, 0] = 1
    F_vec[0] = u0
    F_vec[-1] += F

    u = np.linalg.solve(K, F_vec)
    print("Solver: Розв'язок FEM успішно обчислений.")
    return u.tolist()

print("Solver: Очікуємо на повідомлення з топіка 'fem_input'...")

for msg in consumer:
    try:
        data = msg.value
        print(f"Solver: Отримано дані: {data}")

        result = solve_fem(**data)

        output_message = {"id": data["id"], 'u': result}
        producer.send('fem_output', output_message)
        producer.flush()
        print(f"Solver: Розв'язок надіслано: {output_message}")
    except Exception as e:
        print("Solver: Помилка при обробці повідомлення:", e)


