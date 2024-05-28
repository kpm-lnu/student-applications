import tensorflow as tf
import numpy as np
import time

image = np.array([[i for i in range(1, 10001)] for _ in range(10000)])

image_tensor = tf.constant(image, dtype=tf.float32)

@tf.function
def compute_variance_static(image_tensor):
    mean_value_static = tf.reduce_mean(image_tensor)
    variance_static = tf.reduce_mean(tf.square(image_tensor - mean_value_static))
    return mean_value_static, variance_static

def compute_variance_dynamic(image_tensor):
    mean_value_dynamic = tf.reduce_mean(image_tensor)
    variance_dynamic = tf.reduce_mean(tf.square(image_tensor - mean_value_dynamic))
    return mean_value_dynamic, variance_dynamic

start_time_static = time.time()
mean_static, variance_static = compute_variance_static(image_tensor)
end_time_static = time.time()
computation_time_static = end_time_static - start_time_static

start_time_dynamic = time.time()
mean_dynamic, variance_dynamic = compute_variance_dynamic(image_tensor)
end_time_dynamic = time.time()
computation_time_dynamic = end_time_dynamic - start_time_dynamic

print("Середнє значення (статична побудова графа):", mean_static.numpy())
print("Дисперсія (статична побудова графа):", variance_static.numpy())
print("Час обчислення (статична побудова графа):", computation_time_static, "секунд")

print("Середнє значення (динамічна побудова графа):", mean_dynamic.numpy())
print("Дисперсія (динамічна побудова графа):", variance_dynamic.numpy())
print("Час обчислення (динамічна побудова графа):", computation_time_dynamic, "секунд")