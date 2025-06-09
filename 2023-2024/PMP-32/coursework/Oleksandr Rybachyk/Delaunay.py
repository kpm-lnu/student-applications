import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import time
import triangle as tr

# Згенеруємо випадковий набір точок для прикладу
np.random.seed(42)
points = np.random.rand(1000000, 2)

start_time = time.time()  # Початок вимірювання часу

# Виконуємо триангуляцію Делоне
triangulation = Delaunay(points)

end_time = time.time()  # Кінець вимірювання часу
execution_time = end_time - start_time
print("Час виконання програми: {:.5f} секунд".format(execution_time))

# Виводимо трикутники
plt.triplot(points[:, 0], points[:, 1], triangulation.simplices, c='r')
plt.plot(points[:, 0], points[:, 1], 'o')
plt.title('Триангуляція Делоне')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


