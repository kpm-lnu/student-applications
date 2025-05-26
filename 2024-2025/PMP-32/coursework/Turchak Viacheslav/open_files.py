import numpy as np
data = np.load("crs_matrix_100x100.npz")
print(data.files)
for i in data["values"]:
    print(i , end = " , ")