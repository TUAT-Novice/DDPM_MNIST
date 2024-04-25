import numpy as np


x = np.array([i for i in range(500)])
print(x.shape)
x = x[::20]
print(x.shape)
