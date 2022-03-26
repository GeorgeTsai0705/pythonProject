import math
import matplotlib.pyplot as plt

x = list(range(200))
y = [math.sin(i/50) for i in x]
z = [math.cos(i/50) for i in x]
plt.plot(x, y, color="r")
plt.plot(x, z, color ="b")
plt.show()