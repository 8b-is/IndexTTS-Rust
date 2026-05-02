import numpy as np
def snake(x, alpha):
    return x + np.sin(alpha * x)**2 / alpha

x = np.linspace(-5, 5, 1000)
y = snake(x, 0.3)
import matplotlib.pyplot as plt
plt.plot(x, y)
plt.savefig('snake.png')
