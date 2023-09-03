import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建一个网格
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# 计算三个函数的Z值
Z1 = X**2 + Y**2
Z2 = 2 * (X**2 + Y**2)
Z3 = 4 * (X**2 + Y**2)

# 创建一个图形，分成三个子图
# fig = plt.figure()
fig = plt.figure(figsize=(10, 12))

# 子图1
ax1 = fig.add_subplot(311, projection='3d')
ax1.plot_surface(X, Y, Z1, cmap='viridis')
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# ax1.set_zlabel('Z')
# ax1.set_title('z=x^2+y^2')

# 子图2
ax2 = fig.add_subplot(312, projection='3d')
ax2.plot_surface(X, Y, Z2, cmap='plasma')
# ax2.set_xlabel('X')
# ax2.set_ylabel('Y')
# ax2.set_zlabel('Z')
# ax2.set_title('z=2(x^2+y^2)')

# 子图3
ax3 = fig.add_subplot(313, projection='3d')
ax3.plot_surface(X, Y, Z3, cmap='inferno')
# ax3.set_xlabel('X')
# ax3.set_ylabel('Y')
# ax3.set_zlabel('Z')
# ax3.set_title('z=4(x^2+y^2)')

# 调整子图之间的垂直间距
plt.subplots_adjust(hspace=0.01)

# 显示图形
plt.show()

