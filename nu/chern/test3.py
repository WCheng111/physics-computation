import numpy as np
import matplotlib.pyplot as plt

# 定义网格范围和步长
x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)

# 定义不同的 a 值
a_values = [1, 4, 8, 12]

# 计算所有图像的最大最小值，以确定统一的颜色范围
z_min = min(np.min(a * (X**2 + Y**2)) for a in a_values)
z_max = max(np.max(a * (X**2 + Y**2)) for a in a_values)

# 创建子图布局
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs = axs.ravel()

# 循环绘制每个图像
for i, a in enumerate(a_values):
    Z = a * (X**2 + Y**2)
    im = axs[i].imshow(Z, cmap='viridis', extent=(-5, 5, -5, 5), vmin=z_min, vmax=z_max)
    axs[i].set_title(f'a = {a}')
    fig.colorbar(im, ax=axs[i])

plt.tight_layout()
plt.show()

