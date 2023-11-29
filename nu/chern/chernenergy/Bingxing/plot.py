import numpy as np
import matplotlib.pyplot as plt
import math
import time



chern_change1=np.load('Berryphase(A3=A,soc=10).npy')
chern_change2=np.load('Berryphase(A3=A,soc=20).npy')
chern_change3=np.load('Berryphase(A3=A,soc=40).npy')
chern_change4=np.load('Berryphase(A3=-A,soc=10).npy')
chern_change5=np.load('Berryphase(A3=-A,soc=20).npy')
chern_change6=np.load('Berryphase(A3=-A,soc=40).npy')
# mu=np.linspace(-60,60,len(chern_change))
# plt.plot(mu,chern_change)
# plt.title("SOC=20")
# plt.axhline(0.5, color='k', linestyle='--')
# plt.axhline(-0.5, color='k', linestyle='--')
# plt.xlabel(r'$\mu$')
# plt.ylabel("occup states phase")
# plt.show()
import matplotlib.pyplot as plt
import numpy as np

# 定义 x 范围
x = np.linspace(-60,60,len(chern_change1))




fig, axs = plt.subplots(2, 3, figsize=(12, 8))

# 绘制上面三个图
axs[0, 0].plot(x, 2*chern_change1)
axs[0, 0].set_title('SOC=10')
axs[0,0].axhline(1, color='k', linestyle='--')
axs[0,0].axhline(-1, color='k', linestyle='--')
axs[0, 0].text(-20, 0, r'$a < 4$', fontsize=12)
axs[0,0].set_xlabel(r'$\mu$')
axs[0,0].set_ylabel("Fermi surface Berry phase")


axs[0, 1].plot(x, 2*chern_change2)
axs[0, 1].set_title('SOC=20')
axs[0,1].axhline(1, color='k', linestyle='--')
axs[0,1].axhline(-1, color='k', linestyle='--')
axs[0,1].set_xlabel(r'$\mu$')
axs[0,1].set_ylabel("Fermi surface Berry phase")

axs[0, 2].plot(x, 2*chern_change3)
axs[0, 2].set_title('SOC=40')
axs[0,2].axhline(1, color='k', linestyle='--')
axs[0,2].axhline(-1, color='k', linestyle='--')
axs[0,2].set_xlabel(r'$\mu$')
axs[0,2].set_ylabel("Fermi surface Berry phase")


# 绘制下面三个图
axs[1, 0].plot(x, 2*chern_change4)
axs[1, 0].set_title('SOC=10')
axs[1,0].axhline(1, color='k', linestyle='--')
axs[1,0].axhline(-1, color='k', linestyle='--')
axs[1,0].set_xlabel(r'$\mu$')
axs[1,0].set_ylabel("Fermi surface Berry phase")



axs[1, 1].plot(x, 2*chern_change5)
axs[1, 1].set_title('SOC=20')
axs[1,1].axhline(1, color='k', linestyle='--')
axs[1,1].axhline(-1, color='k', linestyle='--')
axs[1,1].set_xlabel(r'$\mu$')
axs[1,1].set_ylabel("Fermi surface Berry phase")




axs[1, 2].plot(x,2*chern_change6)
axs[1, 2].set_title('SOC=40')
axs[1,2].axhline(1, color='k', linestyle='--')
axs[1,2].axhline(-1, color='k', linestyle='--')
axs[1,2].set_xlabel(r'$\mu$')
axs[1,2].set_ylabel("Fermi surface Berry phase")

fig.text(-10, 0, 'Additional Text', fontsize=20)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(hspace=0.5, wspace=0.3)
# 显示图形
plt.show()
