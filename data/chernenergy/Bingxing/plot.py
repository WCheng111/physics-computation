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





chern_change=np.load('Berryphase(A3=-A,soc=40,integrate=400,mu=-25__25,).npy')
sorted_indices = np.argsort(np.abs(chern_change-0.5))[:4]
sorted_indices2=np.argsort(np.abs(chern_change+0.5))[:3]
mu=np.linspace(-25,25,len(chern_change))
print(sorted_indices)
for i in range(len(sorted_indices)):
    print(mu[sorted_indices[i]])
for i in range(len(sorted_indices2)):
    print(mu[sorted_indices2[i]])
# mu=np.linspace(-30,30,len(chern_change))
plt.plot(mu,2*chern_change)
print(chern_change)
plt.axvline(mu[sorted_indices[0]],color='k', linestyle='--',label='x=1')
plt.axvline(mu[sorted_indices[1]],color='k', linestyle='--')
# plt.axvline(mu[sorted_indices[2]],color='k', linestyle='--')
# plt.axvline(mu[sorted_indices[4]],color='k', linestyle='--')
plt.axvline(mu[sorted_indices2[0]],color='k', linestyle='--',label='x=1')
plt.axvline(mu[sorted_indices2[2]],color='k', linestyle='--')
# plt.axvline(mu[sorted_indices[2]],color='k', linestyle='--')
# plt.axvline(mu[sorted_indices[4]],color='k', linestyle='--')
# plt.text(round(mu[sorted_indices[0]],1), 1.2,round(mu[sorted_indices[0]],1), rotation=0, color='black',fontsize=15)
# plt.text(round(mu[sorted_indices[1]],1)-8, 1.2,round(mu[sorted_indices[1]],1), rotation=0, color='black',fontsize=15)
print(mu[sorted_indices[0]],mu[sorted_indices[1]])
plt.title("SOC=40,Integrate=400")
plt.axhline(1, color='k', linestyle='--')
plt.axhline(-1, color='k', linestyle='--')
plt.xlabel(r'$\mu$')
plt.ylabel("occup states phase")
plt.show()

# 定义 x 范围
# x = np.linspace(-60,60,len(chern_change1))




# fig, axs = plt.subplots(2, 3, figsize=(12, 8))
#
# # 绘制上面三个图
# axs[0, 0].plot(x, 2*chern_change1)
# axs[0, 0].set_title('SOC=10')
# axs[0,0].axhline(1, color='k', linestyle='--')
# axs[0,0].axhline(-1, color='k', linestyle='--')
# axs[0, 0].text(-20, 0, r'$a < 4$', fontsize=12)
# axs[0,0].set_xlabel(r'$\mu$')
# axs[0,0].set_ylabel("Fermi surface Berry phase")
#
#
# axs[0, 1].plot(x, 2*chern_change2)
# axs[0, 1].set_title('SOC=20')
# axs[0,1].axhline(1, color='k', linestyle='--')
# axs[0,1].axhline(-1, color='k', linestyle='--')
# axs[0,1].set_xlabel(r'$\mu$')
# axs[0,1].set_ylabel("Fermi surface Berry phase")
#
# axs[0, 2].plot(x, 2*chern_change3)
# axs[0, 2].set_title('SOC=40')
# axs[0,2].axhline(1, color='k', linestyle='--')
# axs[0,2].axhline(-1, color='k', linestyle='--')
# axs[0,2].set_xlabel(r'$\mu$')
# axs[0,2].set_ylabel("Fermi surface Berry phase")
#
#
# # 绘制下面三个图
# axs[1, 0].plot(x, 2*chern_change4)
# axs[1, 0].set_title('SOC=10')
# axs[1,0].axhline(1, color='k', linestyle='--')
# axs[1,0].axhline(-1, color='k', linestyle='--')
# axs[1,0].set_xlabel(r'$\mu$')
# axs[1,0].set_ylabel("Fermi surface Berry phase")
#
#
#
# axs[1, 1].plot(x, 2*chern_change5)
# axs[1, 1].set_title('SOC=20')
# axs[1,1].axhline(1, color='k', linestyle='--')
# axs[1,1].axhline(-1, color='k', linestyle='--')
# axs[1,1].set_xlabel(r'$\mu$')
# axs[1,1].set_ylabel("Fermi surface Berry phase")
#
#
#
#
# axs[1, 2].plot(x,2*chern_change6)
# axs[1, 2].set_title('SOC=40')
# axs[1,2].axhline(1, color='k', linestyle='--')
# axs[1,2].axhline(-1, color='k', linestyle='--')
# axs[1,2].set_xlabel(r'$\mu$')
# axs[1,2].set_ylabel("Fermi surface Berry phase")
#
# fig.text(-10, 0, 'Additional Text', fontsize=20)
#
# # 调整布局
# plt.tight_layout()
# plt.subplots_adjust(hspace=0.5, wspace=0.3)
# # 显示图形
# plt.show()
