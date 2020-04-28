import numpy as np
import matplotlib.pyplot as plt
from models import SSH

plt.rcParams['figure.dpi'] = 100  # dpi of the pictures


def wave_func_on_ep_a(t1l, t1r, Nl, Nr, psi_a_1):
    psi_al = np.abs(psi_a_1 * t1l ** (np.arange(1, Nl + 1) - 1))
    psi_ar = np.abs(psi_a_1 * (1 / t1r) ** (Nl + Nr + 1 - np.arange(Nl + 1, Nl + Nr + 1)))
    return np.append(psi_al, psi_ar)


def wave_func_on_ep_b(t1l, t1r, Nl, Nr, psi_b_Nl):
    psi_bl = np.abs(psi_b_Nl * t1l ** (Nl - np.arange(1, Nl + 1)))
    psi_br = np.abs(psi_b_Nl * (1 / t1r) ** (np.arange(Nl + 1, Nl + Nr + 1) - Nl))
    return np.append(psi_bl, psi_br)


if __name__ == '__main__':
    gamma = 1.3
    t1l = - 0.1
    t1r = 1.5
    ra = gamma*1j
    rb = -gamma*1j
    Nl = 40
    Nr = 40
    N = Nl + Nr
    chain = SSH.SSH(length=N, t1lp=t1l, t1lm=t1l, t1rp=t1r, t1rm=t1r, rla=ra, rlb=rb, rra=ra, rrb=rb, pos=Nl, bloch=True)  # bloch on
    energy, vector = chain.wavefunction
    vector_a = vector[::2, :]
    vector_b = vector[1::2, :]
    # 自动确定边界态位置
    i1 = np.where(np.abs(energy) == np.min(np.abs(energy)))
    i = int(i1[0])
    energy[i] = np.max(energy)
    j1 = np.where(np.abs(energy) == np.min(np.abs(energy)))
    j = int(j1[0])
    plt.plot(np.arange(1, N+1), np.abs(vector_a), 'k', LineWidth=0.8)  # a/b 转换
    # print(np.where(np.abs(vector_b) == np.max(np.abs(vector_b)))) # index begin with 0
    # gamma         | 0  |  0.11  |  1.33   |
    # vec_a/b index | ?  |   ?    |  46/51  |
    plt.plot(np.arange(1, N+1), np.abs(vector_a[:, j-1]), 'g', LineWidth=1.5)  # a/b 转换
    # psi_a_1 = 0.5255736661814411
    # psi_b_Nl = 0.525573666158743
    plt.plot(np.arange(1, N+1), wave_func_on_ep_a(t1l, t1r, Nl, Nr, 0.7432734067238927), 'rx')  # a/b 转换
    plt.xlabel(r'$j$')  # 上一行 +/- 转换
    plt.ylabel(r'$|\psi_{a,j}|$')  # a/b 转换
    plt.show()