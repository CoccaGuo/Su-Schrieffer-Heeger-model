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
    gamma = 0.
    t1l = 0.5
    t1r = 1.5
    t1lp = t1l + 0.5 * gamma
    t1lm = t1l - 0.5 * gamma
    t1rp = t1r + 0.5 * gamma
    t1rm = t1r - 0.5 * gamma
    Nl = 40
    Nr = 40
    N = Nl + Nr
    chain = SSH.SSH(length=N, t1lp=t1lp, t1lm=t1lm, t1rp=t1rp, t1rm=t1rm, pos=Nl, bloch=False)  # bloch on
    energy, vector = chain.wavefunction
    vector_a = vector[::2, :]
    vector_b = vector[1::2, :]
    i1 = np.where(np.abs(energy) == np.min(np.abs(energy)))
    i = int(i1[0])
    energy[i] = np.max(energy)
    j1 = np.where(np.abs(energy) == np.min(np.abs(energy)))
    j = int(j1[0])
    font = {'weight': 'normal', 'size': 18}
    plt.plot(np.arange(1, N+1), np.abs(vector_b), color="#63B8FF", LineWidth=0.8)  # a/b 转换
    # print(np.where(np.abs(vector_b) == np.max(np.abs(vector_b))))
    # gamma         | 0  |  0.11  |  1.33   |
    # vec_a/b index | 50 |   54   | 101/102 |
    plt.plot(np.arange(1, N+1), wave_func_on_ep_b(t1l+0.5*gamma, t1r+0.5*gamma, Nl, Nr, vector_b[39, j]), color="red")  #  +/- a/b i/j 0/39
    plt.plot(np.arange(6,73,4), np.abs(vector_b[5:72:4, i]), 'x', color="k", LineWidth=1.5)  # a/b 转换
    plt.plot(np.arange(1, 5), np.abs(vector_b[0:4:, i]), 'x', color="k", LineWidth=1.5)  # a/b 转换
    plt.plot(np.arange(74, 80), np.abs(vector_b[73:79, i]), 'x', color="k", LineWidth=1.5)  # a/b 转换
    plt.xlabel(r'$j$',font)  # 上一行 +/- 转换
    plt.ylabel(r'$|\psi_{b,j}|$',font)  # a/b 转换
    plt.show()
