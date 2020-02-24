import numpy as np
import matplotlib.pyplot as plt
from models import SSH

plt.rcParams['figure.dpi'] = 100  # dpi of the pictures

if __name__ == '__main__':
    gamma = 0
    t1l = - 0.1
    t1r = 1.5
    t1lp = t1l + 0.5 * gamma
    t1lm = t1l - 0.5 * gamma
    t1rp = t1r + 0.5 * gamma
    t1rm = t1r - 0.5 * gamma
    chain = SSH.SSH(length=80, t1lp=t1lp, t1lm=t1lm, t1rp=t1rp, t1rm=t1rm, pos=40, bloch=True)  # bloch on
    energy, vector = chain.wavefunction
    vector_a = vector[::2, :]
    vector_b = vector[1::2, :]
    plt.plot(np.arange(1, 81), np.abs(vector_b), 'k', LineWidth=0.8)
    print(np.where(np.abs(vector_b) == np.max(np.abs(vector_b))))
    # gamma         | 0  |  0.11  |  1.33   |
    # vec_a/b index | 50 |   54   | 101/102 |
    plt.plot(np.arange(1, 81), np.abs(vector_b[:, 50]), 'g', LineWidth=1.5)
    # ********* a ********
    # funr1 = (1/t1r)**(np.arange(3, 15))
    # funl1 = np.abs(t1l) ** (np.arange(0.3, 2, 0.2))
    # funr2 = (1 / t1r) ** (np.arange(15, 40, 3))
    # funl2 = np.abs(t1l) ** (np.arange(2, 40, 3))
    # plt.plot(range(67, 79), np.flipud(np.abs(funr1)), 'rx')
    # plt.plot(np.arange(0.3, 2, 0.2), np.abs(funl1), 'rx')
    # plt.plot(range(42, 67, 3), np.flipud(np.abs(funr2)), 'rx')
    # plt.plot(np.arange(2, 40, 3), np.abs(funl2), 'rx')
    # ********* b *********

    def wave_func_on_ep_b(t1l, t1r, Nl, Nr, psi_b_Nl):
        psi_bl = np.abs(psi_b_Nl* t1l**(Nl-np.arange(1, 41)))
        psi_br = np.abs(psi_b_Nl* (1/t1r)**(Nl-np.arange(41, 81)))
        return np.append(psi_bl, psi_br)


    # funr1 = (1 / t1r) ** (np.arange(2, 14, 0.5))
    # funl1 = np.abs(t1l) ** (np.arange(0.3, 2, 0.1))
    # funr2 = (1 / t1r) ** (np.arange(15, 40, 3))
    # funl2 = np.abs(t1l) ** (np.arange(2, 39, 3))
    # plt.plot(np.arange(39.5, 51.5, 0.5), np.abs(funr1), 'rx')
    # plt.plot(np.arange(37.1, 38.8, 0.1), np.flipud(np.abs(funl1)), 'rx')
    # plt.plot(np.arange(51.5, 76.5, 3), np.abs(funr2), 'rx')
    # plt.plot(np.arange(0.1, 37.1, 3), np.flipud(np.abs(funl2)), 'rx')
    plt.xlabel(r'$j$')
    plt.ylabel(r'$|\psi_{b,j}|$')
    plt.show()
