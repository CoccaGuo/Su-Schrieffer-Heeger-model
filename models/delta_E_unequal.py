import numpy as np
import matplotlib.pyplot as plt
from models import SSH

plt.rcParams['figure.dpi'] = 100  # dpi of the pictures

if __name__ == '__main__':
    gamma = 0.11
    t1l = -0.5
    t1r = 1.5
    t2 = 1
    t1lp = t1l + 0.5 * gamma
    t1lm = t1l - 0.5 * gamma
    t1rp = t1r + 0.5 * gamma
    t1rm = t1r - 0.5 * gamma
    res = []
    dellist = []
    for l in range(20, 100, 2):
        Nl = Nr = int(l / 2)
        N = Nl + Nr
        chain = SSH.SSH(length=N, t1lp=t1rp, t1lm=t1rm, t1rp=t1rp, t1rm=t1rm, pos=Nl, bloch=True)  # bloch on
        energy, _ = chain.wavefunction
        en = list(np.round(np.abs(energy), 13))
        min_energy = min(en)
        en.remove(min_energy)
        if min_energy in en:
            en.remove(min_energy)
        another_small_energy = min(en)
        res.append(another_small_energy - min_energy)
        delta = (t1l ** 2 - t2 ** 2 - gamma ** 2) / t2 * (-(t1l ** 2 - gamma ** 2) / t2 ** 2) ** (N / 2)
        dellist.append(delta)
    plt.plot(range(20, 100, 2), res, 'k')
    plt.plot(range(20, 100, 2), dellist, 'kx')
    plt.xlabel(r'$n$')
    plt.ylabel(r'$\Delta E$')
    plt.show()
