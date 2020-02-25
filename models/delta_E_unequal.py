import numpy as np
import matplotlib.pyplot as plt
from models import SSH

plt.rcParams['figure.dpi'] = 100  # dpi of the pictures

if __name__ == '__main__':
    gamma = 1.33
    t1l = - 0.1
    t1r = 1.5
    t1lp = t1l + 0.5 * gamma
    t1lm = t1l - 0.5 * gamma
    t1rp = t1r + 0.5 * gamma
    t1rm = t1r - 0.5 * gamma
    res = []
    for l in range(10, 200, 2):
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
    plt.plot(range(10, 200, 2), res)
    plt.show()
