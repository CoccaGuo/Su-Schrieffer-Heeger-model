import numpy as np
import matplotlib.pyplot as plt
from models import EffSSH, SSH

if __name__ == '__main__':
    N = 100
    r = 120  # 解析度
    l = 3  # 起始和终止
    t2 = 1
    res = np.zeros((r, r))
    t1counter = -1
    t1range = np.linspace(-l, l, r)
    gammarange = np.linspace(-l, l, r)
    for t1 in t1range:
        t1counter += 1
        rcounter = -1
        for gamma in gammarange:
            rcounter += 1
            # eff = EffSSH.EffSSH(length=N, gamma=gamma, t1=t1, mul=0.2)
            # a = eff.r_bbc
            # b = eff.e_q
            # c = eff.b1
            # d = eff.b2
            # e = eff.psi_m
            # f = eff.psi_p
            chain = SSH.SSH(length=N, t1rp=t1 + gamma, t1rm=t1 - gamma, pos=0, bloch=False)
            energy, vector = chain.wavefunction
            # 自动确定边界态位置
            i1 = np.where(np.abs(energy) == np.min(np.abs(energy)))
            i = int(i1[0][0])
            energy[i] = np.max(energy)
            j1 = np.where(np.abs(energy) == np.min(np.abs(energy)))
            j = int(j1[0][0])
            r_bbc = 1 - np.abs(np.dot(vector[:,i],vector[:,j]))/2
            res[t1counter][rcounter] = r_bbc

plot = plt.contourf(t1range,gammarange,res)
plt.colorbar(plot)