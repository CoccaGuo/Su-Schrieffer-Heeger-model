import numpy as np
import matplotlib.pyplot as plt
from models import SSH

gamma = -0.2
mul = 0
mur = 0
t1 = 0.5
t2 = 1
N = 80
epsilon = 0.5 * (mul - mur)
mu = 0.5 * (mul + mur)
delta = (t1 ** 2 - t2 ** 2 - gamma ** 2) / t2 * (-(t1 ** 2 - gamma ** 2) / t2 ** 2) ** (N / 2)
E = np.sqrt(epsilon ** 2 + delta ** 2)
# np.where(np.abs(energy)==np.min(np.abs(energy)))
e_mqN = np.sqrt((t1 + gamma) / (t2 - gamma)) ** N
chain = SSH.SSH(length=N, t1rp=t1 + gamma, t1rm=t1 - gamma, pos=0, bloch=False)
energy, vector = chain.wavefunction
vector_a = vector[::2, :]
vector_b = vector[1::2, :]
plt.plot(np.arange(1, N + 1), abs(vector_b), 'k', LineWidth=0.8)  # a/b 转换
# 自动确定边界态位置
i1 = np.where(np.abs(energy) == np.min(np.abs(energy)))
i = int(i1[0])
energy[i] = np.max(energy)
j1 = np.where(np.abs(energy) == np.min(np.abs(energy)))
j = int(j1[0])
if i % 2 == 0:
    cai, cbi = 1, 0
    iab = i // 2
else:
    cai, cbi = 0, 1
    iab = i // 2
if j % 2 == 0:
    caj, cbj = 1, 0
    jab = j // 2
else:
    caj, cbj = 0, 1
    jab = j // 2
# 绘图
psi_p = 1 / (np.sqrt(delta ** 2 + (E - epsilon) ** 2 * e_mqN ** 2)) * (
            (epsilon - E) * e_mqN * (cai * vector_a[:, iab] + cbi * vector_b[:, iab]) + delta * (
                caj * vector_a[:, jab] + cbj * vector_b[:, jab]))
psi_m = 1 / (np.sqrt((E + epsilon) ** 2 * e_mqN ** 2 + delta ** 2)) * (
            (epsilon + E) * e_mqN * (cai * vector_a[:, iab] + cbi * vector_b[:, iab]) - delta * (
                caj * vector_a[:, jab] + cbj * vector_b[:, jab]))
fp, = plt.plot(np.arange(1, N + 1), np.abs(psi_p))
fm, = plt.plot(np.arange(1, N + 1), np.abs(psi_m))
plt.legend([fp, fm], [r'$\psi_+$', r'$\psi_-$'])
plt.ylabel(r'$|\psi|$')
