import numpy as np
import matplotlib.pyplot as plt
from models import SSH

gamma = 0.
mul = 0
mur = 0
t1 = 0.1
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
# 自动确定边界态位置
i1 = np.where(np.abs(energy) == np.min(np.abs(energy)))
i = int(i1[0])
energy[i] = np.max(energy)
j1 = np.where(np.abs(energy) == np.min(np.abs(energy)))
j = int(j1[0])
# 以下代码是错误的， 因为vector_a 和 vector_b 和 vector 对应的态的索引是相同的， 都是0～2*N-1内取值
# if i % 2 == 0:
#     cai, cbi = 1, 0
#     iab = i // 2
# else:
#     cai, cbi = 0, 1
#     iab = i // 2
# if j % 2 == 0:
#     caj, cbj = 1, 0
#     jab = j // 2
# else:
#     caj, cbj = 0, 1
#     jab = j // 2
# 绘图
# 以下代码是错误的， 因为vector_a 和 vector_b 和 vector 对应的态的索引是相同的， 都是0～2*N-1内取值
# psi_p = 1 / (np.sqrt(delta ** 2 + (E - epsilon) ** 2 * e_mqN ** 2)) * (
#             (epsilon - E) * e_mqN * (cai * vector_a[:, iab] + cbi * vector_b[:, iab]) + delta * (
#                 caj * vector_a[:, jab] + cbj * vector_b[:, jab]))
# psi_m = 1 / (np.sqrt((E + epsilon) ** 2 * e_mqN ** 2 + delta ** 2)) * (
#             (epsilon + E) * e_mqN * (cai * vector_a[:, iab] + cbi * vector_b[:, iab]) - delta * (
#                 caj * vector_a[:, jab] + cbj * vector_b[:, jab]))
# b1 = vector[:, i]
# b2 = vector[:, j]
# c = 0.001
# b1 = c*np.array(N*[1,0])
# b2 = c*np.array(N*[0,1])
# setup b1/b2

psi_p = 1 / (np.sqrt(delta ** 2 + (E - epsilon) ** 2 * e_mqN ** 2)) * (
            (epsilon - E) * e_mqN * b1 + delta * b2)
psi_m = 1 / (np.sqrt((E + epsilon) ** 2 * e_mqN ** 2 + delta ** 2)) * (
            (epsilon + E) * e_mqN * b1 - delta * b2)
# 分别在a b 子格上的情况
psi_pa = psi_p[::2]
psi_pb = psi_p[1::2]
psi_ma = psi_m[::2]
psi_mb = psi_m[1::2]
plt.plot(np.arange(1, 2*N + 1), np.abs(vector), 'kx', LineWidth=0.8)
fp, = plt.plot(np.arange(1, 2*N + 1), np.abs(b1))
fm, = plt.plot(np.arange(1, 2*N + 1), np.abs(b2))
plt.legend([fp, fm], [r'$\psi_+$', r'$\psi_-$'])
plt.legend([fp, fm], [r'$b_1$', r'$b_2$'])
plt.ylabel(r'$|\psi|$')
plt.title(r'$\gamma=$'+str(gamma)+r'  $t_1 =$'+str(t1)+r'  N='+str(N))
