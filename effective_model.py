import numpy as np
import matplotlib.pyplot as plt
from models import SSH

gamma = 0.1
mul = 0
mur = 0
t1 = 0.5
t2 = 1
N = 60
epsilon = 0.5 * (mul - mur)
mu = 0.5 * (mul + mur)
t1_1 = np.sqrt((t1-gamma)*(t1+gamma))
delta = (t1 ** 2 - t2 ** 2 - gamma ** 2) / t2 * (-(t1 ** 2 - gamma ** 2) / t2 ** 2) ** (N / 2)
E = np.sqrt(epsilon ** 2 + delta ** 2)
# np.where(np.abs(energy)==np.min(np.abs(energy)))
e_q = np.sqrt((t1 - gamma) / (t1 + gamma))
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
# 因为vector_a 和 vector_b 和 vector 对应的态的索引是相同的， 都是0～2*N-1内取值

# setup b1/b2
NN = np.sqrt(((1 - (t1_1 / t2) ** (2 * N)) / (1 - (t1_1 / t2) ** 2)))
b1 = np.kron((-t1_1 / t2 *e_q) ** (np.arange(1, N+1)-1), np.array([1, 0])) / NN    # index begins in 0 but should be 1
b2 = np.kron((-t1_1 / (t2 *e_q)) ** (N - np.arange(1, N+1)), np.array([0, 1])) / NN  # index begins in 0 but should be 1

psi_p = 1 / (np.sqrt(delta ** 2 + e_q**(-2*N)*(E - epsilon) ** 2)) * (
        (epsilon - E) * e_q**(-N)* b1 + delta * b2)
psi_m = 1 / (np.sqrt(e_q**(-2*N)*(E + epsilon) ** 2 + delta ** 2)) * (
        (epsilon + E) * e_q**(-N) * b1 - delta * b2)

# 分别在a b 子格上的情况
psi_pa = psi_p[::2]
psi_pb = psi_p[1::2]
psi_ma = psi_m[::2]
psi_mb = psi_m[1::2]
fp, = plt.plot(np.arange(1, 2 * N + 1), np.abs(b1))
fm, = plt.plot(np.arange(1, 2 * N + 1), np.abs(b2))
plt.plot(np.arange(1, 2 * N + 1), np.abs(vector[:, i]), 'kx', MarkerSize=4)
plt.plot(np.arange(1, 2 * N + 1), np.abs(vector[:, j]), 'kx', MarkerSize=4)
plt.legend([fp, fm], [r'$\psi_+$', r'$\psi_-$'])
plt.ylabel(r'$|\psi|$')
plt.title(r'$\gamma=$' + str(gamma) + r'  $t_1 =$' + str(t1) + r'  N=' + str(N))
