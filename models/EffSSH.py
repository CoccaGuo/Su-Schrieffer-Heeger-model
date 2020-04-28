import numpy as np
import matplotlib.pyplot as plt
from models import SSH


class EffSSH:
    def __init__(self, length, t1=0.5, t2=1.0, gamma=0., mul=0., mur=0.):
        self.gamma = gamma
        self.mul = mul
        self.mur = mur
        self.t1 = t1
        self.t2 = t2
        self.N = length
        self.epsilon = 0.5 * (mul - mur)
        self.mu = 0.5 * (mul + mur)
        self.t1_1 = ((self.t1 - self.gamma) * (t1 + self.gamma))**0.5
        self.delta = (self.t1 ** 2 - self.t2 ** 2 - self.gamma ** 2) / t2 * (-(t1 ** 2 - self.gamma ** 2) / t2 ** 2) ** (self.N / 2)
        self.E = np.sqrt(self.epsilon ** 2 + self.delta ** 2)
        # np.where(np.abs(energy)==np.min(np.abs(energy)))
        self.e_q = ((self.t1 - self.gamma) / (self.t1 + self.gamma))**0.5
        self.NN = np.sqrt(((1 - (self.t1_1 / t2) ** (2 * self.N)) / (1 - (self.t1_1 / t2) ** 2)))
        self.chain = SSH.SSH(length=length, t1rp=t1 + gamma, t1rm=t1 - gamma, pos=0, bloch=False)

    @property
    def b1(self):
        return np.kron((-self.t1_1 / self.t2 * self.e_q) ** (np.arange(1, self.N + 1) - 1),
                          np.array([1, 0])) / self.NN  # index begins in 0 but should be 1
    @property
    def b2(self):
        return np.kron((-self.t1_1 / (self.t2 * self.e_q)) ** (self.N - np.arange(1, self.N + 1)),
                np.array([0, 1])) / self.NN  # index begins in 0 but should be 1

    @property
    def psi_p(self):
        return 1 / (np.sqrt(self.delta ** 2 + self.e_q ** (-2 * self.N) * (self.E - self.epsilon) ** 2)) * (
                (self.epsilon - self.E) * self.e_q ** (-self.N) * self.b1 + self.delta * self.b2)

    @property
    def psi_m(self):
        return 1 / (np.sqrt(self.e_q ** (-2 * self.N) * (self.E + self.epsilon) ** 2 + self.delta ** 2)) * (
                (self.epsilon + self.E) * self.e_q ** (-self.N) * self.b1 - self.delta * self.b2)

    @property
    def r_bbc(self):
        return 1-np.abs(np.conj(self.psi_p).dot(self.psi_m))/2

if __name__ == '__main__':
    N = 120
    t1 = 0.5
    gamma = 0
    chain = SSH.SSH(length=N, t1rp=t1 + gamma, t1rm=t1 - gamma, pos=0, bloch=False)
    energy, vector = chain.wavefunction
    # 自动确定边界态位置
    i1 = np.where(np.abs(energy) == np.min(np.abs(energy)))
    i = int(i1[0])
    energy[i] = np.max(energy)
    j1 = np.where(np.abs(energy) == np.min(np.abs(energy)))
    j = int(j1[0])
    # 因为vector_a 和 vector_b 和 vector 对应的态的索引是相同的， 都是0～2*N-1内取
    eff = EffSSH(length=N, t1=t1, gamma=gamma)
    # 分别在a b 子格上的情况
    # psi_pa = eff.psi_p[::2]
    # psi_pb = eff.psi_p[1::2]
    # psi_ma = eff.psi_m[::2]
    # psi_mb = eff.psi_m[1::2]
    fp, = plt.plot(np.arange(1, 2 * N + 1), np.abs(eff.psi_p))
    fm, = plt.plot(np.arange(1, 2 * N + 1), np.abs(eff.psi_m))
    plt.plot(np.arange(1, 2 * N + 1), np.abs(vector[:, i]), 'kx', MarkerSize=4)
    plt.plot(np.arange(1, 2 * N + 1), np.abs(vector[:, j]), 'bx', MarkerSize=4)
    plt.legend([fp, fm], [r'$\psi_+$', r'$\psi_-$'])
    plt.ylabel(r'$|\psi|$')
    plt.title(r'$\gamma=$' + str(gamma) + r'  $t_1 =$' + str(t1) + r'  N=' + str(N))

if __name__ == '/__main__':
    eff = EffSSH(length=100,t1=0.99,gamma=0.0,mul=0.2)
    print(eff.delta)
    print(eff.e_q)
    print(eff.delta*eff.e_q)