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
        self.t1_1 = ((self.t1 - self.gamma) * (t1 + self.gamma)) ** 0.5
        self.delta = (self.t1 ** 2 - self.t2 ** 2 - self.gamma ** 2) / t2 * (
                    -(t1 ** 2 - self.gamma ** 2) / t2 ** 2) ** (self.N / 2)
        self.E = np.sqrt(self.epsilon ** 2 + self.delta ** 2)
        # np.where(np.abs(energy)==np.min(np.abs(energy)))
        self.e_q = ((self.t1 - self.gamma) / (self.t1 + self.gamma)) ** 0.5
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
        return 1 - np.abs(np.conj(self.psi_p).dot(self.psi_m)) / 2


if __name__ == '__main__':
    N = 25
    t1 = 0.5
    gamma = 0.3
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
    vector_a = vector[::2,:]
    vector_b = vector[1::2,:]
    # 分别在a b 子格上的情况
    psi_pa = eff.psi_p[::2]
    psi_pb = eff.psi_p[1::2]
    psi_ma = eff.psi_m[::2]
    psi_mb = eff.psi_m[1::2]
    # fp, = plt.plot(np.arange(1, 2 * N + 1), (eff.psi_p))
    # fm, = plt.plot(np.arange(1, 2 * N + 1), (eff.psi_m))
    # plt.plot(np.arange(1, 2 * N + 1), (vector[:, i]), 'kx', MarkerSize=4)
    # plt.plot(np.arange(1, 2 * N + 1), (vector[:, j]), 'bx', MarkerSize=4)
    # plt.legend([fp, fm], [r'$\psi_+$', r'$\psi_-$'])
    # plt.plot((vector_a),color='grey')

    plt.plot((vector_a[:, i]), color='dodgerblue')
    plt.plot((vector_a[:, j]), color='dodgerblue')
    # plt.plot((psi_pa),'kx')
    plt.plot((psi_ma),'kx')
    # plt.plot((vector_b[:, i]), color='dodgerblue')
    # plt.plot((vector_b[:, j]), color='dodgerblue')
    # plt.plot((psi_pb),'kx')
    # plt.plot((psi_mb),'kx')
    font = {'weight': 'normal', 'size': 18}
    plt.ylabel(r'$\psi_a$',font)
    plt.xlabel('j',font)
    # plt.ylim(-5e-7,5e-7)
    # plt.title(r'$\gamma=$' + str(gamma) + r'  $t_1 =$' + str(t1) + r'  N=' + str(N))

if __name__ == '/__main__':
    t1 = 0.5
    t2 = 1
    gamma = 0.3
    resp = []
    resm = []
    ran = [3, 28]
    rang = np.arange(ran[0], ran[1])
    xrang = np.arange(ran[0] - 0.5, ran[1], 0.02)
    e_q = np.sqrt((t1 - gamma) / (t1 + gamma))
    cache1 = ((t1 ** 2 - gamma ** 2) / t2 ** 2)
    cache2 = ((t1 ** 2 - gamma ** 2) / t2 ** 2)
    delta_lp = (t1 ** 2 - t2 ** 2 - gamma ** 2) / t2 * (
            np.sign(cache1) * np.abs(cache1) ** (xrang / 2)) * e_q ** (xrang)
    delta_lm = (t1 ** 2 - t2 ** 2 - gamma ** 2) / t2 * (np.abs(cache2) ** (xrang / 2)) * e_q ** (
        -xrang)
    for N in rang:
        eff = EffSSH(length=N, t1=t1, gamma=gamma)
        chain = SSH.SSH(length=N, t1rp=t1 + gamma, t1rm=t1 - gamma, pos=0, bloch=False)
        resp.append(eff.b2.dot(chain.hamiltonian).dot(eff.b1))
        resm.append(eff.b1.dot(chain.hamiltonian).dot(eff.b2))
    anp, = plt.plot(xrang, np.abs(delta_lp), color='dodgerblue', LineWidth=1.5)
    anp, = plt.plot(xrang, np.abs(delta_lm), color='dodgerblue', LineWidth=1.5)
    dp, = plt.plot(rang, np.abs(resp), 'kx')
    dp, = plt.plot(rang, np.abs(resm), 'kx')
    font = {'size': 16}
    font2 = {'size': 12}
    plt.xlabel('N', font)
    plt.ylabel(r'$\Delta^\pm$', font)
    plt.title('Non-Hermitian approach of SSH model')
    # plt.title('Hermitian approach of SSH model')
    plt.legend([dp, anp], [r'$Analytical$', r'$Numerical$'], prop=font2)
