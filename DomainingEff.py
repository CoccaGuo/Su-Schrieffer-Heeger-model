import numpy as np
import matplotlib.pyplot as plt
from models import SSH


class DomainingEff:
    def __init__(self, Nl, Nr, t1l, t1r, t2=1, gamma=0):
        self.Nl = Nl
        self.Nr = Nr
        self.t1l = t1l
        self.t1r = t1r
        self.t2 = t2
        self.gamma = gamma
        self.e_q_l = np.sqrt((t1l - gamma) / (t1l + gamma))
        self.e_q_r = np.sqrt((t1r - gamma) / (t1r + gamma))

    @property
    def chain(self):
        return SSH.SSH(length=self.Nl + self.Nr, t1lp=self.t1l + self.gamma, t1lm=self.t1l - self.gamma,
                        t1rp=self.t1r + self.gamma, t1rm=self.t1r - self.gamma,
                        pos=self.Nl, bloch=True)

    def calc(self, i, j):
        NNl = np.sqrt(2) * np.sqrt(((1 - (self.t1l / self.t2) ** (2 * self.Nl)) / (1 - (self.t1l / self.t2) ** 2)))
        NNr = np.sqrt(2) * np.sqrt(((1 - (self.t1r / self.t2) ** (2 * self.Nr)) / (1 - (self.t1r / self.t2) ** 2)))
        b1l = np.kron((-self.t1l / self.t2 * self.e_q_l) ** (np.arange(1, self.Nl + 1) - 1),
                      np.array([1, 0])) / NNl  # index begins in 0 but should be 1
        b1r = np.kron((-self.t1r / self.t2 * self.e_q_r) ** (np.arange(1, self.Nr + 1) - 1),
                      np.array([1, 0])) / NNr  # index begins in 0 but should be 1
        b2l = np.kron((-self.t1l / (self.t2 * self.e_q_l)) ** (self.Nl - np.arange(1, self.Nl + 1)),
                      np.array([0, 1])) / NNl  # index begins in 0 but should be 1
        b2r = np.kron((-self.t1r / (self.t2 * self.e_q_r)) ** (self.Nr - np.arange(1, self.Nr + 1)),
                      np.array([0, 1])) / NNr  # index begins in 0 but should be 1
        # ref:
        # b1ll = np.append(b1l, np.zeros((1, 2 * self.Nr)))  # 用0补全成一致的长度，这样可以矩阵运算
        # b2ll = np.append(b2l, np.zeros((1, 2 * self.Nr)))
        # b1rr = np.append(np.zeros((1, 2 * self.Nl)), b1r)
        # b2rr = np.append(np.zeros((1, 2 * self.Nl)), b2r)
        ind = [np.append(b1l, np.zeros((1, 2 * self.Nr))), np.append(b2l, np.zeros((1, 2 * self.Nr))),
               np.append(np.zeros((1, 2 * self.Nl)), b1r), np.append(np.zeros((1, 2 * self.Nl)), b2r)]
        return ind[i - 1].dot(self.chain.hamiltonian).dot(ind[j - 1])

    # 数值直接计算结果--变化Nl
    def numerical_Nl(self, vary, i, j):  # i,j refers to the position in the matrix, starts from 1!
        res = []
        for self.Nl in vary:
           res.append(self.calc(i,j))
        return res

    # 数值直接计算结果--变化Nr
    def numerical_Nr(self, vary, i, j):  # i,j refers to the position in the matrix, starts from 1!
        res = []
        for self.Nr in vary:
           res.append(self.calc(i,j))
        return res

    # 数值直接计算结果--变化t1l
    def numerical_t1l(self, vary, i, j):  # i,j refers to the position in the matrix, starts from 1!
        res = []
        for self.t1l in vary:
           res.append(self.calc(i,j))
        return res

    # 数值直接计算结果--变化t1r
    def numerical_t1r(self, vary, i, j):  # i,j refers to the position in the matrix, starts from 1!
        res = []
        for self.t1r in vary:
           res.append(self.calc(i,j))
        return res

if __name__ == '/__main__':
    ## by N ##
    Nl = 20
    Nr = 20
    t1l = 0.5
    t1r = 1.5
    gamma = 0.22
    t2 = 1
    ran = [1, 20]
    vary = np.arange(ran[0], ran[1], 1)
    dw = DomainingEff(Nl, Nr, t1l, t1r, t2, gamma)
    dw_N = np.linspace(ran[0]-0.3, ran[1], 201)
    cache1 = ((t1l ** 2 - gamma ** 2) / t2 ** 2)
    cache2 = ((t1r ** 2 - gamma ** 2) / t2 ** 2)
    delta_l = 0.5 * (t1l ** 2 - t2 ** 2 - gamma ** 2) / t2 * (
                np.sign(cache1) * np.abs(cache1) ** (dw_N / 2)) * dw.e_q_l ** (dw_N)  # 0.25是直接计算那一串计算得到的:  (-(t1l ** 2 - gamma ** 2) / t2 ** 2)
    delta_r = 0.5 * (t1r ** 2 - t2 ** 2 - gamma ** 2) / t2 * (
            np.abs(cache2) ** (-dw_N / 2))  # : (-(t1r ** 2 - gamma ** 2) / t2 ** 2)  ,计算非厄米问题的时候一定要重新算
    font = {'weight': 'normal',
            'size': 16,
            }
    font2 = {
        'weight': 'normal',
        'size': 14,
    }
    dl, = plt.plot(dw_N, -(delta_l),color='dodgerblue')
    anl, = plt.plot(vary, (np.abs(dw.numerical_Nl(vary, 2, 1))), 'kx:', MarkerSize=5)
    # dr, = plt.plot(dw_N,np.abs(delta_r))
    # anr, = plt.plot(range(4,41,1), np.real(mat34), 'kx', MarkerSize = 3.5)
    # plt.legend([dl,anl,dr,anr],[r'$\Delta_L$',r'$\langle b^1_L | H_{SSH}|b^2_L \rangle$',r'$\Delta_R$',r'$\langle b^1_R | H_{SSH}|b^2_R \rangle$'])
    plt.legend([dl, anl], [r'$Analytical$', r'$Numerical$'],prop=font2)
    plt.xlabel('$N_L$',font)
    plt.ylabel(r'$|\Delta_L|$',font)
    plt.title(r'exotic secondary competitions in non-Hermitian condition',font2)

if __name__ == '/__main__':
    ## by t1 ##
    Nl = 20
    Nr = 20
    t1l = 0.5
    t1r = 1.5
    gamma = 0.0
    t2 = 1
    ran = [1, 2.5]
    vary = np.arange(ran[0], ran[1], 0.05)
    dw = DomainingEff(Nl, Nr, t1l, t1r, t2, gamma)
    dw_t = np.arange(ran[0]+0.01, ran[1], 0.01)
    ### 注意t1l 和 t1r 的区别
    # cache1 = ((dw_t ** 2 - gamma ** 2) / t2 ** 2)
    cache2 = ((dw_t ** 2 - gamma ** 2) / t2 ** 2)
    # delta_lp = 0.5 * (dw_t ** 2 - t2 ** 2 - gamma ** 2) / t2 * (np.sign(cache1) * np.abs(cache1) ** (Nl / 2)) * dw.e_q_l ** (
    #      Nl)  # 0.25是直接计算那一串计算得到的:  (-(t1l ** 2 - gamma ** 2) / t2 ** 2)
    delta_rp = 0.5 * (dw_t ** 2 - t2 ** 2 - gamma ** 2) / t2 * (np.sign(cache2) * np.abs(cache2) ** (-Nr / 2)) * dw.e_q_r ** (
         Nr)  # : (-(t1r ** 2 - gamma ** 2) / t2 ** 2)  ,计算非厄米问题的时候一定要重新算
    font = {'weight': 'normal',
            'size': 16,
            }
    font2 = {
             'weight': 'normal',
             'size': 14,
             }
    # dl, = plt.plot(dw_t, np.abs(delta_lp),color='dodgerblue')
    # anl, = plt.plot(vary, (np.abs(dw.numerical_t1l(vary, 2, 1))), 'kx', MarkerSize=5)
    dl, = plt.plot(dw_t, np.abs(delta_rp),color='dodgerblue')
    anl, = plt.plot(vary, (np.abs(dw.numerical_t1r(vary, 4, 3))), 'kx', MarkerSize=5)
    plt.legend([dl, anl], [r'$Analytical$', r'$Numerical$'],prop=font2)
    plt.xlabel(r'$t_1^R$',font)
    plt.ylabel(r'$\Delta_R$',font)
    plt.title(r'$\Delta$ changes by $t_1^R$',font2)


if __name__ == '/__main__':
    Nl = 20
    Nr = 20
    t1l = 0.5
    t1r = 1.5
    gamma = 0.00
    t2 = 1
    ran = [-1, 1]
    vary = np.linspace(ran[0], ran[1], 200)
    en = np.zeros((len(vary),(Nl+Nr)*2))
    i = 0
    for t1l in vary:
        dw = DomainingEff(Nl, Nr, t1l, t1r, t2, gamma)
        ener, _ = dw.chain.wavefunction
        en[i,:] = ener
        i+=1
    plt.plot(vary,(en[:,:]),'k.',MarkerSize=0.5)