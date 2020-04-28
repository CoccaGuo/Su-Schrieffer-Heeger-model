import numpy as np
import matplotlib.pyplot as plt
from models import SSH

gamma = 0.05
t1l = 0.5
t1r = 1.5
t2 = 1
mat12 = []
mat34 = []
Nr = 40
Nl = 40
e_q_l = np.sqrt((t1l - gamma) / (t1l + gamma))
e_q_r = np.sqrt((t1r - gamma) / (t1r + gamma))
for t1l in np.arange(-0.999,0.999,0.02): #(1.001,3,0.02)  (-0.999,0.999,0.02)
    # 解析
    NNl = np.sqrt(2)*np.sqrt(((1 - (t1l / t2) ** (2 * Nl)) / (1 - (t1l / t2) ** 2)))
    NNr = np.sqrt(2)*np.sqrt(((1 - (t1r / t2) ** (2 * Nr)) / (1 - (t1r / t2) ** 2)))
    b1l = np.kron((-t1l / t2 * e_q_l) ** (np.arange(1, Nl+1)-1), np.array([1, 0])) / NNl    # index begins in 0 but should be 1
    b1r = np.kron((-t1r / t2 * e_q_r) ** (np.arange(1, Nr+1)-1), np.array([1, 0])) / NNr    # index begins in 0 but should be 1
    b2l = np.kron((-t1l / (t2 * e_q_l)) ** (Nl - np.arange(1, Nl+1)), np.array([0, 1])) / NNl  # index begins in 0 but should be 1
    b2r = np.kron((-t1r / (t2 * e_q_r)) ** (Nr - np.arange(1, Nr+1)), np.array([0, 1])) / NNr  # index begins in 0 but should be 1

    #数值
    chain = SSH.SSH(length=Nl + Nr, t1lp=t1l + gamma, t1lm=t1l - gamma, t1rp=t1r + gamma, t1rm=t1r - gamma, pos=Nl,
                    bloch=True)
    #对比
    b1ll = np.append(b1l,np.zeros((1,2*Nr))) # 用0补全成一致的长度，这样可以矩阵运算
    b2ll = np.append(b2l, np.zeros((1, 2*Nr)))
    b1rr = np.append(np.zeros((1, 2*Nl)), b1r)
    b2rr = np.append(np.zeros((1, 2*Nl)), b2r)
    mat12.append(b1ll.dot(chain.hamiltonian).dot(b2ll))

# for Nr in range(4,41,1):
#     # 解析
#     Nr = 20
#     NNl = np.sqrt(2)*np.sqrt(((1 - (t1l / t2) ** (2 * Nl)) / (1 - (t1l / t2) ** 2)))
#     NNr = np.sqrt(2)*np.sqrt(((1 - (t1r / t2) ** (2 * Nr)) / (1 - (t1r / t2) ** 2)))
#     b1l = np.kron((-t1l / t2 * e_q_l) ** (np.arange(1, Nl+1)-1), np.array([1, 0])) / NNl    # index begins in 0 but should be 1
#     b1r = np.kron((-t1r / t2 * e_q_r) ** (np.arange(1, Nr+1)-1), np.array([1, 0])) / NNr    # index begins in 0 but should be 1
#     b2l = np.kron((-t1l / (t2 * e_q_l)) ** (Nl - np.arange(1, Nl+1)), np.array([0, 1])) / NNl  # index begins in 0 but should be 1
#     b2r = np.kron((-t1r / (t2 * e_q_r)) ** (Nr - np.arange(1, Nr+1)), np.array([0, 1])) / NNr  # index begins in 0 but should be 1
#
#     #数值
#     chain = SSH.SSH(length=Nl + Nr, t1lp=t1l + gamma, t1lm=t1l - gamma, t1rp=t1r + gamma, t1rm=t1r - gamma, pos=Nl,
#                     bloch=True)
#     #对比
#     b1ll = np.append(b1l,np.zeros((1,2*Nr))) # 用0补全成一致的长度，这样可以矩阵运算
#     b2ll = np.append(b2l, np.zeros((1, 2*Nr)))
#     b1rr = np.append(np.zeros((1, 2*Nl)), b1r)
#     b2rr = np.append(np.zeros((1, 2*Nl)), b2r)
#     mat34.append(b1rr.dot(chain.hamiltonian).dot(b2rr))

# 绘图
t1l = np.arange(-1,1,0.005)
cache1 = ((t1l ** 2 - gamma ** 2) / t2 ** 2)
cache2 = ((t1r ** 2 - gamma ** 2) / t2 ** 2)
delta_l = 0.5 * (t1l ** 2 - t2 ** 2 - gamma ** 2) / t2 * (np.sign(cache1)*np.abs(cache1) ** (Nl / 2)) #0.25是直接计算那一串计算得到的:  (-(t1l ** 2 - gamma ** 2) / t2 ** 2)
delta_r = 0.5 * (t1r ** 2 - t2 ** 2 - gamma ** 2) / t2 * (np.abs(cache2) ** (-Nr / 2)) # : (-(t1r ** 2 - gamma ** 2) / t2 ** 2)  ,计算非厄米问题的时候一定要重新算
zero = 0*t1r
dl, = plt.plot(t1l,np.abs(delta_l))
anl, = plt.plot(np.arange(-0.999,0.999,0.02), (np.abs(mat12)), 'kx', MarkerSize=3.5)

# dr, = plt.plot(dw_N,np.abs(delta_r))
# anr, = plt.plot(range(4,41,1), np.real(mat34), 'kx', MarkerSize = 3.5)
# plt.legend([dl,anl,dr,anr],[r'$\Delta_L$',r'$\langle b^1_L | H_{SSH}|b^2_L \rangle$',r'$\Delta_R$',r'$\langle b^1_R | H_{SSH}|b^2_R \rangle$'])
# plt.legend([dl,anl],[r'$Analytical$',r'$Numerical$'])
plt.xlabel('$t_1^L$')
# plt.title(r'$h_{1_l2_r}$')


