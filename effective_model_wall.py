import numpy as np
import matplotlib.pyplot as plt
from models import SSH

gamma = 0
t1l = 0.5
t1r = 1.5
t2 = 1
Nl = 10
Nr = 10
delta_l = (t1l ** 2 - t2 ** 2 - gamma ** 2) / t2 * (-(t1l ** 2 - gamma ** 2) / t2 ** 2) ** (Nl / 2)
E_l = np.sqrt(delta_l ** 2)
e_q_l = np.sqrt((t1l - gamma) / (t1l + gamma))

delta_r = (t1r ** 2 - t2 ** 2 - gamma ** 2) / t2 * (-(t1r ** 2 - gamma ** 2) / t2 ** 2) ** (Nr / 2)
E_r = np.sqrt(delta_r ** 2)
e_q_r = np.sqrt((t1r - gamma) / (t1r + gamma))
# 解析

NNl = np.sqrt(2)*np.sqrt(((1 - (t1l / t2) ** (2 * Nl)) / (1 - (t1l / t2) ** 2)))
NNr = np.sqrt(2)*np.sqrt(((1 - (t1r / t2) ** (2 * Nr)) / (1 - (t1r / t2) ** 2)))
b1l = np.kron((-t1l / t2 * e_q_l) ** (np.arange(1, Nl+1)-1), np.array([1, 0])) / NNl    # index begins in 0 but should be 1
b1r = np.kron((-t1r / t2 * e_q_r) ** (np.arange(1, Nr+1)-1), np.array([1, 0])) / NNr    # index begins in 0 but should be 1
b2l = np.kron((-t1l / (t2 * e_q_l)) ** (Nl - np.arange(1, Nl+1)), np.array([0, 1])) / NNl  # index begins in 0 but should be 1
b2r = np.kron((-t1r / (t2 * e_q_r)) ** (Nr - np.arange(1, Nr+1)), np.array([0, 1])) / NNr  # index begins in 0 but should be 1

psi_lp = 1 / (np.sqrt(1+e_q_l**(2*Nl))) * ((-e_q_l**Nl)*b1l + b2l)
psi_lm = 1 / (np.sqrt(1+e_q_l**(2*Nl))) * ((e_q_l**Nl)*b1l + b2l)
psi_rp = 1 / (np.sqrt(1+e_q_r**(2*Nr))) * ((-e_q_r**Nr)*b1r + b2r)
psi_rm = 1 / (np.sqrt(1+e_q_r**(2*Nr))) * ((e_q_r**Nr)*b1r + b2r)

# 数值
chain = SSH.SSH(length=Nl+Nr, t1lp=t1l+gamma, t1lm=t1l-gamma, t1rp=t1r+gamma, t1rm=t1r-gamma, pos=Nl, bloch=True)
energy, vector = chain.wavefunction
vector_a = vector[::2, :]
vector_b = vector[1::2, :]

# # 自动确定边界态位置
i1 = np.where(np.abs(energy) == np.min(np.abs(energy)))
i = int(i1[0])
energy[i] = np.max(energy)
j1 = np.where(np.abs(energy) == np.min(np.abs(energy)))
j = int(j1[0])

# vector_a 和 vector_b 和 vector 对应的态的索引是相同的， 都是0～2*N-1内取值
# 绘图
N = Nl + Nr
#plt.plot(np.arange(1, 2*N+1), np.abs(vector), 'k', LineWidth=0.8)  # a/b 转换
plt.plot(np.arange(1, 2*N+1), np.abs(vector[:, j]), 'k', LineWidth=1)  # a/b 转换
plt.plot(np.arange(1, 2*N+1), np.abs(vector[:, i]), LineWidth=1)  # a/b 转换
#plt.plot(np.arange(1, 2*Nl + 1), np.abs(psi_lm),'r',MarkerSize=3)
plt.plot(np.arange(1, 2*Nl + 1), np.abs(psi_lp),'r',LineWidth=0.5,MarkerSize=3)
#plt.plot(np.arange(2*Nl + 1, 2*N + 1), np.abs(psi_rm),'r',MarkerSize=3)
plt.plot(np.arange(2*Nl + 1, 2*N + 1), np.abs(psi_rp),'r',LineWidth=0.5,MarkerSize=3)

# plt.legend([fp, fm], [r'$\psi_+$', r'$\psi_-$'])
# plt.ylabel(r'$|\psi|$')

