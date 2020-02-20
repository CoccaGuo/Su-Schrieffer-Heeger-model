import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300  # dpi of the pictures


class SSH:
    """
    Su-Schrieffer-Heeger(SSH) model: A-B sublattice
    A--B...A--B...A--B... ...
     length: equals to N, the length of the lattice.
     t1: intracell
     t2: intercell
        --l: 链左侧
        --r: 链右侧
        --p: 左侧向右侧跳跃
        --m: 右侧向左侧跳跃
     r:stagger虚化学势
        --l: 链左侧
        --r: 链右侧
        --a:作用在A格子上
        --b:作用在B格子上
    pos: 左右分隔位置
    delta: 分隔宽度
    """

    def __init__(self, length=20, t1lp=1, t1lm=1, t2lp=1, t2lm=1, rla=0, rlb=0, pos=0, delta=1, t1rp=1, t1rm=1, t2rp=1,
                 t2rm=1, rra=0, rrb=0):
        self.length = length
        self.t1lp = t1lp
        self.t1lm = t1lm
        self.t2lp = t2lp
        self.t2lm = t2lm
        self.rla = rla
        self.rlb = rlb
        self.pos = pos
        self.delta = delta
        self.t1rp = t1rp
        self.t1rm = t1rm
        self.t2rp = t2rp
        self.t2rm = t2rm
        self.rra = rra
        self.rrb = rrb

    @staticmethod
    def __simple_hamiltonian(t1p, t1m, t2p, t2m, ra, rb, length, is_complex):
        """
        a simple SSH model without domaining wall.
        :param is_complex: if is_complex = true, returns a complex matrix.
        :return: a simple SSH hamiltonian
        """
        hamil = np.zeros((2 * length, 2 * length), dtype='complex') if is_complex else np.zeros(
            (2 * length, 2 * length))
        for k in range(0, 2 * length - 1):
            hamil[k + 1, k + 1] = rb
            if k % 2 == 0:
                hamil[k, k + 1], hamil[k + 1, k], hamil[k, k] = t1p, t1m, ra
            else:
                hamil[k, k + 1], hamil[k + 1, k], hamil[k, k] = t2p, t2m, rb
        return hamil

    def hamiltonian(self):
        is_complex = False if self.rla == 0 and self.rlb == 0 and self.rra == 0 and self.rrb == 0 else True
        hamil = np.block([
            [self.__simple_hamiltonian(self.t1lp, self.t1lm, self.t2lp, self.t2lm, self.rla, self.rlb, self.pos, is_complex),
             np.zeros((2 * self.pos, 2 * (self.length - self.pos)))],
            [np.zeros((2 * (self.length - self.pos), 2 * self.pos)),
             self.__simple_hamiltonian(self.t1rp, self.t1rm, self.t2rp, self.t2rm, self.rra, self.rrb, self.length - self.pos, is_complex)]
        ])
        hamil[2 * self.pos, 2 * self.pos - 1] = self.delta  # don't forget that python counts from zero
        hamil[2 * self.pos - 1, 2 * self.pos] = self.delta
        return hamil


    def wavefunction(self):





    # def energy_spectrum(self, t1s, t1e, resolution=0.01):
    #     """
    #     gives out the energy spectra of a certain ssh chain.
    #
    #
    #     :param t1s: the beginning value of t1
    #     :param t1e: the end value of t1
    #     :return:
    #     """
    #     res = 1/resolution
    #     N = 2*self.length
    #     energy = np.zeros((N, res*np.abs(t1e - t1s)))
    #     k = 0
    #     for t1 in np.arange(t1s, t1e, res):
    #         pass


# def energy_spectrum(t2, L):
#     """
#     gives out an energy spectrum of SSH models (in Hermitian case)
#     t1 changes from -2 to 2
#     usage: energy_spectrum(1, 100)
#
#     :param t2: intercell hopping
#     :param L: string length
#     :return: no parameters but a plot figure
#     """
#     N = 2 * L
#     energy = np.zeros((N, 400))
#     k = 0
#     for t1 in np.arange(-2, 2, 0.01):
#         H = H(t1, t2, L)
#         energy[:, k], _ = np.linalg.eig(H)
#         k = k + 1
#     plt.plot(np.arange(-2, 2, 0.01), energy.T, 'k.', markersize=1)
#     plt.show()


if __name__ == '__main__':
    chain = SSH(length=3, pos=1, delta=2, t1lp=0.5, t1lm=-0.5, rra=0.1j, rrb=-0.1j)
    print(chain.hamiltonian())
