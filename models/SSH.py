import numpy as np
from scipy import linalg


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
                 t2rm=1, rra=0, rrb=0, bloch=False):
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
        self.bloch = bloch

    @staticmethod
    def _simple_hamiltonian(t1p, t1m, t2p, t2m, ra, rb, length, is_complex):
        """
        a simple SSH model without domaining wall.
        :param is_complex: if is_complex = true, returns a complex matrix.
        :return: a simple SSH hamiltonian
        """
        hamil = np.zeros((2 * length, 2 * length), dtype='complex')
        for k in range(0, 2 * length - 1):
            hamil[k + 1, k + 1] = rb
            if k % 2 == 0:
                hamil[k, k + 1], hamil[k + 1, k], hamil[k, k] = t1p, t1m, ra
            else:
                hamil[k, k + 1], hamil[k + 1, k], hamil[k, k] = t2p, t2m, rb
        return hamil

    @property
    def hamiltonian(self):
        is_complex = False if self.rla == 0 and self.rlb == 0 and self.rra == 0 and self.rrb == 0 else True
        hamil = np.block([
            [self._simple_hamiltonian(self.t1lp, self.t1lm, self.t2lp, self.t2lm, self.rla, self.rlb, self.pos,
                                      is_complex),
             np.zeros((2 * self.pos, 2 * (self.length - self.pos)))],
            [np.zeros((2 * (self.length - self.pos), 2 * self.pos)),
             self._simple_hamiltonian(self.t1rp, self.t1rm, self.t2rp, self.t2rm, self.rra, self.rrb,
                                      self.length - self.pos, is_complex)]
        ])
        if self.pos != 0:  # please specify the position if used
            hamil[2 * self.pos, 2 * self.pos - 1] = self.delta  # don't forget that python counts from zero
            hamil[2 * self.pos - 1, 2 * self.pos] = self.delta
        if self.bloch:
            hamil[2 * self.length - 1, 0] = self.t2rp
            hamil[0, 2 * self.length - 1] = self.t2rm
        return hamil

    @property
    def wavefunction(self):
        """
        gives out the discrete value of the wavefunction matrix
        :return: energy, vector
        """
        energy, vector = linalg.eig(self.hamiltonian)
        return energy, vector


if __name__ == '__main__':
    chain = SSH(length=4, delta=3, t1rm=4, pos=2)
    print(chain.hamiltonian)
