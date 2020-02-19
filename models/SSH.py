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
    length = 20
    t1lp = 1
    t1lm = 1
    t2lp = 1
    t2lm = 1
    rla = 0j
    rlb = 0j
    pos = length // 2
    delta = 1
    t1rp = 1
    t1rm = 1
    t2rp = 1
    t2rm = 1
    rra = 0j
    rrb = 0j

    @staticmethod
    def __simple_hamiltonian(t1p, t1m, t2p, t2m, ra, rb, length, is_complex):
        """
        a simple SSH model without gap and domaining wall.
        :param is_complex: if is_complex = true, returns a complex matrix.
        :return: a simple SSH hamiltonian
        """
        hamil = np.zeros((2*length, 2*length), dtype='complex') if is_complex else np.zeros((2*length, 2*length))
        for k in range(0, 2*length - 1):
            hamil[k, k + 1], hamil[k + 1, k], hamil[k, k] = t1p, t1m, ra if k % 2 == 0 else t2p, t2m, rb
        return hamil

    def hamiltonian(_):
        is_complex = False if _.rla == 0 and _.rlb == 0 and _.rra == 0 and _.rrb == 0 else True
        hamil = np.block([
            [_.__simple_hamiltonian(_.t1lp, _.t1lm, _.t2lp, _.t2lm, _.rla, _.rlb, _.pos, is_complex),
             np.zeros((2 * _.pos, 2 * (_.length - _.pos)))],
            [np.zeros((2 * (_.length - _.pos), 2 * _.pos)),
             _.__simple_hamiltonian(_.t1rp, _.t1rm, _.t2rp, _.t2rm, _.rra, _.rrb, _.length - _.pos, is_complex)]
        ])
        hamil[2 * _.pos, 2 * _.pos - 1] = _.delta  # don't forget that python counts from zero
        hamil[2 * _.pos - 1, 2 * _.pos] = _.delta
        return hamil

def energy_spectrum(t2, L):
    """
    gives out an energy spectrum of SSH models (in Hermitian case)
    t1 changes from -2 to 2
    usage: energy_spectrum(1, 100)

    :param t2: intercell hopping
    :param L: string length
    :return: no parameters but a plot figure
    """
    N = 2 * L
    energy = np.zeros((N, 400))
    k = 0
    for t1 in np.arange(-2, 2, 0.01):
        H = H(t1, t2, L)
        energy[:, k], _ = np.linalg.eig(H)
        k = k + 1
    plt.plot(np.arange(-2, 2, 0.01), energy.T, 'k.', markersize=1)
    plt.show()


if __name__ == '__main__':
    chain = SSH()
    print(chain.hamiltonian())
