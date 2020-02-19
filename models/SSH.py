import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300  # dpi of the pictures


def H(t1, t2, L):
    """
    gives out a Hamiltonian of SSH model
    :param t1: intracell hopping
    :param t2: intercell hopping
    :param L: SSH string length
    :return: hamiltonian
    """
    N = 2 * L
    if isinstance(t1, complex) or isinstance(t2, complex):
        H = np.zeros((N, N), dtype='complex')
    else:
        H = np.zeros((N, N))
    for k in range(0, N - 1):
        if k % 2 == 0:
            H[k, k + 1] = t1
            H[k + 1, k] = t1
        else:
            H[k, k + 1] = t2
            H[k + 1, k] = t2
    return H


def H_domain_wall_delta_hopping1(t1, t2, L, delta, pos):
    """
    Hamiltonian of a domain wall system of SSH model with a delta gap in it
    :param t1: intracell hopping
    :param t2: intercell hopping
    :param L: SSH string length
    :param delta: hopping strength
    :param pos: position of the delta-wall,e.g input 3 means:
        a(1)--t1--b(2)..t2..a(2)--t1--b(2)..t2..a(3)--t1--b(3) delta b(4)..t2..a(4)--t1--b(5)--...
    :return: hamiltonian
    """
    res = np.block([
        [H(t1, t2, pos), np.zeros((2 * pos, 2 * (L - pos)))],
        [np.zeros((2 * (L - pos), 2 * pos)), H(t2, t1, L - pos)]
    ])
    res[2 * pos, 2 * pos - 1] = delta  # don't forget that python counts from zero
    res[2 * pos - 1, 2 * pos] = delta
    return res

def H_domain_wall_delta_hopping2(t1l, t1r, t2, L, pos):
    """
    Hamiltonian of a domain wall system of SSH model with a delta gap in it
    :param t1l: intracell hopping left side
    :param t2: intercell hopping
    :param L: SSH string length
    :param t1r: intracell hopping right side
    :param pos: position of the domain wall,e.g input 3 means:
        a(1)--t1l±γ--b(2)..t2..a(2)--t1l±γ--b(2)..t2..a(3)--t1l±γ--b(3)..t2..a(4)--t1r±γ--b(4)..t2..a(5)--...
    :return: hamiltonian
    """
    return np.block([
        [H(t1l, t2, pos), np.zeros((2 * pos, 2 * (L - pos)))],
        [np.zeros((2 * (L - pos), 2 * pos)), H(t1r, t2, L - pos)]
    ])



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
    # print(H(0.5, 1, 10))
    # energy_spectrum(1, 10)
    print(H_domain_wall_delta_hopping2(0.5, 1, 4, 2, 2))
