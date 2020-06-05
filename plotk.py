from numpy import *
from matplotlib.pyplot import *
def dk(k,v,w):
    return [v+w*cos(k),w*sin(k),0]


def ekdk(v=0, w=1):
    kran = linspace(-pi, pi, 200)
    dx, dy = dk(kran, v, w)[:2]
    subplot(211)
    title(r"$t_1=%.1f$"%(v))
    plot(kran, sqrt(dx ** 2 + dy ** 2), 'k-', linewidth=2.5, color='dodgerblue')  # This creates the
    plot(kran, -sqrt(dx ** 2 + dy ** 2), 'k-', linewidth=2.5, color='dodgerblue')  # two bandlines
    plot([-pi,pi],[0,0],color='grey')
    ylabel('E', fontsize=20);
    xlabel(r'$k$', fontsize=20);
    xlim(-pi, pi);
    xticks([-pi, 0, pi], ['$-\pi$', '0', '$\pi$'], fontsize=18)
    ylim(-2.02, 2.02);
    yticks([-2, -1, 0, 1, 2], ['-2', '-1', '0', '1', '2'], fontsize=18);

    subplot(212)
    plot(dx, dy, 'k-', linewidth=3, color='dodgerblue')
    xlim(-2.1, 2.1)
    ylim(-2.1, 2.1)
    plot([-2, 2], [0, 0], '-', color='grey')
    plot([0, 0], [-2, 2], '-', color='grey')
    xticks([-2, -1, 0, 1, 2], ['-2', '-1', '0', '1', '2'], fontsize=18)
    yticks([-2, -1, 0, 1, 2], ['-2', '-1', '0', '1', '2'], fontsize=18)
    ylabel(r'$\sigma_y$', fontsize=22)
    xlabel(r'$\sigma_x$', fontsize=22)
    show()

ekdk(-0.5,0.1)