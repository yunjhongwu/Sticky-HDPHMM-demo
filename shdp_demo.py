import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib import animation
from matplotlib.colors import PowerNorm
from cycler import cycler        

from shdp import StickyHDPHMM

if __name__ == '__main__':    
    #np.random.seed(11)
    H = 3
    L = 30
    colors = ['r', 'b', 'g']
    data = np.loadtxt("simulated_data.txt")
    T = data.shape[0]    
    vmin, vmax = np.min(data) * 0.5, np.max(data) * 1.5
    xs = np.logspace(np.log10(vmin), np.log10(vmax), 100)
    logxs = np.log10(xs)
    logdata = np.log10(data)
    hdp = StickyHDPHMM(logdata, L=L)#, kmeans_init=True)
    shdp = StickyHDPHMM(logdata, kappa=10, L=L, 
                        kmeans_init=False)

    def init():
        for h in range(H):
            line_shdp[h].set_data([], [])
            dist_shdp[h].set_data([], [])
            areas[h].set_xy([(0, 1), (0, 1)])
            
        text.set_text("")
        trans_shdp.set_data(shdp.PI)
        for h in range(H):
            ax4.add_patch(areas[h])
        return line_shdp + dist_shdp + [trans_shdp, text] + areas
        
    def update(t):
        shdp.sampler()
        for h in range(H):
            estimates_shdp = shdp.getPath(h)
            line_shdp[h].set_data(np.arange(T), 10 ** estimates_shdp)
            density = gaussian_kde(estimates_shdp)
            density.set_bandwidth(0.1)
            ys = density(logxs)

            areas[h].set_xy(list(zip(ys, xs)) + [(0, xs[-1]), (0, xs[0])])
            dist_shdp[h].set_data(ys, xs)
            
        trans_shdp.set_data(shdp.PI.copy())
        text.set_text("MCMC iteration {0}".format(t))
        return line_shdp + dist_shdp + [trans_shdp, text] + areas

    cycle = cycler('color', colors)
    fig = plt.figure(figsize=(14, 8), facecolor='w')
    
    ax1 = plt.subplot2grid((15, 20), (0, 0), colspan=13, rowspan=5)    
    plt.gca().set_prop_cycle(cycle)

    ax1.set_title("Simulated data")
    ax1.set_yscale("log")

    ax1.plot(data)
    ax1.set_ylabel("$f(t)$")
    ax1.set_xticklabels([])
    ax1.set_ylim([vmin, vmax])
    ax1.set_xlim([0, 288])
    ax1.grid()
    
    ax2 = plt.subplot2grid((15, 20), (7, 0), colspan=13, rowspan=5)
    plt.gca().set_prop_cycle(cycle)
    ax2.set_title("Sticky HDP-HMM")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("$f(t)$")
    ax2.set_yscale("log")
    ax2.plot(data, alpha=0.5)     
    estimates_shdp = np.array([10 ** shdp.getPath(h) for h in range(H)]).T
    line_shdp = ax2.plot(np.arange(T), estimates_shdp, linewidth=2) 
    ax2.set_ylim([vmin, vmax])
    ax2.set_xlim([0, 288])
    ax2.grid()
    
    ax3 = plt.subplot2grid((15, 20), (0, 13), colspan=2, rowspan=5)    
    plt.gca().set_prop_cycle(cycle)
    ax3.set_yscale("log")
    density = [gaussian_kde(logdata[:, h]) for h in range(H)]
    for h in range(H):
        density[h].set_bandwidth(0.1)
    ys = np.array([density[h](logxs) for h in range(H)]).T
    ax3.plot(ys, xs)
    for h in range(H):
        ax3.add_patch(plt.Polygon(list(zip(ys[:, h], xs)) + [(0, xs[-1]), (0, xs[0])], 
                                  color=colors[h], alpha=0.3))

    ax3.set_title("Distribution")
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_ylim([vmin, vmax])
    ax3.grid()
    
    ax4 = plt.subplot2grid((15, 20), (7, 13), colspan=2, rowspan=5)  
    plt.gca().set_prop_cycle(cycle)
    ax4.set_yscale("log")
    ys = np.array([density[h](logxs)  for h in range(H)]).T
    dist_shdp = ax4.plot(ys, xs)
    areas = [plt.Polygon([(0, 1), (0, 1)], 
                          color=colors[h], alpha=0.3) for h in range(H)]

    for h in range(H):
        ax4.add_patch(areas[h])

    ax4.set_xticklabels([])
    ax4.set_yticklabels([])
    ax4.set_ylim([vmin, vmax])
    ax4.grid()
    
    ax5 = plt.subplot2grid((15, 20), (0, 16), colspan=5, rowspan=5)  
    ax5.set_title("Transition matrix")
    ax5.axis('off')


    ax6 = plt.subplot2grid((15, 20), (7, 16), colspan=5, rowspan=5)    
    trans_shdp = ax6.matshow(shdp.PI, norm=PowerNorm(0.2, 0, 1), 
                             vmin=0, vmax=0.1, aspect='auto')
    ax6.axis('off')

    ax7 = plt.subplot2grid((15, 20), (14, 5), colspan=5, rowspan=1)
    text = ax7.text(0, 0.3, '', fontsize=15) 
    ax7.axis('off')

    ani = animation.FuncAnimation(fig, update, interval=0, blit=True, 
                            frames=10000, init_func=init)
    plt.show()
