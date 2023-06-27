import matplotlib.pyplot as plt


'''
functions for plotting some stuff related to decoding
'''


def plot_scores(scores,lower,upper,show=False,vline=0,plot_list=[]):
    cpu_scores=scores.to("cpu")
    T,H,E=scores.size()
    top = min(T,upper)
    for i in range(H):
        for j in range(E):
            if i in plot_list:
                style="-"
                if j==0: style = "--"
                plt.plot(range(lower,top),cpu_scores[lower:top,i,j],color='k',zorder=100,linestyle=style)
            else:
                plt.plot(range(lower,top),cpu_scores[lower:top,i,j])
    plt.ylim(-6000,-250)
    plt.axvline(x=vline)
    if show: plt.show()
    
