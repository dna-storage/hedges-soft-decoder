import matplotlib.pyplot as plt


'''
functions for plotting some stuff related to decoding
'''


def plot_scores(scores,lower,upper,show=False,vline=0,plot_list=[]):
    cpu_scores=scores.to("cpu")
    T,H,E=scores.size()
    print(int(lower))
    print(int(upper))
    top = min(T,int(upper))
    for i in range(H):
        for j in range(E):
            if i in plot_list:
                style="-"
                if j==0: style = "--"
                plt.plot(range(int(lower),top),cpu_scores[int(lower):top,i,j],color='k',zorder=100,linestyle=style)
            else:
                plt.plot(range(int(lower),top),cpu_scores[int(lower):top,i,j])
    plt.ylim(-2000,-100)
    #plt.axvline(x=vline)
    plt.savefig("scores.png",dpi=1200)
    print("done plotting")
    
