import argparse
import numpy as np
import matplotlib.pyplot as plt


def draw_bubble(args, i):
    plt.scatter(args.X[1, i], args.X[0, i],
                    c=args.colors[i],
                    s=args.z[args.y == i]*300,
                    alpha=0.6,
                    label=args.labels[i],
                    linewidths=2,
                    )
    
    # Center of circle
    if args.labels[i] == 'DGNet-S':
        plt.scatter(args.X[1, i], args.X[0, i],
                    c='red',
                    s=24*1.5,
                    label=args.labels[i],
                    linewidths=2,

                    )

    elif args.labels[i] == 'DGNet':
        plt.scatter(args.X[1, i], args.X[0, i],
                    c='red',
                    s=24*1.5,
                    label=args.labels[i],
                    linewidths=2,

                    )
    elif args.labels[i] == 'SINetV2':
        plt.scatter(args.X[1, i], args.X[0, i],
                    c='black',
                    s=24*1.5,
                    label=args.labels[i],
                    linewidths=2,

                    )
    elif args.labels[i] == 'JCSOD':
        plt.scatter(args.X[1, i], args.X[0, i],
                    c='black',
                    s=24*1.5,
                    label=args.labels[i],
                    linewidths=2,
                    )
    else:
        plt.scatter(args.X[1, i], args.X[0, i],
                c='#001146',
                s=24*1.5,
                label=args.labels[i]
                )


def draw_text(args, i):
    if args.labels[i] == 'SCRN':
        plt.annotate(args.labels[i], (args.X[1, i], args.X[0, i]), xycoords='data',
                    xytext=(2, 0.655),
                    arrowprops=dict(arrowstyle='-',
                                    connectionstyle='arc,angleA=-90,angleB=0,armA=30,armB=0,rad=0',
                                    color="0.4"),
                    fontsize=args.font_size)
    elif args.labels[i] == 'UCNet':
        plt.annotate(args.labels[i], (args.X[1, i], args.X[0, i]), xycoords='data',
                    xytext=(2, args.X[0, i]-0.018),
                    arrowprops=dict(arrowstyle='-',
                                    connectionstyle='arc,angleA=90,angleB=0,armA=30,armB=0,rad=0',
                                    color="0.4"),
                        fontsize=args.font_size)
    elif args.labels[i] == 'CSNet-R':
        plt.annotate(args.labels[i], (args.X[1, i], args.X[0, i]), xycoords='data',
                    xytext=(args.X[1, i]+5, args.X[0, i] - 0.035),
                    arrowprops=dict(arrowstyle='-',
                                    connectionstyle='arc,angleA=90,angleB=0,armA=30,armB=0,rad=0',
                                    color="0.4"),
                    fontsize=args.font_size)
    elif args.labels[i] == 'DGNet-S':
        plt.text(args.X[1, i]-6.8, args.X[0, i]+0.003,
                args.labels[i],
                fontsize=args.font_size,
                color='black',
                fontweight='bold')
    elif args.labels[i] == 'DGNet':
        plt.text(args.X[1, i] + 1, args.X[0, i] + 0.002,
                args.labels[i],
                fontsize=args.font_size,
                color='black',
                fontweight='bold')
    elif args.labels[i] == 'PFNet':
        plt.annotate(args.labels[i], (args.X[1, i], args.X[0, i]), xycoords='data',
                    xytext=(37, 0.718),
                    arrowprops=dict(arrowstyle='-',
                                    connectionstyle='arc,angleA=-90,angleB=180,armA=30,armB=0,rad=0',
                                    color="0.4"),
                    fontsize=args.font_size)
    elif args.labels[i] == 'MINet-R':
        plt.text(args.X[1, i] - 21, args.X[0, i]-0.01,
                args.labels[i],
                fontsize=args.font_size)
    elif args.labels[i] == 'R-MGL':
        plt.annotate(args.labels[i], (args.X[1, i], args.X[0, i]), xycoords='data',
                    xytext=(82, 0.7),
                    arrowprops=dict(arrowstyle='-',
                                    connectionstyle='arc,angleA=-90,angleB=180,armA=30,armB=0,rad=0',
                                    color="0.4"),
                    fontsize=args.font_size)
    elif args.labels[i] == 'S-MGL':
        plt.annotate(args.labels[i], (args.X[1, i], args.X[0, i]), xycoords='data',
                    xytext=(85, 0.68),
                    arrowprops=dict(arrowstyle='-',
                                    connectionstyle='arc,angleA=-90,angleB=180,armA=30,armB=0,rad=0',
                                    color="0.4"),
                    fontsize=args.font_size)
    elif args.labels[i] == 'ITSD':
        plt.text(args.X[1, i]-10, args.X[0, i] - 0.01,
                args.labels[i],
                fontsize=args.font_size)
    elif args.labels[i] == 'PraNet':
        plt.annotate(args.labels[i], (args.X[1, i], args.X[0, i]), xycoords='data',
                    xytext=(2, 0.6725),
                    arrowprops=dict(arrowstyle='-',
                                    connectionstyle='arc,angleA=-90,angleB=180,armA=30,armB=0,rad=0',
                                    color="0.4"),
                    fontsize=args.font_size)
    elif args.labels[i] == 'F3Net':
        plt.text(args.X[1, i]-15, args.X[0, i] + 0.005,
                args.labels[i],
                fontsize=args.font_size)
    elif args.labels[i] == 'CPD':
        plt.text(args.X[1, i]+3, args.X[0, i] - 0.01,
                args.labels[i],
                fontsize=args.font_size)
    elif args.labels[i] == 'PoolNet':
        plt.text(args.X[1, i]+2, args.X[0, i] - 0.01,
                args.labels[i],
                fontsize=args.font_size)
    elif args.labels[i] == 'EGNet':
        plt.text(args.X[1, i]+2, args.X[0, i] - 0.01,
                args.labels[i],
                fontsize=args.font_size)
    elif args.labels[i] == 'BAS':
        plt.annotate(args.labels[i], (args.X[1, i], args.X[0, i]), xycoords='data',
                    xytext=(102, 0.66),
                    arrowprops=dict(arrowstyle='-',
                                    connectionstyle='arc,angleA=-90,angleB=180,armA=30,armB=0,rad=0',
                                    color="0.4"),
                    fontsize=args.font_size)
    elif args.labels[i] == 'JCSOD':
        plt.text(args.X[1, i] + 2, args.X[0, i] - 0.01,
                args.labels[i],
                fontsize=args.font_size)
    elif args.labels[i] == 'SINet':
        plt.annotate(args.labels[i], (args.X[1, i], args.X[0, i]), xycoords='data',
                    xytext=(55, 0.62),
                    arrowprops=dict(arrowstyle='-',
                                    connectionstyle='arc,angleA=90,angleB=180,armA=30,armB=0,rad=0',
                                    color="0.4"),
                    fontsize=args.font_size)
    elif args.labels[i] == 'UGTR':
        plt.annotate(args.labels[i], (args.X[1, i], args.X[0, i]), xycoords='data',
                    xytext=(55+2, 0.718),
                    arrowprops=dict(arrowstyle='-',
                                    connectionstyle='arc,angleA=-90,angleB=180,armA=30,armB=0,rad=0',
                                    color="0.4"),
                    fontsize=args.font_size)
    elif args.labels[i] == 'TINet':
        plt.annotate(args.labels[i], (args.X[1, i], args.X[0, i]), xycoords='data',
                    xytext=(2, 0.69),
                    arrowprops=dict(arrowstyle='-',
                                    connectionstyle='arc,angleA=270,angleB=180,armA=30,armB=0,rad=0',
                                    color="0.4"),
                    fontsize=args.font_size)
    elif args.labels[i] == 'SINetV2':
        plt.text(args.X[1, i] + 3, args.X[0, i] - 0.01,
                args.labels[i],
                fontsize=args.font_size)
    elif args.labels[i] == 'C2FNet':
        plt.annotate(args.labels[i], (args.X[1, i], args.X[0, i]), xycoords='data',
                    xytext=(2, 0.7075),
                    arrowprops=dict(arrowstyle='-',
                                    connectionstyle='arc,angleA=90,angleB=180,armA=30,armB=0,rad=0',
                                    color="0.4"),
                    fontsize=args.font_size)
    elif args.labels[i] == 'LSR':
        plt.annotate(args.labels[i], (args.X[1, i], args.X[0, i]), xycoords='data',
                    xytext=(70, 0.708),
                    arrowprops=dict(arrowstyle='-',
                                    connectionstyle='arc,angleA=-90,angleB=180,armA=30,armB=0,rad=0',
                                    color="0.4"),
                    fontsize=args.font_size)

    else:
        plt.text(args.X[1, i], args.X[0, i],
                args.labels[i],
                fontsize=args.font_size)


def draw_lines(args):
    # Horizontal
    plt.annotate("",
                xy=(8.30, 0.728),
                xytext=(121.63, 0.728),
                arrowprops=dict(color="r",  headlength = 20, headwidth = 20, linewidth=1.5),
                )
    plt.annotate("",
                xy=(121.63, 0.728),
                xytext=(8.30, 0.728),
                arrowprops=dict(color="r",  headlength = 20, headwidth = 20, linewidth=1.5),
                )

    plt.vlines(x=8.30, ymin=0.747-0.03, ymax=0.754, color='r', linewidth=4.0)
    plt.vlines(x=121.63, ymin=0.728-0.01, ymax=0.728+0.01, color='r', linewidth=4.0)
    plt.text((121.63-8.30)/2+8.30, 0.743-0.01,
                    "113.3M",
                    fontsize=args.font_size,
            c='r',
            style='oblique',
            fontweight='bold'
            )

    # Vertical
    plt.annotate("",
                xy=(26.98+3, 0.769),
                xytext=(26.98+3, 0.743),
                arrowprops=dict(color="r",  headlength = 20, headwidth = 20, linewidth=1.5)
                )
    plt.annotate("",
                xy=(26.98+3, 0.743),
                xytext=(26.98+3, 0.769),
                arrowprops=dict(color="r",  headlength = 20, headwidth = 20, linewidth=1.5)
                )
    plt.hlines(y=0.769, xmin=21.02, xmax=26.98+5, color='r', linewidth=4.0)
    plt.hlines(y=0.743, xmin=26.98-5, xmax=26.98+5, color='r', linewidth=4.0)
    plt.text(21.02+3+8, (0.769-0.743)/2+0.743,
                    "2.6%",
                    fontsize=args.font_size,
            c='r',
            style='oblique',
            fontweight='bold'
            )

    plt.arrow(170, 0.53, 0, 0.08,
            width=3.5, head_width=8, head_length=0.016, color='gray',)
    plt.text(171, 0.55 ,
                    "The higher the better",
                    fontsize=args.font_size-10,
            c='gray',
            fontweight='bold',
            rotation=270
            )
    plt.arrow(170, 0.53, -50, 0,
            width=0.006, head_width=0.015, head_length=8, color='gray',)
    plt.text(122, 0.535,
                    "The lower the better",
                    fontsize=args.font_size-10,
            c='gray',
            fontweight='bold'
            )


def draw_engine(args):

    for i in range(args.X.shape[1]):
        
        draw_bubble(args, i)
        
        draw_text(args, i)

    draw_lines(args)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--font_size', default=38)
    parser.add_argument('--X', default=
                        np.array([[0.553, 0.604, 0.643, 0.642, 0.564, 0.640, 0.610, 0.613, 0.644, 0.663, 
                                   0.646, 0.695, 0.664, 0.673, 0.719, 0.678, 0.728, 0.696, 0.686, 0.743,
                                   0.754, 0.769], # Fw
                                   [29.23, 114.64, 25.23, 33.71, 25.54, 31.26, 26.47, 162.38, 48.95, 32.55, 
                                    87.06, 46.5, 63.6, 67.64, 28.41, 28.56, 121.63, 57.9, 48.87, 26.98, 
                                    8.30, 21.02] # Model Parameters (M)
                                    ]))
    parser.add_argument('--y', default=
                        np.arange(0, 22))
    parser.add_argument('--z', default=
                        np.array([29.23, 114.64, 25.23, 33.71, 25.54, 31.26, 26.47, 162.38, 48.95, 32.55, 
                                  87.06, 46.5, 63.6, 67.64, 28.41, 28.56, 121.63, 57.9, 48.87, 26.98,
                                  8.30, 21.02]), help='Bubble size')
    parser.add_argument('--colors', default=
                        ['cornflowerblue','lightsteelblue','#a2cffe','mediumslateblue','khaki','khaki','lavender','lavender', '#a2cffe','yellowgreen',
                         'mediumslateblue','gold','yellowgreen','gold','lemonchiffon','lemonchiffon', 'lemonchiffon','lightpink','lightpink','lightpink',
                         'orchid','orchid',])
    parser.add_argument('--labels', default=
                        ["CPD", "EGNet", "SCRN", "CSNet-R", "F3Net", "UCNet", "ITSD", "MINet-R", "SINet", "PraNet", 
                         "BAS", "PFNet", "S-MGL", "R-MGL", "C2FNet", "TINet", "JCSOD", "LSR", "UGTR", "SINetV2",
                         "DGNet-S", "DGNet"])
    args = parser.parse_args()

    plt.rcParams['font.family'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['figure.figsize'] = (15.5, 15)
    
    plt.rc('axes', axisbelow=True)

    draw_engine(args)

    plt.rc('axes', axisbelow=True)
    plt.grid(True, linestyle='--', linewidth=4, color='lightgray')
    plt.ylabel(r'Performance ($F^w_\beta$)', fontsize=args.font_size)

    plt.xlabel('Model Parameters (M)', fontsize=args.font_size)
    plt.tick_params(labelsize=args.font_size-5)
    plt.xlim(0, 170)
    plt.ylim(0.53, 0.8)

    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.spines['top'].set_linewidth(4)
    ax.spines['right'].set_linewidth(4)

    plt.tight_layout()

    plt.savefig('BubbleFig.pdf')
