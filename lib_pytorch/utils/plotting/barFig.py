import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def top_fig(args, gs, x, sorted_ele):
    plt.subplot(gs[0,0:])

    plt.bar(x, height=[float(ele[0]) for ele in sorted_ele],
            tick_label=[ele[1][4] for ele in sorted_ele],
            color=[ele[1][3] for ele in sorted_ele],
            edgecolor='black',
            linewidth=1.5)

    for xx, yy in zip(x, [ele[0] for ele in sorted_ele]):
        plt.text(xx, float(yy) + 10, str(yy), ha='center', fontsize=args.score_scale, rotation=args.score_rot)

    plt.ylabel('Parameters (M)', fontsize=args.font_size)
    plt.ylim(0, 250)
    plt.rc('axes', axisbelow=True)
    plt.grid(True, linestyle='--', linewidth=4, color='lightgray')
    plt.tick_params(labelsize=args.font_size-5)
    plt.xticks(rotation=90)
    plt.margins(0.01)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.spines['top'].set_linewidth(4)
    ax.spines['right'].set_linewidth(4)
    plt.tight_layout()


def mid_fig(args, gs, x, sorted_ele):
    plt.subplot(gs[1,0:])

    plt.ylabel('MACs (G)', fontsize=args.font_size)
    plt.bar(x,  height=[float(ele[1][1]) for ele in sorted_ele],
            tick_label=[ele[1][4] for ele in sorted_ele],
            color=[ele[1][3] for ele in sorted_ele],
            edgecolor='black',
            linewidth=1.5)
    plt.ylim(0, 790)


    for xx, yy in zip(x, [ele[1][1] for ele in sorted_ele]):
        plt.text(xx, float(yy) + 25, str(yy), ha='center', fontsize=args.score_scale, rotation=args.score_rot)

    plt.tick_params(labelsize=args.font_size-5)
    plt.xticks(rotation=90)
    plt.rc('axes', axisbelow=True)
    plt.grid(True, linestyle='--', linewidth=4, color='lightgray')
    plt.margins(0.01)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.spines['top'].set_linewidth(4)
    ax.spines['right'].set_linewidth(4)
    plt.tight_layout()
    

def low_fig(args, gs, x, sorted_ele):
    plt.subplot(gs[2, 0:])
    plt.ylabel(r'Performance ($F^w_\beta$)', fontsize=args.font_size)
    plt.bar(x, height=[float(ele[1][2]) for ele in sorted_ele],
            tick_label=[ele[1][4] for ele in sorted_ele],
            color=[ele[1][3] for ele in sorted_ele],
            edgecolor='black',
            linewidth=1.5
            )
    plt.ylim(0.5, 0.88)

    for xx, yy in zip(x, [ele[1][2] for ele in sorted_ele]):
        plt.text(xx, float(yy) + 0.015, str(yy), ha='center', fontsize=args.score_scale, rotation=args.score_rot)

    plt.rc('axes', axisbelow=True)
    plt.grid(True, linestyle='--', linewidth=4, color='lightgray')
    plt.tick_params(labelsize=args.font_size-5)
    plt.xticks(rotation=60)
    plt.margins(0.01)

    ax = plt.gca()

    ax.xaxis.set_ticks_position('both')

    labels = ax.get_xticklabels()
    [label.set_fontweight('bold') for label in labels[:2]]
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.spines['top'].set_linewidth(4)
    ax.spines['right'].set_linewidth(4)
    plt.tight_layout()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--font_size', default=38)
    parser.add_argument('--score_scale', default=33)
    parser.add_argument('--score_rot', default=90)
    parser.add_argument('--params', default=
                        np.array([29.23, 114.64, 25.23, 33.71, 25.54, 31.26, 26.47, 162.38, 48.95, 32.55, 
                                  87.06, '46.50', '63.60', 67.64, 28.41, 28.56, 121.63, '57.90', 48.87, 26.98,
                                  '8.30', 21.02]))
    parser.add_argument('--macs', default=
                        np.array([59.46, 527.97, 15.09, 6.05, 16.43, 16.11, 15.96, 87.11, 19.42, 13.11, 
                                  161.19, 26.54, 236.60, 249.89, 13.12, 8.58, '25.20', 25.21, 127.12, 12.28,
                                  '1.20', 2.77]))
    parser.add_argument('--f_ws', default=
                        np.array([0.553, 0.604, 0.643, 0.642, 0.564, '0.640', 0.610, 0.613, 0.644, 0.663, 
                                  0.646, 0.695, 0.664, 0.673, 0.719, 0.678, 0.728, 0.696, 0.686, 0.743,
                                  0.754, 0.769]))
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

    x = np.array([x*1 for x in range(len(args.params))])

    element = {}
    for index in range(len(args.params)):
        element[args.params[index]] = [args.macs[index], args.macs[index], args.f_ws[index], args.colors[index], args.labels[index]]

    sorted_ele = sorted(element.items(), key = lambda kv:(float(kv[0]), kv[1]))

    gs = gridspec.GridSpec(3, 1, hspace=0)

    top_fig(args, gs, x, sorted_ele)
    mid_fig(args, gs, x, sorted_ele)
    low_fig(args, gs, x, sorted_ele)    

    plt.savefig('BarFig.pdf')
