#!/opt/anaconda3.7/bin/python3.7
import numpy as np
import matplotlib.pyplot as plt
import ast
from setup_functions import *


def plot_bars(data_table, keywords, cosmetics, print_string, format_dict, bool_dict, plot_name):
    """
    The major subroutine for creating the bar plots
    :param data_table:      array of data with n+1 columns, where n is the number of plots to make
    :param keywords:        dict of assigned routecard keywords
    :param cosmetics:       dict of assigned titlecard keywords
    :param print_string:    output string
    :param format_dict:     truncated version of the 'cosmetics' dict, featuring only formatting information
    :param bool_dict:       truncated version of the 'cosmetics' dict, featuring only boolean tests
    :param plot_name:       list of filenames for the resulting figure(s)
    :return:                returns the created figure, axis canvas, and the updated print_string
    """
    barcols = {}
    barhatch = {}
    legends = cosmetics["legend"].split(";")
    models = cosmetics["model"].split(";")

    for i, icol in zip(range(len(cosmetics["devcolour"].split(';'))), cosmetics["devcolour"].split(';')):
        c = icol.split(":")
        barhatch[c[0]] = cosmetics["devpattern"][i]

        try:        # Attempt to parse RGB tuples (e.g. (1,0,0) for Red)
            barcols[c[0]] = ast.literal_eval(c[1])

        except:     # Else use the provided colour specification (e.g. "red" or "r")
            barcols[c[0]] = c[1]

    # Generate the width of the bars such that there is 'nbarsep' number of bars between each group of bars
    width = 1 / (len(data_table[0]) + cosmetics["nbarsep"])
    rot = cosmetics['rot']
    ind = np.arange(len(data_table))
    # nbars = len(barcols)
    nbars = len(legends)
    barloc = barlocs(nbars, width=width)
    fig, ax = plt.subplots(figsize=(12, 6))

    if bool_dict["gridbool"]:
        plt.grid(axis=cosmetics['gridlines'])

    for column, ibar, colorkey, leg in zip(np.transpose(data_table), np.arange(nbars), barcols.keys(), legends):
        ax.bar(ind + barloc[ibar], column, width, color=barcols[colorkey], label=leg, zorder=2,
               hatch=barhatch[colorkey], alpha=1)

    ax.plot((-1, len(ind) + 1), (0, 0), alpha=0.4, color='black', linewidth=2, zorder=1)

    if cosmetics["yrange"]:
        yrange, print_string = determine_yrange(cosmetics["yrange"], print_string)
        yrange = yrange[0]  # hotfix: determine_yrange spits out a 2D tuple, but 'bar' requires 1D tuple

    else:
        yrange = lha1((np.min(data_table), np.max(data_table)))

    # print(yrange, type(yrange))

    ax.set(xlim=(-0.5, len(ind) - 0.5), ylim=yrange)
    plt.xticks(ind, models, fontsize=cosmetics["axisfontsize"], rotation=rot, ha=cosmetics["tickalignment"])

    label_fig(ax, cosmetics['xaxis'], cosmetics['yaxis'], cosmetics["title"], cosmetics['axisfontsize'],
              cosmetics['titlefontsize'], cosmetics['legendfontsize'], cosmetics['tickmarksize'],
              loc=format_dict["legloc"], ncol=int(cosmetics['legcolumns']), tight=False,
              legend=cosmetics["show_legend"], legbox=cosmetics["legendbox"])
    fig.set_tight_layout(cosmetics["tight"])

    for name in plot_name:
        fig.savefig(name, dpi=cosmetics["dpi"])

    return fig, ax, print_string


