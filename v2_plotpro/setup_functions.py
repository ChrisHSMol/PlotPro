import numpy as np, matplotlib.pyplot as plt
from iminuit import Minuit
from probfit import Chi2Regression
from scipy import stats


def rearrange_table(table, ref=0):
    table_2 = np.zeros_like(table)
    table_2[:, 0] = table[:, ref]
    table_2[:, 1:] = np.delete(table, ref, axis=1)
    return table_2


def bin_data(x, N_bins=100, xmin=0.0, xmax=1.0, density=False):
    """ Function that bins the input data given bins and returns the binned data.

    Parameters
    ----------
    x : array-like, shape=[N_numbers]
        Input points.
    bins : int, default 100
        Number of bins to use in the binning.
    xmin : float, default 0.0
        The minimum value of the range of the binning.
    xmax : float, default 1.0
        The maximum value of the range of the binning.

    Returns
    -------
    hist_x : array-like, shape=[bins]
        The centers of the bins.
    hist_y : array-like, shape=[bins]
        The entries of the bins.
    hist_sy : array-like, shape=[bins]
        The standard deviation of the bins.
    hist_mask : array-like, shape=[bins]
        Boolean mask with True where there are non-empty bins.
    """

    hist_y, hist_edges = np.histogram(x, bins=N_bins, range=(xmin, xmax), density=density)
    hist_x = 0.5 * (hist_edges[1:] + hist_edges[:-1])
    hist_sy = np.sqrt(hist_y)
    hist_mask = hist_y > 0

    return hist_x, hist_y, hist_sy, hist_mask


def label_fig(ax, x_label=None, y_label=None, title=None,
              label_fontsize=18, title_fontsize=16, legend_fontsize=18, tick_size=14,
              legend=True, loc=0, ncol=1, tight=True, legbox=True):
    """Put lables and titles on your plot real easy"""
    # ax.set(ylim=(ax.get_ylim()[0] * 0.9, ax.get_ylim()[1] * 1.1))
    ax.set_xlabel(x_label, fontsize=label_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.xaxis.set_tick_params(labelsize=tick_size)
    ax.yaxis.set_tick_params(labelsize=tick_size)
    if legend:
        ax.legend(fontsize=legend_fontsize, loc=loc, ncol=ncol, frameon=legbox)
    if tight:
        # plt.tight_layout()
        ax.set_tight


def dev_bars(list_of_mean_absolute_error, list_of_mean_error, std_error,
             ax, xtick_list, rot=0, width=0.25, full = False):
    """Generate grouped bar plots of statistical data"""
    legends = ["Mean absolute deviation", "Mean deviation", "Standard deviation"]
    """
    if full:
        fig, ax = plt.subplots(figsize=(12, 6))
        width = 0.2
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
    """
    n_data = np.max((len(list_of_mean_absolute_error), len(list_of_mean_error), len([std_error])))
    ind = np.arange(n_data)
    ax.bar(ind - width, list_of_mean_absolute_error, width)
    ax.bar(ind, list_of_mean_error, width)
    ax.bar(ind + width, std_error, width)
    plt.legend(legends, fontsize=14)
    ax.set_xticks(ind + width/2)
    plt.xticks(np.arange(len(list_of_mean_absolute_error)), xtick_list, fontsize=16, rotation = rot)
    ax.plot((-1, len(list_of_mean_error) + 1), (0, 0), alpha=0.2, color='black')
    ax.set(xlim=(-0.5, len(list_of_mean_error)-0.5))


def calc_mae(table, exp_index, theo_index, frac=False):
    """Calculate the mean absolute error (MAE) of each column in the given table. Returns a list of MAEs"""
    list_of_mae = []
    shp = np.shape(table)
    for j in range(theo_index, shp[1]):
        ae = []
        for i in range(shp[0]):
            if frac:
                abs_dif = abs((table[i, exp_index] - table[i, j]) / table[i, exp_index]) * 100
            else:
                abs_dif = abs(table[i, exp_index] - table[i, j])
            ae.append(abs_dif)
        list_of_mae.append(np.mean(ae))
    return list_of_mae


def calc_me(table, exp_index, theo_index, frac=False):
    """Calculate the mean error (ME) of each column in the given table. Returns a list of MEs"""
    list_of_me = []
    shp = np.shape(table)
    for j in range(theo_index, shp[1]):
        err = []
        for i in range(shp[0]):
            if frac:
                dif = (table[i, j] - table[i, exp_index]) / table[i, exp_index] * 100
            else:
                dif = table[i, j] - table[i, exp_index]
            err.append(dif)
        list_of_me.append(np.mean(err))
    return list_of_me


def calc_err(table, exp_index, theo_index, frac=False):
    """Calculate the mean error (ME) of each column in the given table. Returns a list of MEs"""
    shp = np.shape(table)
    for j in range(theo_index, shp[1]):
        err = []
        for i in range(shp[0]):
            if frac:
                dif = (table[i, j] - table[i, exp_index]) / table[i, exp_index] * 100
            else:
                dif = table[i, j] - table[i, exp_index]
            err.append(dif)
    return err


def calc_std(table, frac=False):
    """Calculate the standard error (SE) of each column in the given table. Returns a lsit of SEs."""
    list_of_er = []
    shp = np.shape(table)
    for j in range(1, shp[1]):
        err = []
        for i in range(shp[0]):
            if frac:
                dif = (table[i, j] - table[i, 0]) / table[i, 0] * 100
            else:
                dif = table[i, j] - table[i, 0]
            err.append(dif)
        list_of_er.append(err)
    std = []
    for col in range(len(list_of_er)):
        std.append(np.std(list_of_er[col]))
    if len(std) == 1:
        std = std[0]
    return std


# coding: utf-8


import numpy as np


def format_value(value, decimals):
    """
    Checks the type of a variable and formats it accordingly.
    Floats has 'decimals' number of decimals.
    """

    if isinstance(value, (float, np.float)):
        return f'{value:.{decimals}f}'
    elif isinstance(value, (int, np.integer)):
        return f'{value:d}'
    else:
        return f'{value}'


def values_to_string(values, decimals):
    """
    Loops over all elements of 'values' and returns list of strings
    with proper formating according to the function 'format_value'.
    """

    res = []
    for value in values:
        if isinstance(value, list):
            tmp = [format_value(val, decimals) for val in value]
            res.append(f'{tmp[0]} +/- {tmp[1]}')
        else:
            res.append(format_value(value, decimals))
    return res


def len_of_longest_string(s):
    """ Returns the length of the longest string in a list of strings """
    return len(max(s, key=len))


def nice_string_output(d, extra_spacing=5, decimals=3):
    """
    Takes a dictionary d consisting of names and values to be properly formatted.
    Makes sure that the distance between the names and the values in the printed
    output has a minimum distance of 'extra_spacing'. One can change the number
    of decimals using the 'decimals' keyword.
    """

    names = d.keys()
    max_names = len_of_longest_string(names)

    values = values_to_string(d.values(), decimals=decimals)
    max_values = len_of_longest_string(values)

    string = ""
    for name, value in zip(names, values):
        spacing = extra_spacing + max_values + max_names - len(name) - 1
        string += "{name:s} {value:>{spacing}} \n".format(name=name, value=value, spacing=spacing)
    return string[:-2]


def add_text_to_ax(x_coord, y_coord, string, ax, fontsize=12, color='k'):
    """ Shortcut to add text to an ax with proper font. Relative coords."""
    ax.text(x_coord, y_coord, string, family='monospace', fontsize=fontsize,
            transform=ax.transAxes, verticalalignment='top', color=color)
    return None


def calc_std_2(y1, y2):
    sqdif = []
    for i, j in zip(y1, y2):
        sqdif.append((i - j)**2)
    std = np.sqrt(np.mean(sqdif))
    return std


def subdim(number):
    """Generate the grid dimension of the subplots"""
    res = []
    for i in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        res.append(number % i)
        if number % i == 0:
            n = i
            m = number // i
            return n, m
    if not 0 in res:
        return subdim(number + 1)


def splitter(min, max):
    """Generate axis limits"""
    # minsub = min - 5
    # maxadd = max + 5
    if min > 0:
        mindiv = -1.0 * min - 10
    else:
        mindiv = np.sign(min) * np.abs(min) * 2
    maxmul = np.sign(max) * np.abs(max) * 1.5
    """
    if maxmul > max + 5:
        maxmul -= 5
    """
    if mindiv < min - 10:
        mindiv += 10
    # return minsub, mindiv, maxadd, maxmul
    return mindiv, maxmul


def axstep(min, max):
    dif = max-min
    if dif > 30:
        if dif > 100:
            if dif > 300:
                if dif > 500:
                    if dif > 2000:
                        return 1000
                    else:
                        return 200
                else:
                    return 50
            else:
                return 25
        else:
            return 10
    elif dif < 20:
        return 2
    elif dif > 20:
        return 5
    else:
        return 4


def barlocs(n, width=0.1, mean=0):
    """Distributes n amount of numbers separated by width to be centered around mean"""
    barloc = np.arange(n) * width
    if np.mean(barloc) > mean:
        barloc = barloc - (np.mean(barloc) - mean)
    elif np.mean(barloc) < mean:
        barloc = barloc + (np.mean(barloc) - mean)
    return barloc


def lha(list1, list2):
    min1, min2 = np.min(list1), np.min(list2)
    max1, max2 = np.max(list1), np.max(list2)
    step1, step2 = axstep(min1, max1), axstep(min2, max2)
    axmin1 = np.round(min1-0.5)
    axmin2 = np.round(min2-0.5)
    axmax1 = np.round(max1+0.5)
    axmax2 = np.round(max2+0.5)
    while axmin1 % step1 != 0:
        axmin1 += -1
    while axmin2 % step2 != 0:
        axmin2 += -1
    while axmax1 % step1 != 0:
        axmax1 += +1
    while axmax2 % step2 != 0:
        axmax2 += +1

    while (axmax2-axmin2)//step2 < (axmax1-axmin1)//step1:
        axmax2 += step2
    while (axmax2-axmin2)//step2 > (axmax1-axmin1)//step1:
        axmax1 += step1

    if axmin1 < 0:
        axmin1 -= np.mean((step1, step2))
        axmax1 += np.mean((step1, step2))
    else:
        axmax1 += step1 + step2

    if axmin2 < 0:
        axmin2 -= np.mean((step1, step2))
        axmax2 += 3 * np.mean((step1, step2))
        axmax1 += 2 * np.mean((step1, step2))
    else:
        axmax2 += step1 + step2
    return (axmin1, axmax1), (axmin2, axmax2)


def lha1(list1):
    min1, max1 = np.min(list1), np.max(list1)
    step1 = axstep(min1, max1)
    axmin1 = np.round(min1 - 0.5)
    axmax1 = np.round(max1 + 0.5)
    while axmin1 % step1 != 0:
        axmin1 += -1
    while axmax1 % step1 != 0:
        axmax1 += +1
    while (axmax1 - max1) < step1:
        axmax1 += step1
    return axmin1, axmax1


def determine_yrange(rangey, print_string):

    if isinstance(rangey, (tuple, list, np.ndarray)):
        if len(np.shape(rangey)) > 1:
            yrange = rangey
        else:
            yrange = (rangey, rangey)

    else:
        try:
            yrange = ast.literal_eval(rangey)

        except:
            yrange = ((0, 1), (0, 1))
            print_string += f"\n\n The input yrange of {rangey} was not interpreted properly." \
                            f"\n The range {yrange} has been employed.\n\n"

    return yrange, print_string


def calc_errs(table, ref=0, frac=False):

    if len(table) == 0:
        return table

    else:
        errs = np.zeros_like(table)
        for i in range(np.shape(table)[1]):
            if frac:
                errs[:, i] = (table[:, i] - table[:, ref]) / table[:, ref]
            else:
                errs[:, i] = table[:, i] - table[:, ref]
        return np.delete(errs, ref, axis=1)


def calc_stat(errs, stattype=""):
    if len(errs) == 0:
        return errs
    elif stattype == "mae":
        return calc_mae(errs)
    elif stattype == "me":
        return calc_me(errs)
    elif stattype == "std":
        return calc_std(errs)
    elif stattype == "maxerr":
        return calc_maxerr(errs)
    else:
        return errs


def calc_mae(errs):
    maes = np.zeros_like(errs[0, :])
    for i, col in enumerate(np.transpose(errs)):
        maes[i] = np.mean(np.abs(col))
    return maes


def calc_me(errs):
    mes = np.zeros_like(errs[0, :])
    for i, col in enumerate(np.transpose(errs)):
        mes[i] = np.mean(col)
    return mes


def calc_std(errs):
    stds = np.zeros_like(errs[0, :])
    for i, col in enumerate(np.transpose(errs)):
        stds[i] = np.std(col)
    return stds


def calc_maxerr(errs):
    maxerrs = np.zeros_like(errs[0, :])
    for i, col in enumerate(np.transpose(errs)):
        maxerrs[i] = np.max(np.abs(col))
    return maxerrs


def initiate_output():
    s = "\n" \
        "__/\\\\\\\\\\\\\\\\\\\\\\\\\\____/\\\\\\\\\\\\_________________________________/\\\\\\\\\\\\\\\\\\\\\\\\\\" \
        "______________________________________________\n"\
        "__\\/\\\\\\/////////\\\\\\_\\////\\\\\\________________________________\\/\\\\\\/////////\\\\\\" \
        "___________________________________________\n"\
        "___\\/\\\\\\_______\\/\\\\\\____\\/\\\\\\______________________/\\\\\\______\\/\\\\\\_______\\/\\\\\\" \
        "__________________________________________\n" \
        "____\\/\\\\\\\\\\\\\\\\\\\\\\\\\\/_____\\/\\\\\\________/\\\\\\\\\\_____/\\\\\\\\\\\\\\\\\\\\\\_\\/" \
        "\\\\\\\\\\\\\\\\\\\\\\\\\\/___/\\\\/\\\\\\\\\\\\\\______/\\\\\\\\\\" \
        "________________\n" \
        "_____\\/\\\\\\/////////_______\\/\\\\\\______/\\\\\\///\\\\\\__\\////" \
        "\\\\\\////__\\/\\\\\\/////////____\\/\\\\\\/////\\\\\\___/\\\\\\///\\\\\\" \
        "_____________\n" \
        "______\\/\\\\\\________________\\/\\\\\\_____/\\\\\\__\\//\\\\\\____\\/\\\\\\______\\/" \
        "\\\\\\_____________\\/\\\\\\___\\///___/\\\\\\__\\//\\\\\\" \
        "___________\n" \
        "_______\\/\\\\\\________________\\/\\\\\\____\\//\\\\\\__/\\\\\\_____\\/\\\\\\_/\\\\__\\/" \
        "\\\\\\_____________\\/\\\\\\_________\\//\\\\\\__/\\\\\\" \
        "___________\n" \
        "________\\/\\\\\\______________/\\\\\\\\\\\\\\\\\\__\\///\\\\\\\\\\/______\\//" \
        "\\\\\\\\\\___\\/\\\\\\_____________\\/\\\\\\__________\\///\\\\\\\\\\/" \
        "___________\n" \
        "_________\\///______________\\/////////_____\\/////_________\\/////____\\///______________\\///" \
        "_____________\\/////____________\n"
    return s


def update_default_bar_colours(legends):
    """
    Determine the new default bar colours based on the following site:
    https://learnui.design/tools/data-color-picker.html
    :param legends:     List of legends
    :return:
    """
    n_cols = len(legends)
    if n_cols == 4:
        legends = "{}:#003f5c;{}:#7a5195;{}:#ef5675;{}:#ffa600".format(legends[0], legends[1], legends[2], legends[3])
    elif n_cols == 3:
        legends = "{}:#003f5c;{}:#bc5090;{}:#ffa600".format(legends[0], legends[1], legends[2])
    elif n_cols == 2:
        legends = "{}:#003f5c;{}:#ffa600".format(legends[0], legends[1])
    elif n_cols == 1:
        legends = "{}:#003f5c".format(legends[0])
    return legends


def definitions():
    save_list = ["eps", "pdf", "pgf", "png", "ps", "raw", "rgba", "svg", "svgz"]

    keys_list = ["plot", "fit", "fit_plot", "deviation", "subplot", "ref", "uncertainty"]

    keys = {"plot": ["scatter", "bar", "dev"],
            "fit": ["linear", "exp"],
            "fit_plot": ["split", "line"],
            "deviation": ["mae", "me", "std", "maxerr", "scatter", "scatline", "ref", "type"],
            "subplot": ["sticky", "loose"],
            "ref": np.arange(0, 10),
            "uncertainty": np.arange(1.0E-5, 1.0E-4),
            "type": "rel"
            }

    keys_default = {"plot": "scatter", "fit": "", "fit_plot": "line",
                    "deviation": "", "subplot": "", "ref": 0, "uncertainty": 1, "type": "rel"}

    cosmetics_list = ["xaxis", "yaxis", "title", "legend", "model", "print_fit", "extrapolate",
                      "axisfontsize", "titlefontsize", "legendfontsize", "tickmarksize", "textfontsize",
                      "legloc", "rot", "pointsize", "tickalignment",
                      "linewidth", "legendbox", "legcolumns", "plotall", "diagonal", "gridlines", "yrange",
                      "datacolour", "fitcolour", "devcolour", "devpattern", "subplotframe", "figtype", "show_legend",
                      "devpointcol", "outdec", "tight", "nbarsep", "dpi", "background", "fade"]

    cosmstr = ["xaxis", "yaxis", "title", "legend", "devpattern", "subplotframe", "figtype", "show_legend", "print_fit",
               "legendbox", "plotall", "diagonal", "gridlines", "tickalignment", "tight", "extrapolate"]
    cosmval = ["axisfontsize", "titlefontsize", "legendfontsize", "tickmarksize", "rot", "pointsize", "linewidth",
               "legcolumns", "textfontsize", "outdec", "nbarsep", "dpi", "fade"]
    cosmcol = ["datacolour", "fitcolour", "devcolour", "legloc", "devpointcol", "background"]

    cosmetic_defaults = {"xaxis": "", "yaxis": "", "title": "", "legend": "", "model": "Model 1;Model 2;Model 3",
                         "axisfontsize": 18, "titlefontsize": 18, "legendfontsize": 18, "tickmarksize": 18, "rot": 0,
                         "datacolour": "k", "fitcolour": "r", "devcolour": "mae:b;me:r;std:g;maxerr:c",
                         "devpattern": "off",
                         "legloc": "upper right", 'subplotframe': "grid", "figtype": "rectangle", "legendbox": "on",
                         "show_legend": "on", "print_fit": "(0.05, 0.9)", "pointsize": 20, "linewidth": 1.0,
                         "plotall": "off", "diagonal": "off", "legcolumns": 1, "gridlines": "off", "textfontsize": 16,
                         "tickalignment": "center", "yrange": None, "devpointcol": "w", "outdec": 20, "tight": "on",
                         "nbarsep": 1, "dpi": None, "extrapolate": False, "background": "white", "fade": [1]}

    hatch_list = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

    return (save_list, keys_list, keys, keys_default, cosmetics_list, cosmstr, cosmval, cosmcol, cosmetic_defaults,
            hatch_list)
