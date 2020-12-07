import numpy as np, matplotlib.pyplot as plt
from iminuit import Minuit
from probfit import Chi2Regression
from scipy import stats


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


def chi2_fit(function, x_values, y_values, unc_y, **start_parameters):
    """Chi2 Minuit fit with arbitrary function. Returns dict(values), dict(errors), chi2-value, prob(chi2)-value"""
    chi2_object = Chi2Regression(function, x_values, y_values, unc_y)
    minuit = Minuit(chi2_object, pedantic=False, print_level=0, **start_parameters)
    minuit.migrad()  # perform the actual fit
    if (not minuit.get_fmin().is_valid):
        print("  WARNING: The ChiSquare fit DID NOT converge!!!")
        conv = "No"
    else:
        conv = "Yes"
    minuit_output = [minuit.get_fmin(), minuit.get_param_states()]
    Ndof_fit = len(y_values) - len(start_parameters)
    prop_chi2 = stats.chi2.sf(minuit.fval, Ndof_fit)
    return minuit.values, minuit.errors, minuit.fval, Ndof_fit, prop_chi2, conv


def fit_linear(x, y, unc_y, a=1, b=0):
    def linear(x, a, b):
        return a * x + b
    fit_params, unc_fit_params, Chi2, Ndof, ProbChi2, conv = chi2_fit(linear, x, y, unc_y, a=a, b=b)
    # print(fit_params, unc_fit_params, Chi2, Ndof, ProbChi2, conv)
    a_fit, b_fit = fit_params['a'], fit_params['b']
    aunc_fit, bunc_fit = unc_fit_params['a'], unc_fit_params['b']
    x_fit  = x
    y_fit  = linear(x_fit, a_fit, b_fit)
    y_low  = linear(x_fit, a_fit - aunc_fit, b_fit)
    y_high = linear(x_fit, a_fit + aunc_fit, b_fit)
    r2 = 1 - (np.sum((y - linear(x, **fit_params))**2) /
                  np.sum((x - np.mean(x))**2))
    # print(f"y = {a_fit} x + {b_fit}\nR$^2$ = 1")
    # print(aunc_fit, bunc_fit)
    return x_fit, y_fit, y_low, y_high, a_fit, aunc_fit, b_fit, bunc_fit, r2, Chi2, Ndof, ProbChi2, conv


def fit_exp(x, y, unc_y, N=1, s=1, b=0):
    def exponential(x, N, s, b):
        return N * np.exp(s * x) + b
    fit_params, unc_fit_params, Chi2, Ndof, ProbChi2, conv = chi2_fit(exponential, x, y, unc_y, N=N, s=s, b=b)
    # print(fit_params, unc_fit_params, Chi2, Ndof, ProbChi2, conv)
    N_fit, s_fit, b_fit = fit_params['N'], fit_params['s'], fit_params['b']
    Nunc_fit, sunc_fit, bunc_fit = unc_fit_params['N'], unc_fit_params['s'], unc_fit_params['b']
    x_fit  = x
    y_fit  = exponential(x_fit, N_fit, s_fit, b_fit)
    y_low  = exponential(x_fit, N_fit, s_fit - sunc_fit, b_fit - bunc_fit)
    y_high = exponential(x_fit, N_fit, s_fit + sunc_fit, b_fit + bunc_fit)
    r2 = 1 - (np.sum((y - exponential(x, **fit_params))**2) /
                  np.sum((x - np.mean(x))**2))
    # print(sunc_fit, bunc_fit)
    return x_fit, y_fit, y_low, y_high, (N_fit, s_fit, b_fit), (Nunc_fit, sunc_fit, bunc_fit), r2, \
           Chi2, Ndof, ProbChi2, conv


def label_fig(ax, x_label=None, y_label=None, title=None,
              label_fontsize=18, title_fontsize=16, legend_fontsize=18, tick_size=14,
              legend=True, loc=0, ncol=1, tight=True, legbox=True):
    """Put lables and titles on your plot real easy"""
    # ax.set(ylim=(ax.get_ylim()[0] * 0.9, ax.get_ylim()[1] * 1.1))
    ax.set_xlabel(x_label, fontsize=label_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.xaxis.set_tick_params(labelsize = tick_size)
    ax.yaxis.set_tick_params(labelsize = tick_size)
    if legend:
        ax.legend(fontsize=legend_fontsize, loc=loc, ncol=ncol, frameon=legbox)
    if tight:
        plt.tight_layout()


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
        std.append(np.std(list_of_er[col], ddof=1))
    if len(std) == 1:
        std = std[0]
    return std


def calc_deviations(table, ref=0, frac=False):
    """
    Calculate individual deviations
    :param table:   2D array of points sorted by columns
    :param ref:     index of reference column (0-indexed) - default is 0
    :param frac:    Boolean: calculate deviations as fractions/relative deviations? Default is False
    :return:        2D array of deviations with same number of rows, but one fewer columns
    """
    dev_table = []
    for line in table:
        devline = []
        for i, val in enumerate(line):
            if not i == ref:
                if frac:
                    devline.append((val - line[ref]) / line[ref] * 100)
                else:
                    devline.append(val - line[ref])
        dev_table.append(devline)
    return dev_table


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
    if max-min > 30:
        if max-min > 100:
            if max-min > 500:
                if max-min > 2000:
                    return 1000
                else:
                    return 200
            else:
                return 25
        else:
            return 10
    elif max-min <= 10:
        return 2
    else:
        return 5


def plot_diagonal(x, y, ax, legend=True):
    # lims = [
    #     np.min([np.min(x), np.min(y)]),  # min of both axes
    #     np.max([np.max(x), np.max(y)]),  # max of both axes
    # ]
    lims = [-1E10, 1E10]
    if legend:
        ax.plot(lims, lims, color='black', alpha=0.5, label="y=x")
    else:
        ax.plot(lims, lims, color='black', alpha=0.5)


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
    return (axmin1, axmax1)


def initiate_output():
    """
    Print a fancy version of "PlotPro"
    """
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



