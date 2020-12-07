#!/opt/anaconda3.7/bin/python3.7
import numpy as np
import matplotlib.pyplot as plt
import ast
from setup_functions import *


def plot_dev(data_table, keywords, cosmetics, print_string, format_dict, bool_dict, plot_name, output_name):
    """
    Major subroutine for performing the deviation plots.
    :param data_table:      array of raw data
    :param keywords:        dict of assigned routecard keywords
    :param cosmetics:       dict of assigned titlecard keywords
    :param print_string:    output string
    :param format_dict:     truncated version of the 'cosmetics' dict, featuring only formatting information
    :param bool_dict:       truncated version of the 'cosmetics' dict, featuring only boolean tests
    :param plot_name:       list of filenames for the resulting figure(s)
    :return:                returns the created figure, axis canvas, and the updated print_string
    """
    spac, dec, sep2 = format_dict["spac"], format_dict["dec"], format_dict["sep2"]
    """Start by opening the main deviation bar figure and re-structure the given data such that the reference
            column is at index 0 (for convenience)"""
    figbar, axbar = plt.subplots(figsize=(12, 6))

    if bool_dict["gridbool"]:
        plt.grid(axis=cosmetics['gridlines'])

    shp = np.shape(data_table)[1]

    """Input deviation bar colors written in such a way, that both strings and colour tuples
    are recognised properly"""
    barcols = {}

    for i in cosmetics["devcolour"].split(';'):
        c = i.split(":")

        try:
            barcols[c[0]] = ast.literal_eval(c[1])

        except:
            barcols[c[0]] = c[1]

    models = cosmetics["model"].split(";")

    errs, boolstats, statvals, legdict, limsbar = prepare_data(data_table, keywords["deviation"], keywords["frac"])

    """Set input parameters for things like bar width, rotation of labels and such"""
    n_bars = sum([legdict[key] != False for key in legdict.keys()])
    width = 1 / (sum([legdict[key] != False for key in legdict.keys()]) + cosmetics["nbarsep"])
    # n_bars = len(barcols)       # number of bars in each group
    rot = cosmetics['rot']
    n_groups = len(models)      # number of groups in the plot
    ind = np.arange(n_groups)

    """Draw the baseline for the bars (easily identifiable y=0 line)"""
    axbar.plot((-1, n_groups + 1), (0, 0), alpha=0.4, color='black', linewidth=2, zorder=1)

    if "scatter" in keywords["deviation"] or "scatline" in keywords["deviation"]:
        """If scatter errors are requested in deviation keyword, also draw the y=0 line for this data set"""
        axdotline = axbar.twinx()
        axbar.set_zorder(2)  # Puts axbar in front of axdotline
        axbar.patch.set_visible(False)  # Makes the axbar background transparent
        axdotline.tick_params(axis='y', right=False, labelright=False)  # Remove tickmarks and -labels on this axis
        axdotline.plot((-1, n_groups + 1), (0, 0), alpha=0.4, color='black', linewidth=2, zorder=1)

        if "scatline" in keywords["deviation"]:
            width = 1 / (len(legdict) + cosmetics["nbarsep"] + 1)
            n_bars += 1

        if len(cosmetics["yaxis"].split(";")) > 1:
            ybar, ydot = cosmetics['yaxis'].split(";")[:2]

        else:
            ybar, ydot = cosmetics['yaxis'].split(";")[0], ""

    else:
        ybar, ydot = cosmetics['yaxis'].split(";")[0], ""

    barloc = barlocs(n_bars, width=width)

    axbar, legs, counter = plot_devbars(axbar, ind, barloc, width, boolstats, statvals, legdict, barcols, cosmetics)

    """Place the tick labels on the x-axis and lock the x-axis to a symmetrical view"""
    plt.xticks(ind, models, fontsize=cosmetics["axisfontsize"])  # , rotation=100)
    plt.setp(axbar.xaxis.get_majorticklabels(), rotation=rot)

    for label in axbar.get_xticklabels():
        label.set_horizontalalignment(cosmetics["tickalignment"])

    axbar.set_xlabel(cosmetics['xaxis'], fontsize=cosmetics['axisfontsize'])
    axbar.set_ylabel(ybar, fontsize=cosmetics['axisfontsize'])
    axbar.set_title(cosmetics['title'], fontsize=cosmetics['titlefontsize'])
    axbar.xaxis.set_tick_params(labelsize=cosmetics['tickmarksize'])
    axbar.yaxis.set_tick_params(labelsize=cosmetics['tickmarksize'])

    print_string = print_deviations(data_table, print_string, keywords, models, spac, dec, sep2, output_name)

    scat_bool = False
    if "scatter" in keywords["deviation"] or "scatline" in keywords["deviation"]:
        scat_bool = True
        axdot = axbar.twinx()
        print_string += "Deviation scatter plot".center(len(sep2))
        print_string += f"\n{sep2}".ljust(len(sep2))
        axdot, axbar, axdotline, legs, print_string, axbarlims, axdotlims = overlay_scatter(
            axdot, axbar, axdotline,
            keywords, cosmetics,
            barloc, counter, legs,
            shp, ind, width,
            print_string, limsbar, ydot, errs
        )

        # --------------------------------------------------------------------------------------------------------------

    else:

        try:
            yrange, print_string = determine_yrange(cosmetics["yrange"], print_string)

            if len(np.shape(yrange)) > 1:
                yrange = yrange[0]

            axbarlims = yrange

        except:
            axbarlims = lha(limsbar, limsbar)[0]

    axbar.set(xlim=(-0.5, n_groups - 0.5), ylim=axbarlims)
    ysteps = axstep(axbarlims[0], axbarlims[1])
    axbar.yaxis.set_major_locator(plt.MultipleLocator(ysteps))
    axbar.yaxis.set_minor_locator(plt.MultipleLocator(0.5 * ysteps))

    """Add all legends to the figure legend"""
    leglabs = [l.get_label() for l in legs]

    if bool_dict["show_legend"]:

        if scat_bool:
            axdot.legend(legs, leglabs, loc=format_dict["legloc"], fontsize=cosmetics['legendfontsize'],
                         frameon=cosmetics["legendbox"],
                         ncol=int(cosmetics['legcolumns']))  # , zorder=axdot.get_zorder() + 1)
            print_string += "Deviation scatter plot has been constructed and overlayed the deviation bars."

        else:
            axbar.legend(legs, leglabs, loc=format_dict["legloc"], fontsize=cosmetics['legendfontsize'],
                         frameon=cosmetics["legendbox"],
                         ncol=int(cosmetics['legcolumns']))  # , zorder=axdot.get_zorder() + 1)

    if cosmetics["tight"]:
        plt.tight_layout()

    for name in plot_name:
        figbar.savefig(name, dpi=cosmetics["dpi"])

    return figbar, (axbar, axdot), print_string


def prepare_data(data_table, deviations, frac):
    """
    Calculate the arrays of MAE, ME, StD, and MaxErr when requested
    :param data_table:      Array of data
    :param deviations:      List of statistical deviation types to calculate
    :param frac:            Boolean, where True corresponds to performing a fractional calculation ( (x-ref)/ref )
    :return:                Returns an array of errors (points), a Boolean dict for each of the statistical parameters,
                            a dict of arrays with the deviations of the different kinds, a dict of legends, and a list
                            of limits to use in the bar plot.
    """
    legends = []
    statnames = ["mae", "me", "std", "maxerr"]
    statlegs = ["Mean Abs Dev", "Mean Dev", "Std Dev", "Max Abs Dev"]
    boolstats = {}
    statvals = {}
    legdict= {}

    for i, name in enumerate(statnames):

        if name in deviations:
            boolstats[name] = True
            legends.append(statlegs[i])
            legdict[name] = statlegs[i]

        else:
            boolstats[name] = False
            legdict[name] = False

    """Calculate the array of errors, the MAEs, MEs, STDs, and MaxErr"""
    limsbar = []
    errs = calc_errs(data_table, frac=frac)

    for name in statnames:

        if boolstats[name]:
            stat = calc_stat(errs, name)

            if len(stat) > 0:
                statvals[name] = stat
                limsbar.append(stat)

            else:
                statvals[name] = False
                limsbar.append(0.0)

        else:
            statvals[name] = False

    limsbar = [np.min(limsbar), np.max(limsbar)]

    return errs, boolstats, statvals, legdict, limsbar


def plot_devbars(ax, ind, barlocs, width, boolstats, statvals, legdict, barcols, cosmetics):
    """Plot bars using the provided bar colours (or default if none are specified)"""

    legs = ()
    counter = 0

    for name, hatch in zip(boolstats.keys(), cosmetics["devpattern"]):

        if boolstats[name]:
            leg = ax.bar(ind + barlocs[counter], statvals[name], width, label=legdict[name], color=barcols[name],
                         edgecolor='k', hatch=hatch, zorder=2)
            legs = legs + (leg,)
            counter += 1

    return ax, legs, counter


def overlay_scatter(axdot, axbar, axdotline, keywords, cosmetics, barloc, counter, legs, shp, ind, width, print_string,
                    limsbar, ydot, err):
    """If scatter errors have been requested, generate new axis to plot these"""
    axdot.set_zorder(axbar.get_zorder() + 1)  # Ensure that this new scatter axis is placed in front of axbar
    if "scatter" in keywords["deviation"]:

        if width == 0.5:
            width = 0.25

        for i in np.arange(shp - 1):
            """Generate an x-list for the scatter points to be spread out on"""
            x_scat = np.linspace(ind[i] - width, ind[i] + width, np.shape(err)[0])
            legdot = axdot.scatter(x_scat, err[:, i], color=cosmetics['devpointcol'], edgecolor='k',
                                   s=cosmetics['pointsize'],
                                   label="Deviations")

        legs = legs + (legdot,)

    if "scatline" in keywords["deviation"]:
        x_scat = np.ones_like(err) * np.arange(shp - 1) + barloc[counter]
        legdot = axdot.scatter(x_scat, err, color=cosmetics['devpointcol'], edgecolor='k',
                               s=cosmetics['pointsize'],
                               label="Deviations")

        legs = legs + (legdot,)

    """Fix the tickmarksize and scale the axdotline y-axis to match axdot (quick and dirty way to 
    draw the line behind the bars, but the dots in front of the bars - couldn't find a better way)"""
    limserr = [np.min(err), np.max(err)]

    if cosmetics["yrange"]:
        yrange, print_string = determine_yrange(cosmetics["yrange"], print_string)

        (axbarlims, axdotlims) = yrange

    else:
        axbarlims, axdotlims = lha(limsbar, limserr)

    if not ydot:
        yrange = [np.min((axbarlims, axdotlims)), np.max((axbarlims, axdotlims))]
        axbarlims, axdotlims = yrange, yrange

    ysteps = axstep(axdotlims[0], axdotlims[1])
    axdot.yaxis.set_major_locator(plt.MultipleLocator(ysteps))
    axdot.yaxis.set_minor_locator(plt.MultipleLocator(0.5 * ysteps))

    axdot.yaxis.set_tick_params(labelsize=cosmetics['tickmarksize'])
    axdot.set(ylim=axdotlims)
    axdot.set_ylabel(ylabel=ydot, fontsize=cosmetics['axisfontsize'], rotation=-90, labelpad=20)
    axdotline.set(ylim=axdot.get_ylim())

    return axdot, axbar, axdotline, legs, print_string, axbarlims, axdotlims


def print_deviations(data_table, print_string, keywords, models, spac, dec, sep2, output_name):
    from subroutines.parse_input import sanitate_data
    """
    Print both the fractional and numerical deviation statistical parameters
    :param boolstats:
    :param statvals:
    :param data_table:
    :param keywords:
    :param models:
    :param spac:
    :param dec:
    :param sep2:
    :return:
    """
    data_table, print_string = sanitate_data(
        data_table,
        keywords["ref"],
        output_name,
        print_string,
        allow_exit=False
    )
    individual_deviations = ""
    statistical_deviations = ""
    type_dict = {"True": "Fractional", "False": "Numeric"}

    for bfrac in [True, False]:
        err2, boolstats2, statvals2, _, _ = prepare_data(data_table, keywords["deviation"], bfrac)
        if len(err2) == 0:
            continue
        
        individual_deviations += f"{type_dict[str(bfrac)]} deviations, points".center(len(sep2))
        individual_deviations += f"\n{' ':11}\t{models[0]:>{spac}}"

        statistical_deviations += f"{type_dict[str(bfrac)]} deviations, statistics".center(len(sep2))
        statistical_deviations += f"\n{' ':11}\t{models[0]:>{spac}}"

        for imodel in np.arange(len(models) - 1) + 1:
            individual_deviations += f"\t{models[imodel]:>{spac}}"
            statistical_deviations += f"\t{models[imodel]:>{spac}}"

        for er in err2:
            individual_deviations += f"\n{' ':11}\t{er[0]:{spac}.{dec}f}"

            for imodel in np.arange(len(models) - 1) + 1:
                individual_deviations += f"\t{er[imodel]:{spac}.{dec}f}"

        individual_deviations += "\n\n"

        for name in boolstats2.keys():

            if boolstats2[name] and isvalid(statvals2[name]):
                statistical_deviations += f"\n{name.upper():11}\t{statvals2[name][0]:{spac}.{dec}f}"

                for imodel in np.arange(len(models) - 1) + 1:
                    statistical_deviations += f"\t{statvals2[name][imodel]:{spac}.{dec}f}"

        statistical_deviations += "\n\n"

    print_string += individual_deviations
    print_string += statistical_deviations

    return print_string


def isvalid(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return bool(len(value))
    else:
        return bool(value)


