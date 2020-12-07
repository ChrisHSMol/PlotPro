#!/opt/anaconda3.7/bin/python3.7
import numpy as np
import matplotlib.pyplot as plt
import ast
from setup_functions import *


def plot_scatter(data_table, keywords, cosmetics, print_string, format_dict, bool_dict, plot_name):
    """
    The major subroutine for creating the scatter plots
    :param data_table:      array of data with n+1 columns, where n is the number of plots to make
    :param keywords:        dict of assigned routecard keywords
    :param cosmetics:       dict of assigned titlecard keywords
    :param print_string:    output string
    :param format_dict:     truncated version of the 'cosmetics' dict, featuring only formatting information
    :param bool_dict:       truncated version of the 'cosmetics' dict, featuring only boolean tests
    :param plot_name:       list of filenames for the resulting figure(s)
    :return:                returns the created figure, axis canvas, and the updated print_string
    """
    shp = np.shape(data_table)

    if len(cosmetics["yaxis"].split(";")) == 1:
        yaxes = [cosmetics["yaxis"] for i in np.arange(0, shp[0] - 1) + 1]
    else:
        yaxes = cosmetics["yaxis"].split(";")

    if len(cosmetics['legend'].split(";")) == 1:
        legends = [cosmetics['legend'] for i in np.arange(0, shp[0] - 1) + 1]
    else:
        legends = cosmetics['legend'].split(";")

    if len(cosmetics['fitcolour'].split(";")) == 1:
        fitcolours = [cosmetics['fitcolour'] for i in np.arange(0, shp[1] - 1) + 1]
    else:
        fitcolours = cosmetics['fitcolour'].split(";")

    x_list = data_table[:, 0]

    ndat = shp[1] - 1
    if keywords["subplot"]:
        bool_sub = True

        if cosmetics["subplotframe"] == "line":
            keywords["subplot"] = "loose"
            print_string += f"Dimensions of the line subplots is 1 row of {ndat} columns.\n\n"
            figtot, axtot = plt.subplots(figsize=(20, ndat + 1), nrows=1, ncols=ndat)
            figtot.subplots_adjust(wspace=0.4)
        elif cosmetics["subplotframe"] == "grid":
            dims = subdim(ndat)
            print_string += f"Dimensions of the grid subplot is {dims[0]} rows of {dims[1]} columns\n\n"
            figtot, axtot = plt.subplots(nrows=dims[0], ncols=dims[1],
                                         figsize=dims[0] * dims[1] * format_dict["figsize"])  # , sharey=True)#, sharex=True)

            if keywords["subplot"] == "sticky":
                figtot.subplots_adjust(hspace=0, wspace=0)
            else:
                figtot.subplots_adjust(hspace=0.3, wspace=0.1)

    else:
        bool_sub = False
        # figtot, axtot = plt.subplots(nrows=1, ncols=ndat+1)
        figtot, axtot = np.array([None, None]), np.array([None, None])

    for col, yax, leg, subax in zip(np.arange(0, shp[1] - 1) + 1, yaxes, legends, axtot.flatten()):
        y_list = data_table[:, col]
        figs, axs = plt.subplots(figsize=format_dict["figsize"])

        axes = np.array([axs, subax])
        mask = [True, bool_sub]
        axes = axes[mask]

        if bool_dict["gridbool"]:

            for ax in axes:
                ax.grid(axis=cosmetics['gridlines'])

        datacolour = cosmetics['datacolour'].split(";")[0]

        for ax in axes:
            ax.scatter(x_list, y_list, c=datacolour, label=leg, s=cosmetics['pointsize'], zorder=3)

        if cosmetics["diagonal"]:
            plot_diagonal(x_list, y_list, axes, bool_dict["show_legend"])

        if "fit" in keywords.keys():
            fitcolour = fitcolours[0]

            if "linear" in keywords['fit']:
                print_string = linear_fit(x_list, y_list, axes, keywords, cosmetics,
                                          format_dict, fitcolour, yax, print_string)

            if "exp" in keywords['fit']:
                print_string = exponential_fit(x_list, y_list, axes, keywords, cosmetics,
                                               format_dict, fitcolour, yax, print_string)

        for ax in axes:
            label_fig(ax, cosmetics['xaxis'], yax, cosmetics['title'], cosmetics['axisfontsize'],
                      cosmetics['titlefontsize'], cosmetics['legendfontsize'], cosmetics['tickmarksize'],
                      loc=format_dict["legloc"], ncol=int(cosmetics['legcolumns']), legend=cosmetics["show_legend"],
                      legbox=cosmetics["legendbox"], tight=False)
            figs.set_tight_layout(cosmetics["tight"])
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.tick_params(direction="inout")
            if cosmetics["extrapolate"]:
                xsteps = axstep(np.min((np.min(x_list), cosmetics["extrapolate"][0])),
                                np.max((np.max(x_list), cosmetics["extrapolate"][1])))
            else:
                xsteps = axstep(np.min(x_list), np.max(x_list))
            ysteps = axstep(np.min(y_list), np.max(y_list))
            ax.xaxis.set_major_locator(plt.MultipleLocator(xsteps))
            ax.yaxis.set_major_locator(plt.MultipleLocator(ysteps))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5 * xsteps))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5 * ysteps))

        if col % 2 == 0 and not cosmetics["subplotframe"] == "line":
            # Alternate between having y-axis labels on the left and right side of the subplot for non-linear plots
            subax.yaxis.tick_right()
            subax.yaxis.set_label_position("right")

        if cosmetics["subplotframe"] == "line":
            subax.set_aspect(1.0)

        if cosmetics["yrange"]:
            yrange = cosmetics["yrange"]
            if len(np.shape(yrange)) > 1:
                axs.set(xlim=yrange[0], ylim=yrange[1])
            else:
                axs.set(ylim=yrange)

        for name in plot_name:
            if keywords["subplot"]:
                nam, ext = name.split(".")
                figs.savefig(f"{nam}_{col}.{ext}", dpi=cosmetics["dpi"])
            else:
                figs.savefig(name, dpi=cosmetics["dpi"])

    if keywords["subplot"]:
        figtot.set_tight_layout(cosmetics["tight"])
        for name in plot_name:
            nam, ext = name.split(".")
            figtot.savefig(f"{nam}_{cosmetics['subplotframe']}.{ext}", dpi=cosmetics["dpi"])
    # else:
    #     plt.close(figtot)

    return figs, axs, print_string


def linear_fit(x_list, y_list, axes, keywords, cosmetics, format_dict, fitcolour, yax, print_string):
    """
    Perform and plot the linear fit
    :param x_list:          List of x-values
    :param y_list:          List of y-values
    :param ax:              The ax-canvas on which to plot
    :param keywords:        Dict of keywords as specified in the "read_input" and "add_defaults" functions
    :param cosmetics:       Dict of cosmetics as specified in the "read_input" and "add_defaults" functions
    :param format_dict:     Shorthand-dict of formatting specifications
    :param fitcolour:       Colour of the fitted line and, if requested, the area
    :param yax:             The name of the y-values as specified by the corresponding yaxis variable
    :param print_string:    The output print_string
    :return:                Returns the updated print_string. Updated with fitting model, input parameters,
                            and output parameters as obtained through Minuit
    """
    spac, dec = format_dict["spac"], format_dict["dec"]
    a_guess = (y_list[-1] - y_list[0]) / (x_list[-1] - x_list[0])
    b_guess = y_list[0] - a_guess * x_list[0]
    (x_fit, y_fit, y_low, y_high, a_fit, a_unc, b_fit, b_unc, r2,
     Chi2, Ndof, ProbChi2, conv) = fit_linear(x_list, y_list, np.zeros_like(y_list) + keywords['uncertainty'],
                                              a_guess, b_guess, cosmetics["extrapolate"])
    for ax in axes:
        ax.plot(x_fit, y_fit, c=fitcolour, label="Linear fit", linewidth=cosmetics['linewidth'], zorder=4)

    if format_dict["print_fit"]:
        for ax in axes:
            ax.text(format_dict["print_fit"][0], format_dict["print_fit"][1],
                    f"y = {a_fit:.3f} x + {b_fit:.3f}\n\tR$^2$ = {r2:.4f}",
                    transform=ax.transAxes, fontsize=cosmetics['textfontsize'], color=fitcolour,
                    va='center', ha='left')

    if "split" in keywords["fit_plot"]:
        for ax in axes:
            plt.fill_between(x_fit, y_high, y_low, color=fitcolour, alpha=0.1)
            ax.plot(x_fit, y_high, color=fitcolour, linestyle='--', alpha=0.4)
            ax.plot(x_fit, y_low, color=fitcolour, linestyle='--', alpha=0.4)

    print_string += f"\n\n\n\nFitting to:\t\t{yax}\n\n" \
                    f"Linear fit model:\ty = a * x + b\n\n" \
                    f"Input paramaters:\n" \
                    f"\ta:\t\t{a_guess:{spac}.{dec}f}\n" \
                    f"\tb:\t\t{b_guess:{spac}.{dec}f}\n\n" \
                    f"Fitted parameters:\n" \
                    f"\ta:\t\t{a_fit:{spac}.{dec}f} +/- {a_unc:{spac}.{dec}f}\t\t" \
                    f"({np.abs(a_unc/a_fit*100):12.6f} % relative deviation)\n" \
                    f"\tb:\t\t{b_fit:{spac}.{dec}f} +/- {b_unc:{spac}.{dec}f}\t\t" \
                    f"({np.abs(b_unc/b_fit*100):12.6f} % relative deviation)\n\n" \
                    f"\t{'R2:':12}\t\t{r2:{spac}.{dec}f}\n" \
                    f"\t{'STD:':12}\t\t{calc_std_2(y_list, y_fit):{spac}.{dec}f}\n" \
                    f"\t{'Chi2 / Ndof:':12}\t{Chi2:8.2f} / {Ndof:3}\n" \
                    f"\t{'p(Chi2):':12}\t{ProbChi2*100:8.2f}\n" \
                    f"\t{'Converged fit?':12}\t{conv:>8}\n\n\n\n"

    return print_string


def exponential_fit(x_list, y_list, axes, keywords, cosmetics, format_dict, fitcolour, yax, print_string):
    """
    Perform and plot the exponential fit
    :param x_list:          List of x-values
    :param y_list:          List of y-values
    :param axes:            List of ax-canvases on which to plot
    :param keywords:        Dict of keywords as specified in the "read_input" and "add_defaults" functions
    :param cosmetics:       Dict of cosmetics as specified in the "read_input" and "add_defaults" functions
    :param format_dict:     Shorthand-dict of formatting specifications
    :param fitcolour:       Colour of the fitted line and, if requested, the area
    :param yax:             The name of the y-values as specified by the corresponding yaxis variable
    :param print_string:    The output print_string
    :return:                Returns the updated print_string. Updated with fitting model, input parameters,
                            and output parameters as obtained through Minuit
    """
    spac, dec = format_dict["spac"], format_dict["dec"]
    N_guess, s_guess, b_guess = np.max(y_list), -np.log(np.abs(y_list[0] - y_list[1])) / np.abs(
        y_list[0] - y_list[1]) * 10, y_list[-1]
    x_fit, y_fit, y_low, y_high, (N_fit, s_fit, b_fit), (Nunc_fit, sunc_fit, bunc_fit), r2, \
    Chi2, Ndof, ProbChi2, conv = fit_exp(x_list, y_list,
                                          np.zeros_like(y_list) + keywords['uncertainty'],
                                          N_guess, s_guess, b_guess, cosmetics["extrapolate"])

    for ax in axes:
        ax.plot(x_fit, y_fit, c=fitcolour, label="Exponential fit", linewidth=cosmetics['linewidth'], zorder=4)

    if format_dict["print_fit"]:
        for ax in axes:
            ax.text(format_dict["print_fit"][0], format_dict["print_fit"][1],
                    f" y = {N_fit:.3f}* exp({s_fit:.3f} * x) + {b_fit:.3f}\nR$^2$ = {r2:.4f}",
                    transform=ax.transAxes, fontsize=cosmetics['textfontsize'], color=fitcolour,
                    va='center', ha='left')

    if "split" in keywords["fit_plot"]:
        for ax in axes:
            plt.fill_between(x_fit, y_high, y_low, color=fitcolour, alpha=0.1)
            ax.plot(x_fit, y_high, color=fitcolour, linestyle='--', alpha=0.4)
            ax.plot(x_fit, y_low, color=fitcolour, linestyle='--', alpha=0.4)

    print_string += f"\n\nFitting to:\t\t{yax}\n\n" \
                    f"Exponential fit model:\ty = N * exp(s * x) + b\n\n" \
                    f"Input paramaters:\n" \
                    f"\tN:\t{N_guess:{spac}.{dec}f}\n" \
                    f"\ts:\t{s_guess:{spac}.{dec}f}\n" \
                    f"\tb:\t{b_guess:{spac}.{dec}f}\n\n" \
                    f"Fitted parameters:\n" \
                    f"\tN:\t{N_fit:{spac}.{dec}f} +/- {Nunc_fit:{spac}.{dec}f}\t\t" \
                    f"({np.abs(Nunc_fit/N_fit*100):12.6f} % relative deviation)\n" \
                    f"\ts:\t{s_fit:{spac}.{dec}f} +/- {sunc_fit:{spac}.{dec}f}\t\t" \
                    f"({np.abs(sunc_fit/s_fit*100):12.6f} % relative deviation)\n" \
                    f"\tb:\t{b_fit:{spac}.{dec}f} +/- {bunc_fit:{spac}.{dec}f}\t\t" \
                    f"({np.abs(bunc_fit/b_fit*100):12.6f} % relative deviation)\n\n" \
                    f"\t{'R2:':12}\t\t{r2:{spac}.{dec}f}\n" \
                    f"\t{'STD:':12}\t\t{calc_std_2(y_list, y_fit):{spac}.{dec}f}\n" \
                    f"\t{'Chi2 / Ndof:':12}\t{Chi2:8.2f} / {Ndof:3}\n" \
                    f"\t{'p(Chi2):':12}\t{ProbChi2*100:8.2f}\n" \
                    f"\t{'Converged fit?':12}\t{conv:>8}\n\n\n\n"

    return print_string


def fit_linear(x, y, unc_y, a=1, b=0, extrapolate=False):
    """
    Perform the linear fit.
    :param x:               list of x-values
    :param y:               list of y-values
    :param unc_y:           list of uncertainties on y-values
    :param a:               initial value (guess) for the slope of the linear fit
    :param b:               initial value (guess) for the intercept of the linear fit
    :param extrapolate:     interval (min, max) on which to define the fitted function
    :return:                returns the fitted parameters, including the values for y=f(x) using the fitted function,
                            the lower and higher bounds of y using one standard deviation on the slope, the fitted
                            parameters and uncertainties of a and b, the R2 and Chi2 values, degrees of freedom
                            in the fit, probability of Chi2 value, and a string concluding whether the fit
                            converged or not.
    """
    def linear(x, a, b):
        return a * x + b
    fit_params, unc_fit_params, Chi2, Ndof, ProbChi2, conv = chi2_fit(linear, x, y, unc_y, a=a, b=b)
    a_fit, b_fit = fit_params['a'], fit_params['b']
    aunc_fit, bunc_fit = unc_fit_params['a'], unc_fit_params['b']
    if extrapolate:
        x_fit = np.linspace(extrapolate[0], extrapolate[1], 1000)
    else:
        x_fit = x
    y_fit = linear(x_fit, a_fit, b_fit)
    y_low = linear(x_fit, a_fit - aunc_fit, b_fit)
    y_high = linear(x_fit, a_fit + aunc_fit, b_fit)
    r2 = 1 - (np.sum((y - linear(x, **fit_params))**2) /
                  np.sum((x - np.mean(x))**2))
    return x_fit, y_fit, y_low, y_high, a_fit, aunc_fit, b_fit, bunc_fit, r2, Chi2, Ndof, ProbChi2, conv


def fit_exp(x, y, unc_y, N=1, s=-1, b=0, extrapolate=False):
    """

    :param x:               list of x-values
    :param y:               list of y-values
    :param unc_y:           list of uncertainties on y-values
    :param N:               initial guess of the scaling constant
    :param s:               initial guess of the decay rate
    :param b:               initial guess of the offset
    :param extrapolate:     interval (min, max) on which to define the fitted function
    :return:                returns the fitted parameters, including the values for y=f(x) using the fitted function,
                            the lower and higher bounds of y using one standard deviation on the slope, the fitted
                            parameters and uncertainties of N, s, and b, the R2 and Chi2 values, degrees of freedom
                            in the fit, probability of Chi2 value, and a string concluding whether the fit
                            converged or not.
    """
    def exponential(x, N, s, b):
        return N * np.exp(s * x) + b
    fit_params, unc_fit_params, Chi2, Ndof, ProbChi2, conv = chi2_fit(exponential, x, y, unc_y, N=N, s=s, b=b)
    N_fit, s_fit, b_fit = fit_params['N'], fit_params['s'], fit_params['b']
    Nunc_fit, sunc_fit, bunc_fit = unc_fit_params['N'], unc_fit_params['s'], unc_fit_params['b']
    if extrapolate:
        x_fit = np.linspace(extrapolate[0], extrapolate[1], 1000)
    else:
        x_fit = x
    y_fit = exponential(x_fit, N_fit, s_fit, b_fit)
    y_low = exponential(x_fit, N_fit, s_fit - sunc_fit, b_fit - bunc_fit)
    y_high = exponential(x_fit, N_fit, s_fit + sunc_fit, b_fit + bunc_fit)
    r2 = 1 - (np.sum((y - exponential(x, **fit_params))**2) / np.sum((x - np.mean(x))**2))
    return x_fit, y_fit, y_low, y_high, (N_fit, s_fit, b_fit), (Nunc_fit, sunc_fit, bunc_fit), r2, \
           Chi2, Ndof, ProbChi2, conv


def chi2_fit(function, x_values, y_values, unc_y, **start_parameters):
    """
    Perform the actual Chi2 Minuit fit to a custom function.
    :param function:            function defining the mathematical expression to fit the data to
    :param x_values:            list of x-values
    :param y_values:            list of y-values
    :param unc_y:               list of uncertainties on the y-values
    :param start_parameters:    an arbitrary list of start parameters. For linear fit, this would be the initial
                                a and b values
    :return:                    returns all fitted parameters (minuit.values), their corresponding uncertainties
                                (minuit.errors), the Chi2 value, the degrees of freedom, probability of the Chi2 value,
                                and a string concluding whether the fit converged or not.
                                Quick note on the use of Chi2 and associated probability: As a rule-of-thumb, each
                                data point should contribute approximately +1 to the Chi2 value - significantly larger
                                or smaller contributions are either unphysical or the result of relatively large
                                uncertainties on the y-values. Thus, these Chi2 and p(Chi2) values should not be
                                taken as be-all-end-all parameters for determining the quality of the fit.
    """
    # """Chi2 Minuit fit with arbitrary function. Returns dict(values), dict(errors), chi2-value, prob(chi2)-value"""
    chi2_object = Chi2Regression(function, x_values, y_values, unc_y)
    minuit = Minuit(chi2_object, pedantic=False, print_level=0, **start_parameters)
    minuit.migrad()  # perform the actual fit
    if not minuit.get_fmin().is_valid:
        print("  WARNING: The ChiSquare fit DID NOT converge!!!")
        conv = "No"
    else:
        conv = "Yes"
    Ndof_fit = len(y_values) - len(start_parameters)
    prop_chi2 = stats.chi2.sf(minuit.fval, Ndof_fit)
    return minuit.values, minuit.errors, minuit.fval, Ndof_fit, prop_chi2, conv


def plot_diagonal(x, y, axes, legend=True):
    lims = [
        -1E10,
        1E10
    #     np.min([np.min(x), np.min(y)]),  # min of both axes
    #     np.max([np.max(x), np.max(y)]),  # max of both axes
    ]
    for ax in axes:
        # axdotline.set(ylim=axdot.get_ylim())
        xl, yl = ax.get_xlim(), ax.get_ylim()
        if legend:
            ax.plot(lims, lims, color='black', alpha=0.5, label="y=x")
        else:
            ax.plot(lims, lims, color='black', alpha=0.5)
        ax.set(xlim=xl, ylim=yl)


