#!/opt/anaconda3.7/bin/python3.7
import numpy as np, matplotlib.pyplot as plt, os, sys, matplotlib as mpl
import regex as re, ast
from setup_functions import *

save_list = ["eps", "pdf", "pgf", "png", "ps", "raw", "rgba", "svg", "svgz"]

keys_list = ["plot", "fit", "fit_plot", "deviation", "subplot", "ref", "uncertainty"]

keys = {"plot": ["scatter", "bar", "dev"],
        "fit": ["linear", "exp"],
        "fit_plot": ["split", "line"],
        "deviation": ["mae", "me", "std", "maxerr", "scatter", "scatline", "ref", "type"],
        "subplot": ["sticky", "loose"],
        "ref": np.arange(1, 10),
        "uncertainty": np.arange(1.0E-5, 1.0E-4)
        }

keys_default = {"plot": "scatter", "fit": "", "fit_plot": "line",
                "deviation": "", "subplot": "", "ref": 1, "uncertainty": 1}

cosmetics_list = ["xaxis", "yaxis", "title", "legend", "model", "print_fit",
                  "axisfontsize", "titlefontsize", "legendfontsize", "tickmarksize", "textfontsize",
                  "legloc", "rot", "pointsize", "tickalignment",
                  "linewidth", "legendbox", "legcolumns", "plotall", "diagonal", "gridlines", "yrange",
                  "datacolour", "fitcolour", "devcolour", "devpattern", "subplotframe", "figtype", "show_legend",
                  "devpointcol", "outdec", "tight", "nbarsep", "dpi"]

cosmstr = ["xaxis", "yaxis", "title", "legend", "devpattern", "subplotframe", "figtype", "show_legend", "print_fit",
           "legendbox", "plotall", "diagonal", "gridlines", "tickalignment", "tight"]
cosmval = ["axisfontsize", "titlefontsize", "legendfontsize", "tickmarksize", "rot", "pointsize", "linewidth",
           "legcolumns", "textfontsize", "outdec", "nbarsep", "dpi"]
cosmcol = ["datacolour", "fitcolour", "devcolour", "legloc", "devpointcol"]

cosmetic_defaults = {"xaxis": "", "yaxis": "", "title": "", "legend": "", "model": "Model 1;Model 2;Model 3",
                     "axisfontsize": 18, "titlefontsize": 18, "legendfontsize": 18, "tickmarksize": 18, "rot": 0,
                     "datacolour": "k", "fitcolour": "r", "devcolour": "mae:b;me:r;std:g;maxerr:c", "devpattern": "off",
                     "legloc": "upper right", 'subplotframe': "grid", "figtype": "rectangle", "legendbox": "on",
                     "show_legend": "on", "print_fit": "(0.05, 0.9)", "pointsize": 20, "linewidth": 1.0,
                     "plotall": "off", "diagonal": "off", "legcolumns": 1, "gridlines": "off", "textfontsize": 16,
                     "tickalignment": "center", "yrange": None, "devpointcol": "w", "outdec": 20, "tight": "on",
                     "nbarsep": 2, "dpi": None}

hatch_list = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

# print_string = "\n\nRunning PlotPro"
print_string = initiate_output()
sep2 = "-"*35 + "="*50 + "-"*35 + "\n"*2
sep = "\n"*2 + sep2 + "\n"*2

# Read the file and generate lists of input arguments

test = False

if len(sys.argv) == 1 and not test:
    # print(f"\n\n\tERROR: This script must be called with an additional argument, ie: "
    #       f"\n\n\t\t\t python {sys.argv[0]} input.txt\n\n")
    print(f"\n\n\tERROR: This script must be called with an additional argument, ie: "
          f"\n\n\t\t\t plotpro input.txt\n\n")
    sys.exit()
else:
    if not test:
        filename = sys.argv[1]
    else:
        filenames = {"xy": "xy_test.txt", "exp": "exp_test.txt", "dev1": "deviation_test.txt",
                     "dev2": "deviation_test_2.txt", "sub1": "Datapyplot.txt", "sub2": "Datapyplot_subplots.txt",
                     "sub3": "Datapyplot_subplotframe.txt", "bar": "bar_test.txt", "jcc": "bond1_san.txt",
                     "sarah": "6g_GGA_C.txt", "hdz": "hdz_augj_all_frac.txt", "bar2": "abspic.txt"}
        filename = filenames['sarah']
    print_string += f"\n\n{'Input file':30} {filename:25}\n{'Current working directory':30} {os.getcwd()}{sep}"
    keywords = {}
    cosmetics = {}
    x_list = []
    y_list = []
    data_table = []
    for key in keys_list:
        keywords[key] = []

    output_name = f"{os.path.splitext(filename)[0]}.out"

    with open(filename, 'r') as inFile:
        plot_name = []
        frac = False
        # Read each line in the file, line by line
        for line in inFile:
            if line.startswith("%"):    # Look for savecard specifications
                for save in save_list:  # Look through the supported filetypes for sacing figures
                    reg = re.compile(f"({save}) ?= ?(.*?\.{save})")
                    match = reg.search(line.lower())
                    if match:   # If a given filetype is specified, append the specified filename to the list "plot_name"
                        plot_name.append(match.group(2))
            elif line.startswith("#"):  # Look for routecard specifications
                for key in keys_list:        # Iterate through all supported keys
                    if key in line.lower():
                        # If a supported key is found in the routecard, look through supported values
                        for val in keys[key]:
                            if key == "deviation":  # Special subroutine for deviation key due to the supported subkeys
                                reg = re.compile(f"({key}) ?= ?\((.*?)\)")
                                match = reg.search(line.lower())
                                if match:
                                    devs = match.group(2)#.split(',')
                                    for dev in keys[key]:
                                        if not dev in keywords[key]:
                                            # If the deviation keyword is not already in keywords, look for the deviation keyword
                                            reg = re.compile(f"({dev}) ?=? ?(\d*)")
                                            match = reg.search(devs.lower())
                                            if match:
                                                keywords[key].append(dev)
                                                if dev == "ref":
                                                    ref = int(match.group(2)) - 1   # Set the determined reference column
                                            reg2 = re.compile(f"({dev}) ?=? ?(frac|rel)")
                                            match = reg2.search(devs.lower())
                                            if match and dev == "type":
                                                if match.group(2) == "frac":
                                                    frac = True
                                                else:
                                                    frac = False
                                break
                            elif (key == "ref") or (key == "uncertainty"):  # Similarly, if the ref keyword is specifically given, set the ref to that
                                reg = re.compile(f" ({key}) ?= ?(\d*.?\d*)")
                                match = reg.search(line.lower())
                                if match:
                                    if key == "ref":
                                        keywords[key] = int(match.group(2))
                                    else:
                                        keywords[key] = float(match.group(2))
                            elif key == "subplot":  # Subplot keyword can similarly be subdivided
                                if key in line.lower():
                                    for keyval in keys[key]:
                                        if keyval in line.lower():
                                            keywords[key] = keyval
                                            break
                            # If the keyword is the furthest specification, continue here
                            else:
                                reg = re.compile(f" ({key}) ?= ?\(?({val}) ?=? ?(\w*)\)?")
                                match = reg.search(line.lower())
                                if match:
                                    keywords[key].append(val)
            elif line.startswith("$"):  # Look for titlecard specifications
                for cos in cosmetics_list:
                    if not cos in cosmetics.keys(): # Skip keywords that are already read in a different line
                        if "color" in line.lower():
                            line = re.sub('color', 'colour', line)
                        if (cos == "legloc") or (cos == "print_fit"): # Legloc needs special parsing syntax due to 3 different input types
                            reg = re.compile(f" ({cos}) ?= ?\"?(.*)\"?\s")
                            match = reg.search(line)
                            if match:
                                try:
                                    cosmetics[cos] = ast.literal_eval(match.group(2))   # Used to interpret tuples and floats
                                except:
                                    cosmetics[cos] = match.group(2) # Interpret strings
                        elif cos == "yrange":
                            reg = re.compile(f" ({cos}) ?= ?\"?(\(-?\d*,-?\d*\))\"?\s")
                            match = reg.search(line)
                            if match:
                                try:
                                    cosmetics[cos] = ast.literal_eval(match.group(2))
                                except:
                                    cosmetics[cos] = match.group(2)
                            else:
                                reg = re.compile(f" ({cos}) ?= ?\"?(\(\(-?\d*,-?\d*\),\(-?\d*,-?\d*\)\))\"?\s")
                                match = reg.search(line)
                                if match:
                                    try:
                                        cosmetics[cos] = ast.literal_eval(match.group(2))
                                    except:
                                        cosmetics[cos] = match.group(2)
                        regstr = re.compile(f" ({cos}) ?= ?\"(.*?)\"")   # Look for both string-specific and
                        matchstr = regstr.search(line)
                        regval = re.compile(f" ({cos}) ?= ?([+-]?\d*\.?\d*)")    # float-specific RegExs
                        matchval = regval.search(line)
                        if matchstr:
                            cosmetics[cos] = matchstr.group(2)
                        elif matchval:
                            try:
                                cosmetics[cos] = int(matchval.group(2))
                            except:
                                cosmetics[cos] = float(matchval.group(2))
            elif (line[0].isdigit()) or (line[0] == '-') or (line[0] == '+'):   # Look for data given by either a digit or a minus sign
                line = line.split()
                data_line = []
                for number in line:
                    data_line.append(float(number))
                data_table.append(data_line)
        # data_table = np.array(data_table)   # Convert data to NumPy array for ease of use
    if len(plot_name) > 0:
        print_string += f"Figure will be saved as {plot_name[0]}"
        for iname in np.arange(len(plot_name)-1)+1:
            print_string += f", {plot_name[iname]}"
    else:
        print_string += f"Figure will not be saved by ths program."
    print_string += f"{sep}"

    print_string += "Input parameters".center(len(sep2))
    print_string += f"\n{sep2}".ljust(len(sep2))
    print_string += "routecard:\n"
    for key in keywords:
        if type(keywords[key]) == type(keys_list):
            if len(keywords[key]) == 1:
                print_string += "\t{0:<20} {1}\n".format(key, keywords[key][0])
            elif len(keywords[key]) > 1:
                print_string += "\t{0:<20} ".format(key)
                for i in np.arange(len(keywords[key])):
                    if i == len(keywords[key])-1:
                        print_string += f"{keywords[key][i]}\n"
                    else:
                        print_string += f"{keywords[key][i]}, "
            else:
                continue
        else:
            print_string += "\t{0:<20} {1}\n".format(key, keywords[key])

    print_string += "\n\ntitlecard:\n"
    for key in cosmetics:
        print_string += "\t{0:<20} {1}\n".format(key, cosmetics[key])
    print_string += sep


if frac:
    data_backup = np.array(data_table)
    datasan = []
    thresh = 1.E-10
    for line in data_table:
        if np.abs(line[ref]) < thresh:
            print_string += f"Near-zero reference value found:"
            for i in line:
                print_string += f"\t{i:10}"
            print_string += f"\n\t\tThreshold:\t{thresh}\n\t\tPoint has been removed\n\n"
        else:
            datasan.append(line)
    data_table = np.array(datasan)
    if len(data_table) == 0:
        print_string += f"Data table has been scrubbed as all reference values were below the threshold of {thresh}.\n" \
                        f"Exiting program."
        with open(output_name, 'w') as outFile:
            outFile.write(print_string)
        sys.exit()
else:
    data_table = np.array(data_table)



# Input default values
for key in cosmetics_list:
    if not key in cosmetics.keys():
        cosmetics[key] = cosmetic_defaults[key]
    if key == "devpattern": # Specify the bar plot hatch patterns
        if cosmetics[key] == "on":
            hatches = ["-", "+", "/", "x"]
        elif cosmetics[key] == "off":
            hatches = [None, None, None, None]
        elif cosmetics[key].split(";")[0] in hatch_list:
            hatches = []
            for hatch in cosmetics[key].split(";"):
                if hatch in hatch_list:
                    hatches.append(hatch)
                else:
                    hatches.append(None)
                    print(f"Invalid hatch value: \"{hatch}\"\nNo hatch is used in its place")
                    print_string += f"Invalid hatch value \"{hatch}\" read.\n" \
                                    f"Available values are \"{hatch_list[0]}\""
                    for i in np.arange(len(hatch_list)-1)+1:
                        print_string += f", \"{hatch_list[i]}\""
                    print_string += "\n\n\n\n"
            if len(hatches) < 4:
                while len(hatches) < 4:
                    hatches.append(None)
        else:
            print(f"Invalid value for the \"{key}\" titlecard keyword. "
                  f"The default value of \"{cosmetic_defaults[key]}\" is used.")
            hatches = [None, None, None, None]
    elif key == "show_legend":
        if cosmetics[key] == "on":
            show_legend = True
        elif cosmetics[key] == "off":
            show_legend = False
        else:
            print(f"Invalid value for the \"{key}\" titlecard keyword. "
                  f"The default value of \"{cosmetic_defaults[key]}\" is used.")
            show_legend = True
    elif key == "legendbox":
        if cosmetics[key] == "on":
            legbox = True
        elif cosmetics[key] == "off":
            legbox = False
        else:
            print(f"Invalid value for the \"{key}\" titlecard keyword. "
                  f"The default value of \"{cosmetic_defaults[key]}\" is used.")
            legbox = True
    elif key == "diagonal":
        if cosmetics[key] == "on":
            diagonal = True
        elif cosmetics[key] == "off":
            diagonal = False
        else:
            print(f"Invalid value for the \"{key}\" titlecard keyword. "
                  f"The default value of \"{cosmetic_defaults[key]}\" is used.")
            diagonal = False
    elif key == "gridlines":
        if cosmetics[key] == "off":
            gridbool = False
        elif (cosmetics[key] == "x") or (cosmetics[key] == "y") or (cosmetics[key] == "both"):
            gridbool = True
        else:
            print(f"Invalid value for the \"{key}\" titlecard keyword. "
                  f"The default value of \"{cosmetic_defaults[key]}\" is used.")
            gridbool = False
    elif key == "plotall":
        if cosmetics[key] == "on":
            plotall = True
        elif cosmetics[key] == "off":
            plotall = False
        else:
            print(f"Invalid value for the \"{key}\" titlecard keyword. "
                  f"The default value of \"{cosmetic_defaults[key]}\" is used.")
            plotall = False
    elif key == "tight":
        if cosmetics[key] == "on":
            tightbool = True
        elif cosmetics[key] == "off":
            tightbool = False
        else:
            print(f"Invalid value for the \"{key}\" titlecard keyword. "
                  f"The default value of \"{cosmetic_defaults[key]}\" is used.")
            tightbool = True
        cosmetics[key] = tightbool
    elif key == "print_fit":
        if cosmetics[key] == "off":
            print_fit = False
        elif cosmetics[key] == "on":
            print_fit = (0.05, 0.9)
        else:
            try:
                print_fit = ast.literal_eval(cosmetics[key])  # Literal evaluation of tuples and floats
            except:
                print_fit = cosmetics[key].lower()

if cosmetics["figtype"] == "square":    # Set the figure size based on input (square or rectangular)
    figsize = np.array((6, 6))
else:
    figsize = np.array((12, 6))

# Input default values for keywords
for key in keys_list:
    if not key in keywords:
        keywords[key] = keys_default[key]
    if not keywords[key]:
        keywords[key] = keys_default[key]

# Cast the input legloc position into a variable for ease of use
try:
    legloc = ast.literal_eval(cosmetics['legloc'])  # Literal evaluation of tuples and floats
except:
    legloc = cosmetics['legloc'].lower()    # Force strings into lower case

print_string += "Used parameters".center(len(sep2))
print_string += f"\n{sep2}".ljust(len(sep2))
print_string += "routecard:\n"
for key in keywords:
    if type(keywords[key]) == type(keys_list):
        if len(keywords[key]) == 1:
            print_string += "\t{0:<20} {1}\n".format(key, keywords[key][0])
        elif len(keywords[key]) > 1:
            print_string += "\t{0:<20} ".format(key)
            for i in np.arange(len(keywords[key])):
                if i == len(keywords[key])-1:
                    print_string += f"{keywords[key][i]}\n"
                else:
                    print_string += f"{keywords[key][i]}, "
        else:
            continue
    else:
        print_string += "\t{0:<20} {1}\n".format(key, keywords[key])

print_string += "\n\ntitlecard:\n"
for key in cosmetics:
    if (key == "devpattern") and cosmetics[key] == "on":
        print_string += "\t{0:<20} {1}\t\t\"".format(key, cosmetics[key])
        for i, hat in enumerate(hatches):
            if not i == len(hatches) - 1:
                print_string += "{0};".format(hat)
            else:
                print_string += "{0}\"\n".format(hat)
    else:
        print_string += "\t{0:<20} {1}\n".format(key, cosmetics[key])
print_string += sep









# Start plotting
if "plot" in keywords.keys():
    spac, dec = cosmetics["outdec"] + 5, cosmetics["outdec"]
    if "scatter" in keywords["plot"]:
        print_string += "Scatter plot".center(len(sep2))
        print_string += f"\n{sep2}".ljust(len(sep2))
        ref = keywords["ref"] - 1
        shp = np.shape(data_table)
        data_table_2 = np.zeros_like(data_table)
        data_table_2[:, 0] = data_table[:, ref]
        data_table_2[:, 1:] = np.delete(data_table, ref, axis = 1)

        if len(cosmetics["yaxis"].split(";")) == 1:
            yaxes = [cosmetics["yaxis"] for i in np.arange(0, shp[0]-1)+1]
        else:
            yaxes = cosmetics["yaxis"].split(";")
        if len(cosmetics['legend'].split(";")) == 1:
            legends = [cosmetics['legend'] for i in np.arange(0, shp[0]-1)+1]
        else:
            legends = cosmetics['legend'].split(";")
        if len(cosmetics['fitcolour'].split(";")) == 1:
            fitcolours = [cosmetics['fitcolour'] for i in np.arange(0, shp[1] - 1) + 1]
        else:
            fitcolours = cosmetics['fitcolour'].split(";")

        x_list = data_table_2[:, 0]
        for col, yax, leg in zip(np.arange(0, shp[1] - 1) + 1, yaxes, legends):
            y_list = data_table_2[:, col]
            fig, ax = plt.subplots(figsize=figsize)
            if gridbool:
                plt.grid(axis=cosmetics['gridlines'])
            datacolour = cosmetics['datacolour'].split(";")[0]
            ax.scatter(x_list, y_list, c=datacolour, label=leg, s=cosmetics['pointsize'], zorder=3)
            if diagonal:
                plot_diagonal(x_list, y_list, ax, show_legend)
            if "fit" in keywords.keys():
                fitcolour = fitcolours[0]
                if "linear" in keywords['fit']:
                    a_guess = (y_list[-1] - y_list[0]) / (x_list[-1] - x_list[0])
                    b_guess = y_list[0] - a_guess * x_list[0]
                    (x_fit_lin, y_fit_lin, y_low_lin, y_high_lin, a_fit, a_unc, b_fit, b_unc, r2_lin,
                     Chi2_lin, Ndof_lin, ProbChi2_lin, conv_lin) = fit_linear(x_list, y_list, np.zeros_like(y_list) + keywords['uncertainty'], a_guess, b_guess)
                    ax.plot(x_fit_lin, y_fit_lin, c=fitcolour, label="Linear fit", linewidth=cosmetics['linewidth'],
                            zorder=4)
                    if print_fit:
                        ax.text(print_fit[0], print_fit[1], f"y = {a_fit:.3f} x + {b_fit:.3f}\n\tR$^2$ = {r2_lin:.4f}",
                                transform=ax.transAxes, fontsize=cosmetics['textfontsize'], color=fitcolour,
                                va='center', ha='left')
                    if "split" in keywords["fit_plot"]:
                        plt.fill_between(x_fit_lin, y_high_lin, y_low_lin, color=fitcolour, alpha=0.1)
                        ax.plot(x_fit_lin, y_high_lin, color=fitcolour, linestyle='--', alpha=0.4)
                        ax.plot(x_fit_lin, y_low_lin, color=fitcolour, linestyle='--', alpha=0.4)
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
                                    f"\tR2:\t\t{r2_lin:{spac}.{dec}f}\n" \
                                    f"\tSTD:\t\t{calc_std_2(y_list, y_fit_lin):{spac}.{dec}f}\n" \
                                    f"\tChi2 / Ndof:\t{Chi2_lin:8.2f} / {Ndof_lin:3}\n" \
                                    f"\tp(Chi2):\t{ProbChi2_lin*100:8.2f}\n" \
                                    f"\tConverged fit?\t{conv_lin:>8}\n\n\n\n"
                if "exp" in keywords['fit']:
                    N_guess, s_guess, b_guess = np.max(y_list), -np.log(np.abs(y_list[0] - y_list[1])) / np.abs(
                        y_list[0] - y_list[1]) * 10, y_list[-1]
                    x_fit_exp, y_fit_exp, y_low_exp, y_high_exp, fitpar, uncpar, r2_exp, \
                    Chi2_exp, Ndof_exp, ProbChi2_exp, conv_exp = fit_exp(x_list, y_list, np.zeros_like(y_list) + keywords['uncertainty'], N_guess, s_guess, b_guess)
                    ax.plot(x_fit_exp, y_fit_exp, c=fitcolour, label="Exponential fit", linewidth=cosmetics['linewidth'],
                            zorder=4)
                    if "split" in keywords["fit_plot"]:
                        plt.fill_between(x_fit_exp, y_high_exp, y_low_exp, color=fitcolour, alpha=0.1)
                        ax.plot(x_fit_exp, y_high_exp, color=fitcolour, linestyle='--', alpha=0.4)
                        ax.plot(x_fit_exp, y_low_exp, color=fitcolour, linestyle='--', alpha=0.4)
                    print_string += f"\n\nFitting to:\t\t{yax}\n\n" \
                                    f"Exponential fit model:\ty = N * exp(s * x) + b\n\n" \
                                    f"Input paramaters:\n" \
                                    f"\tN:\t{N_guess:{spac}.{dec}f}\n" \
                                    f"\ts:\t{s_guess:{spac}.{dec}f}\n" \
                                    f"\tb:\t{b_guess:{spac}.{dec}f}\n\n" \
                                    f"Fitted parameters:\n" \
                                    f"\tN:\t{fitpar[0]:{spac}.{dec}f} +/- {uncpar[0]:{spac}.{dec}f}\t\t" \
                                    f"({np.abs(uncpar[0]/fitpar[0]*100):12.6f} % relative deviation)\n" \
                                    f"\ts:\t{fitpar[1]:{spac}.{dec}f} +/- {uncpar[1]:{spac}.{dec}f}\t\t" \
                                    f"({np.abs(uncpar[1]/fitpar[1]*100):12.6f} % relative deviation)\n" \
                                    f"\tb:\t{fitpar[2]:{spac}.{dec}f} +/- {uncpar[2]:{spac}.{dec}f}\t\t" \
                                    f"({np.abs(uncpar[2]/fitpar[2]*100):12.6f} % relative deviation)\n\n" \
                                    f"\tR2:\t\t{r2_exp:{spac}.{dec}f}\n" \
                                    f"\tSTD:\t\t{calc_std_2(y_list, y_fit_exp):{spac}.{dec}f}\n" \
                                    f"\tChi2 / Ndof:\t{Chi2_exp:8.2f} / {Ndof_exp:3}\n" \
                                    f"\tp(Chi2):\t{ProbChi2_exp*100:8.2f}\n" \
                                    f"\tConverged fit?\t{conv_exp:>8}\n\n\n\n"
            
            if cosmetics["yrange"]:
                try:
                    yrange = ast.literal_eval(cosmetics["yrange"])
                except:
                    yrange = ((0, 1), (0, 1))
                    print_string += f"\n\n The input yrange of {cosmetics['yrange']} was not interpreted properly." \
                                    f"\n The range {yrange} has been employed.\n\n"
                if len(np.shape(yrange)) > 1:
                    ax.set(xlim=yrange[0], ylim=yrange[1])
                else:
                    ax.set(ylim=yrange)

            label_fig(ax, cosmetics['xaxis'], yax, cosmetics['title'], cosmetics['axisfontsize'],
                      cosmetics['titlefontsize'], cosmetics['legendfontsize'], cosmetics['tickmarksize'],
                      loc=legloc, ncol=int(cosmetics['legcolumns']), legend=show_legend, legbox=legbox,
                      tight=tightbool)
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.tick_params(direction="inout")
            ax.xaxis.set_major_locator(plt.MultipleLocator(axstep(np.min(x_list), np.max(x_list))))
            ax.yaxis.set_major_locator(plt.MultipleLocator(axstep(np.min(y_list), np.max(y_list))))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5*axstep(np.min(x_list), np.max(x_list))))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5*axstep(np.min(y_list), np.max(y_list))))

        if plotall:
            print_string += f"{sep}Plotting all data lists in one figure.\n\n{sep}"
            fig, ax = plt.subplots(figsize=figsize)
            if diagonal:
                plot_diagonal(x_list, y_list, ax, show_legend)
            if len(cosmetics['datacolour'].split(";")) == 1:
                datacolours = [cosmetics['datacolour'] for i in np.arange(0, shp[1]-1)+1]
            else:
                datacolours = cosmetics['datacolour'].split(";")
            if len(cosmetics['fitcolour'].split(";")) == 1:
                fitcolours = [cosmetics['fitcolour'] for i in np.arange(0, shp[1] - 1) + 1]
            else:
                fitcolours = cosmetics['fitcolour'].split(";")
            if len(cosmetics['legend'].split(";")) == 1:
                legends = [cosmetics['legend'] for i in np.arange(0, shp[1] - 1) + 1]
            # elif len(cosmetics['legend'].split(";")) == 0:
            else:
                legends = cosmetics['legend'].split(";")
            ydec = 0.0
            if gridbool:
                plt.grid(axis=cosmetics['gridlines'])
            for datacolour, fitcolour, col, yax, leg in zip(datacolours, fitcolours, np.arange(0, shp[1] - 1) + 1,
                                                            yaxes, legends):
                y_list = data_table_2[:, col]
                ax.scatter(x_list, y_list, c=datacolour, label=leg, s=cosmetics['pointsize'], zorder=3)
                if "fit" in keywords.keys():
                    if "linear" in keywords['fit']:
                        a_guess = (y_list[-1] - y_list[0]) / (x_list[-1] - x_list[0])
                        b_guess = y_list[0] - a_guess * x_list[0]
                        (x_fit_lin, y_fit_lin, y_low_lin, y_high_lin, a_fit, a_unc, b_fit, b_unc, r2_lin,
                         Chi2_lin, Ndof_lin, ProbChi2_lin, conv_lin) = fit_linear(x_list, y_list, np.zeros_like(y_list) + keywords['uncertainty'], a_guess, b_guess)
                        ax.plot(x_fit_lin, y_fit_lin, c=fitcolour, label="Linear fit", linewidth=cosmetics['linewidth'],
                                zorder=4)
                        if print_fit:
                            ax.text(print_fit[0], print_fit[1]+ydec,
                                    f"y = {a_fit:.3f} x + {b_fit:.3f}\n\tR$^2$ = {r2_lin:.4f}",
                                    transform=ax.transAxes, fontsize=cosmetics['textfontsize'], color=fitcolour,
                                    va='center', ha='left')
                            ydec += -0.15
                        if "split" in keywords["fit_plot"]:
                            plt.fill_between(x_fit_lin, y_high_lin, y_low_lin, color=fitcolour, alpha=0.1)
                            ax.plot(x_fit_lin, y_high_lin, color=fitcolour, linestyle='--', alpha=0.4)
                            ax.plot(x_fit_lin, y_low_lin, color=fitcolour, linestyle='--', alpha=0.4)
                    if "exp" in keywords['fit']:
                        N_guess, s_guess, b_guess = np.max(y_list), -np.log(np.abs(y_list[0] - y_list[1])) / np.abs(
                            y_list[0] - y_list[1]) * 10, y_list[-1]
                        x_fit_exp, y_fit_exp, y_low_exp, y_high_exp, fitpar, uncpar, r2_exp, \
                        Chi2_exp, Ndof_exp, ProbChi2_exp, conv_exp = fit_exp(x_list, y_list, np.zeros_like(y_list) + keywords['uncertainty'], N_guess, s_guess, b_guess)
                        ax.plot(x_fit_exp, y_fit_exp, c=fitcolour, label="Exponential fit",
                                linewidth=cosmetics['linewidth'], zorder=4)
                        if "split" in keywords["fit_plot"]:
                            plt.fill_between(x_fit_exp, y_high_exp, y_low_exp, color=fitcolour, alpha=0.1)
                            ax.plot(x_fit_exp, y_high_exp, color=fitcolour, linestyle='--', alpha=0.4)
                            ax.plot(x_fit_exp, y_low_exp, color=fitcolour, linestyle='--', alpha=0.4)

                label_fig(ax, cosmetics['xaxis'], yax, cosmetics['title'], cosmetics['axisfontsize'],
                          cosmetics['titlefontsize'], cosmetics['legendfontsize'], cosmetics['tickmarksize'],
                          loc=legloc, ncol=int(cosmetics['legcolumns']), legend=show_legend, legbox=legbox,
                          tight=tightbool)
                ax.xaxis.set_ticks_position('both')
                ax.yaxis.set_ticks_position('both')
                ax.tick_params(direction="inout")
                ax.xaxis.set_major_locator(plt.MultipleLocator(axstep(np.min(x_list), np.max(x_list))))
                ax.yaxis.set_major_locator(plt.MultipleLocator(axstep(np.min(y_list), np.max(y_list))))
                ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5 * axstep(np.min(x_list), np.max(x_list))))
                ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5 * axstep(np.min(y_list), np.max(y_list))))

        if keywords["subplot"]:
            print_string += "Subplots".center(len(sep2))
            print_string += f"\n{sep2}".ljust(len(sep2))
            if cosmetics["subplotframe"] == "line":
                keywords["subplot"] = "loose"
                ndat = shp[1] - 1
                print_string += f"Dimensions of the line subplots is 1 row of {ndat} columns.\n\n"
                fig = plt.figure(figsize=(20, ndat+1))
                fig.subplots_adjust(wspace=0.4)
                if gridbool:
                    plt.grid(axis=cosmetics['gridlines'])
                for col, yax in zip(range(shp[1] - 1), yaxes):
                    ax = fig.add_subplot(1, ndat, col+1)
                    ax.set_aspect(1.0)
                    ax.xaxis.set_ticks_position('both')
                    ax.yaxis.set_ticks_position('both')
                    ax.tick_params(direction="inout")
                    y_list = data_table_2[:, col + 1]
                    datacolour = cosmetics['datacolour']
                    ax.scatter(x_list, y_list, c=datacolour, label=cosmetics["legend"], s=cosmetics['pointsize'],
                               zorder=3)
                    if "fit" in keywords.keys():
                        fitcolour = cosmetics['fitcolour']
                        if "linear" in keywords['fit']:
                            a_guess = (y_list[-1] - y_list[0]) / (x_list[-1] - x_list[0])
                            b_guess = y_list[0] - a_guess * x_list[0]
                            (x_fit_lin, y_fit_lin, y_low_lin, y_high_lin, a_fit, a_unc, b_fit, b_unc, r2_lin,
                             Chi2_lin, Ndof_lin, ProbChi2_lin, conv_lin) = fit_linear(x_list, y_list, np.zeros_like(y_list) + keywords['uncertainty'], a_guess, b_guess)
                            ax.plot(x_fit_lin, y_fit_lin, c=fitcolour, label="Linear fit", linewidth=cosmetics['linewidth'],
                                    zorder=4)
                            if print_fit:
                                ax.text(print_fit[0], print_fit[1], f"y = {a_fit:.3f} x + {b_fit:.3f}\n\tR$^2$ = {r2_lin:.4f}",
                                        transform=ax.transAxes, fontsize=cosmetics['textfontsize'], color=fitcolour,
                                        va='top', ha='left')
                            if "split" in keywords["fit_plot"]:
                                plt.fill_between(x_fit_lin, y_high_lin, y_low_lin, color=fitcolour, alpha=0.1)
                                ax.plot(x_fit_lin, y_high_lin, color=fitcolour, linestyle='--', alpha=0.4)
                                ax.plot(x_fit_lin, y_low_lin, color=fitcolour, linestyle='--', alpha=0.4)
                        if "exp" in keywords['fit']:
                            N_guess, s_guess, b_guess = np.max(y_list), -np.log(np.abs(y_list[0] - y_list[1])) / np.abs(
                                y_list[0] - y_list[1]) * 10, y_list[-1]
                            x_fit_exp, y_fit_exp, y_low_exp, y_high_exp, fitpar, uncpar, r2_exp, \
                            Chi2_exp, Ndof_exp, ProbChi2_exp, conv_exp = fit_exp(x_list, y_list, np.zeros_like(y_list) + keywords['uncertainty'], N_guess, s_guess,
                                                                                 b_guess)
                            ax.plot(x_fit_exp, y_fit_exp, c=fitcolour, label="Exponential fit", linewidth=cosmetics['linewidth'],
                                    zorder=4)
                            if "split" in keywords["fit_plot"]:
                                plt.fill_between(x_fit_exp, y_high_exp, y_low_exp, color=fitcolour, alpha=0.1)
                                ax.plot(x_fit_exp, y_high_exp, color=fitcolour, linestyle='--', alpha=0.4)
                                ax.plot(x_fit_exp, y_low_exp, color=fitcolour, linestyle='--', alpha=0.4)
                    if col == 0:
                        label_fig(ax, cosmetics['xaxis'], yax, "", cosmetics['axisfontsize'],
                                  cosmetics['titlefontsize'], cosmetics['legendfontsize'], cosmetics['tickmarksize'],
                                  legend=show_legend, loc=(0, 1.1), ncol=2, tight=tightbool, legbox=legbox)
                    else:
                        label_fig(ax, cosmetics['xaxis'], yax, "", cosmetics['axisfontsize'],
                                  cosmetics['titlefontsize'], cosmetics['legendfontsize'], cosmetics['tickmarksize'],
                                  legend=False, tight=tightbool, legbox=legbox)
                    ax.xaxis.set_major_locator(plt.MultipleLocator(2*axstep(np.min(x_list), np.max(x_list))))
                    ax.yaxis.set_major_locator(plt.MultipleLocator(2*axstep(np.min(x_list), np.max(x_list))))
                    ax.xaxis.set_minor_locator(plt.MultipleLocator(axstep(np.min(x_list), np.max(x_list))))
                    ax.yaxis.set_minor_locator(plt.MultipleLocator(axstep(np.min(x_list), np.max(x_list))))
                plt.suptitle(cosmetics['title'], fontsize=cosmetics['titlefontsize'], y=0.91)
            elif cosmetics["subplotframe"] == "grid":
                dims = subdim(shp[1] - 1)
                print_string += f"Dimensions of the grid subplot is {dims[0]} rows of {dims[1]} columns\n\n"
                fig, axs = plt.subplots(nrows = dims[0], ncols = dims[1], figsize = dims[0]*dims[1]*figsize)#, sharey=True)#, sharex=True)
                if keywords["subplot"] == "sticky":
                    fig.subplots_adjust(hspace=0, wspace=0)
                else:
                    fig.subplots_adjust(hspace=0.3, wspace=0.1)
                for ax, col, yax in zip(axs.flatten(), range(shp[1] - 1), yaxes):
                    if col % 2 == 1:
                        ax.yaxis.tick_right()
                        ax.yaxis.set_label_position("right")
                    ax.xaxis.set_ticks_position('both')
                    ax.yaxis.set_ticks_position('both')
                    ax.tick_params(direction="inout")
                    if gridbool:
                        plt.grid(axis=cosmetics['gridlines'])
                    y_list = data_table_2[:, col + 1]
                    datacolour = cosmetics['datacolour']
                    ax.scatter(x_list, y_list, c=datacolour, label=cosmetics["legend"], s=cosmetics['pointsize'], zorder=3)
                    if "fit" in keywords.keys():
                        fitcolour = cosmetics['fitcolour']
                        if "linear" in keywords['fit']:
                            a_guess = (y_list[-1] - y_list[0]) / (x_list[-1] - x_list[0])
                            b_guess = y_list[0] - a_guess * x_list[0]
                            (x_fit_lin, y_fit_lin, y_low_lin, y_high_lin, a_fit, a_unc, b_fit, b_unc, r2_lin,
                             Chi2_lin, Ndof_lin, ProbChi2_lin, conv_lin) = fit_linear(x_list, y_list, np.zeros_like(y_list) + keywords['uncertainty'], a_guess, b_guess)
                            ax.plot(x_fit_lin, y_fit_lin, c=fitcolour, label="Linear fit", linewidth=cosmetics['linewidth'],
                                    zorder=4)
                            if print_fit:
                                ax.text(print_fit[0], print_fit[1],
                                        f"y = {a_fit:.3f} x + {b_fit:.3f}\n\tR$^2$ = {r2_lin:.4f}",
                                        transform=ax.transAxes, fontsize=cosmetics['textfontsize'], color=fitcolour,
                                        va='center', ha='left')
                            if "split" in keywords["fit_plot"]:
                                plt.fill_between(x_fit_lin, y_high_lin, y_low_lin, color=fitcolour, alpha=0.1)
                                ax.plot(x_fit_lin, y_high_lin, color=fitcolour, linestyle='--', alpha=0.4)
                                ax.plot(x_fit_lin, y_low_lin, color=fitcolour, linestyle='--', alpha=0.4)
                        if "exp" in keywords['fit']:
                            N_guess, s_guess, b_guess = np.max(y_list), -np.log(np.abs(y_list[0] - y_list[1])) / np.abs(
                                y_list[0] - y_list[1]) * 10, y_list[-1]
                            x_fit_exp, y_fit_exp, y_low_exp, y_high_exp, fitpar, uncpar, r2_exp, \
                            Chi2_exp, Ndof_exp, ProbChi2_exp, conv_exp = fit_exp(x_list, y_list, np.zeros_like(y_list) + keywords['uncertainty'], N_guess, s_guess,
                                                                                 b_guess)
                            ax.plot(x_fit_exp, y_fit_exp, c=fitcolour, label="Exponential fit", linewidth=cosmetics['linewidth'],
                                    zorder=4)
                            if "split" in keywords["fit_plot"]:
                                plt.fill_between(x_fit_exp, y_high_exp, y_low_exp, color=fitcolour, alpha=0.1)
                                ax.plot(x_fit_exp, y_high_exp, color=fitcolour, linestyle='--', alpha=0.4)
                                ax.plot(x_fit_exp, y_low_exp, color=fitcolour, linestyle='--', alpha=0.4)
                    label_fig(ax, cosmetics['xaxis'], yax, "", cosmetics['axisfontsize'],
                              cosmetics['titlefontsize'], cosmetics['legendfontsize'], cosmetics['tickmarksize'],
                              loc=legloc, ncol=int(cosmetics['legcolumns']), tight=tightbool, legend=show_legend,
                              legbox=legbox)
                    ax.xaxis.set_major_locator(plt.MultipleLocator(axstep(np.min(x_list), np.max(x_list))))
                    ax.yaxis.set_major_locator(plt.MultipleLocator(axstep(np.min(x_list), np.max(x_list))))
                    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5*axstep(np.min(x_list), np.max(x_list))))
                    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5*axstep(np.min(x_list), np.max(x_list))))
                plt.suptitle(cosmetics['title'], fontsize=cosmetics['titlefontsize'])

    if "dev" in keywords["plot"]:
        print_string += "Deviations".center(len(sep2))
        print_string += f"\n{sep2}".ljust(len(sep2))
        """Start by opening the main deviation bar figure and re-structure the given data such that the reference
        column is at index 0 (for convenience)"""
        figbar, axbar = plt.subplots(figsize=(12, 6))
        if gridbool:
            plt.grid(axis=cosmetics['gridlines'])
        shp = np.shape(data_table)[1]
        data_table_2 = np.zeros_like(data_table)
        data_table_2[:, 0] = data_table[:, ref]
        data_table_2[:, 1:] = np.delete(data_table, ref, axis=1)

        """Input deviation bar colors written in such a way, that both strings and colour tuples
        are recognised properly"""
        barcols = {}
        models = []
        for i in cosmetics["devcolour"].split(';'):
            c = i.split(":")
            try:
                barcols[c[0]] = ast.literal_eval(c[1])
            except:
                barcols[c[0]] = c[1]
        for j in cosmetics["model"].split(";"):
            models.append(j)

        """Start working on generating the actual plot, starting with MAE, ME, and STD."""
        legends = []
        mae_bool = False
        me_bool = False
        std_bool = False
        maxerr_bool = False
        legdir = {}
        mae = []
        me = []
        std = []
        maxerr = []
        if "mae" in keywords["deviation"]:
            mae_bool = True
            legends.append("Mean Abs Dev")
            legdir["mae"] = "Mean Abs Dev"
        if "me" in keywords["deviation"]:
            me_bool = True
            legends.append("Mean Dev")
            legdir["me"] = "Mean Dev"
        if "std" in keywords["deviation"]:
            std_bool = True
            legends.append("Std Dev")
            legdir["std"] = "Std Dev"
        if "maxerr" in keywords["deviation"]:
            maxerr_bool = True
            legends.append("Max Dev")
            legdir["maxerr"] = "Max Dev"

        """Calculate the array of errors, the MAEs, MEs, STDs, and MaxErr"""
        err = []
        limsbar = []
        for i in range(shp):
            if not i == 0:
                er = []
                if mae_bool:
                    val = calc_mae(data_table_2[:, (0, i)], 0, 1, frac=frac)[0]
                    mae.append(val)
                    limsbar.append(val)
                if me_bool:
                    val = calc_me(data_table_2[:, (0, i)], 0, 1, frac=frac)[0]
                    me.append(val)
                    limsbar.append(val)
                if std_bool:
                    val = calc_std(data_table_2[:, (0, i)], frac=frac)
                    std.append(val)
                    limsbar.append(val)
                er = calc_err(data_table_2[:, (0, i)], 0, 1, frac)
                err.append(er)
                if maxerr_bool:
                    val = np.max(np.abs(er))
                    maxerr.append(val)
                    limsbar.append(val)
        err = np.array(err)
        limsbar = [np.min(limsbar), np.max(limsbar)]

        """Set input parameters for things like bar width, rotation of labels and such"""
        # width = 0.1
        width = 1/(len(legdir) + cosmetics["nbarsep"])
        nbars = len(barcols)
        rot = cosmetics['rot']
        n_data = np.max((len(mae), len(me), len([std])))
        ind = np.arange(n_data)

        """Draw the baseline for the bars (easily identifiable y=0 line"""
        axbar.plot((-1, len(me) + 1), (0, 0), alpha=0.4, color='black', linewidth=2, zorder=1)
        if "scatter" in keywords["deviation"] or "scatline" in keywords["deviation"]:
            """If scatter errors are requested in deviation keyword, also draw the y=0 line for this data set"""
            axdotline = axbar.twinx()
            axbar.set_zorder(2)             # Puts axbar in front of axdotline
            axbar.patch.set_visible(False)  # Makes the axbar background transparent
            axdotline.tick_params(axis='y', right=False, labelright=False)  # Remove tickmarks and -labels on this axis
            axdotline.plot((-1, len(me) + 1), (0, 0), alpha=0.4, color='black', linewidth=2, zorder=1)

            if "scatline" in keywords["deviation"]:
                width = 1/(len(legdir) + cosmetics["nbarsep"] + 1)
                nbars += 1

            try:
                ybar, ydot = cosmetics['yaxis'].split(";")[:2]
            except:
                ybar, ydot = cosmetics['yaxis'].split(";")[0], ""

        else:
            ybar = cosmetics['yaxis'].split(";")[0]

        """Plot bars using the provided bar colours (or default if none are specified)"""
        legs = ()
        barloc = barlocs(nbars, width=width)
        counter = 0
        if mae_bool:
            legmae = axbar.bar(ind + barloc[counter], mae, width, label=legdir["mae"], color=barcols['mae'], edgecolor='k',
                               hatch=hatches[0], zorder=2)
            legs = legs + (legmae,)
            counter += 1
        if me_bool:
            legme = axbar.bar(ind + barloc[counter], me, width, label=legdir["me"], color=barcols['me'], edgecolor='k',
                              hatch=hatches[1], zorder=2)
            legs = legs + (legme,)
            counter += 1
        if std_bool:
            legstd = axbar.bar(ind + barloc[counter], std, width, label=legdir["std"], color=barcols['std'], edgecolor='k',
                               hatch=hatches[2], zorder=2)
            legs = legs + (legstd,)
            counter += 1
        if maxerr_bool:
            legmaxerr = axbar.bar(ind + barloc[counter], maxerr, width, label=legdir["maxerr"], color=barcols['maxerr'],
                                  edgecolor='k', hatch=hatches[3], zorder=2)
            legs = legs + (legmaxerr,)
            counter += 1
        """Place the tick labels on the x-axis and lock the x-axis to a symmetrical view"""
        plt.xticks(ind, models, fontsize=cosmetics["axisfontsize"])#, rotation=100)
        plt.setp(axbar.xaxis.get_majorticklabels(), rotation=rot)
        for label in axbar.get_xticklabels():
            label.set_horizontalalignment(cosmetics["tickalignment"])
        axbar.set_xlabel(cosmetics['xaxis'], fontsize=cosmetics['axisfontsize'])
        axbar.set_ylabel(ybar, fontsize=cosmetics['axisfontsize'])
        axbar.set_title(cosmetics['title'], fontsize=cosmetics['titlefontsize'])
        axbar.xaxis.set_tick_params(labelsize=cosmetics['tickmarksize'])
        axbar.yaxis.set_tick_params(labelsize=cosmetics['tickmarksize'])
        
        blank=" "
        print_string += f"{blank:11}\t{models[0]:>{spac}}"
        # print_string += f"\t\t\t{models[0]:20}"
        for imodel in np.arange(len(models)-1)+1:
            print_string += f"\t{models[imodel]:>{spac}}"
        if mae_bool:
            name="MAD"
            print_string += f"\n{name:11}\t{mae[0]:{spac}.{dec}f}"
            for imodel in np.arange(len(models)-1)+1:
                print_string += f"\t{mae[imodel]:{spac}.{dec}f}"
        if me_bool:
            name="MD"
            print_string += f"\n{name:11}\t{me[0]:{spac}.{dec}f}"
            for imodel in np.arange(len(models)-1)+1:
                print_string += f"\t{me[imodel]:{spac}.{dec}f}"
        if std_bool:
            name="STD"
            print_string += f"\n{name:11}\t{std[0]:{spac}.{dec}f}"
            for imodel in np.arange(len(models)-1)+1:
                print_string += f"\t{std[imodel]:{spac}.{dec}f}"
        if maxerr_bool:
            name="MaxDev"
            print_string += f"\n{name:11}\t{maxerr[0]:{spac}.{dec}f}"
            for imodel in np.arange(len(models)-1)+1:
                print_string += f"\t{maxerr[imodel]:{spac}.{dec}f}"
        print_string += "\n\n"
        
        for header, fracbool in zip(("relative", "fraction"), (True, False)):
            print_string += f"Individual deviations, {header}".center(len(sep2))
            print_string += f"\n{sep2}".ljust(len(sep2))
            print_string += f"{models[0]:>{spac}}"
            for imodel in np.arange(len(models) - 1) + 1:
                print_string += f"\t{models[imodel]:>{spac}}"
            for dev in calc_deviations(data_table_2, frac=fracbool):
                print_string += f"\n{dev[0]:{spac}.{dec}f}"
                for i, devv in enumerate(dev):
                    if i:
                        print_string += f"\t{devv:{spac}.{dec}f}"
            print_string += "\n\n"

        scat_bool = False
        if "scatter" in keywords["deviation"] or "scatline" in keywords["deviation"]:
            scat_bool = True
            axdot = axbar.twinx()
            print_string += "Deviation scatter plot".center(len(sep2))
            print_string += f"\n{sep2}".ljust(len(sep2))
            """If scatter errors have been requested, generate new axis to plot these"""
            axdot.set_zorder(axbar.get_zorder() + 1)    # Ensure that this new scatter axis is placed in front of axbar
            if "scatter" in keywords["deviation"]:
                for i in np.arange(shp - 1):
                    """Generate an x-list for the scatter points to be spread out on"""
                    x_scat = np.linspace(ind[i] - width, ind[i] + width, np.shape(err)[1])
                    legdot = axdot.scatter(x_scat, err[i, :], color=cosmetics['devpointcol'], edgecolor='k', s=cosmetics['pointsize'],
                                           label="Deviations")
            if "scatline" in keywords["deviation"]:
                for i in np.arange(shp - 1):
                    """Generate an x-list for the scatter points to be spread out on"""
                    x_scat = np.ones_like(err[i, :]) * i + barloc[counter]
                    legdot = axdot.scatter(x_scat, err[i, :], color=cosmetics['devpointcol'], edgecolor='k', s=cosmetics['pointsize'],
                                           label="Deviations")
            legs = legs + (legdot,)

            """Fix the tickmarksize and scale the axdotline y-axis to match axdot (quick and dirty way to 
            draw the line behind the bars, but the dots in front of the bars - couldn't find a better way)"""
            limserr = [np.min(err), np.max(err)]
            if cosmetics["yrange"]:
                try:
                    yrange = ast.literal_eval(cosmetics["yrange"])
                except:
                    yrange = ((0, 1), (0, 1))
                    print_string += f"\n\n The input yrange of {cosmetics['yrange']} was not interpreted properly." \
                                    f"\n The range {yrange} has been employed.\n\n"
                (axbarlims, axdotlims) = yrange
                # print(axbarlims, axdotlims)
            else:
                axbarlims, axdotlims = lha(limsbar, limserr)
            if ydot:
                axdot.yaxis.set_tick_params(labelsize=cosmetics['tickmarksize'])
                axdot.set(ylim=axdotlims)
                axdot.set_ylabel(ylabel=ydot, fontsize=cosmetics['axisfontsize'], rotation=-90, labelpad=20)
            else:
                axdot.tick_params(axis='y', right=False, labelright=False)  # Remove tickmarks and -labels on this axis
            axdotline.set(ylim=axdot.get_ylim())

        else:
            try:
                yrange = ast.literal_eval(cosmetics["yrange"])
                if len(np.shape(yrange)) > 1:
                    yrange = yrange[0]
                axbarlims = yrange
            except:
                axbarlims = lha(limsbar, limsbar)[0]
        axbar.set(xlim=(-0.5, len(me) - 0.5), ylim=axbarlims)

#        mlom = max([len(x) for x in models])
#        plt.gcf().subplots_adjust(bottom=0.03*mlom*cosmetics["tickmarksize"]/cosmetic_defaults["tickmarksize"])
        if tightbool:
            plt.tight_layout()

        """Add all legends to the figure legend"""
        leglabs = [l.get_label() for l in legs]
        if show_legend:
            if scat_bool:
                axdot.legend(legs, leglabs, loc=legloc, fontsize=cosmetics['legendfontsize'], frameon=legbox,
                             ncol=int(cosmetics['legcolumns']))#, zorder=axdot.get_zorder() + 1)
                print_string += "Deviation scatter plot has been constructed and overlayed the deviation bars."
            else:
                axbar.legend(legs, leglabs, loc=legloc, fontsize=cosmetics['legendfontsize'], frameon=legbox,
                             ncol=int(cosmetics['legcolumns']))  # , zorder=axdot.get_zorder() + 1)

    if "bar" in keywords["plot"]:
        print_string += "Pre-calculated bar plots".center(len(sep2))
        print_string += f"\n{sep2}".ljust(len(sep2))
        print_string += "This bar plot is made directly as a bar plot of the input data.\n" \
                        "The user is therefore responsible for the quality of the calculations.\n\n"
        barcols = {}
        barhatch = {}
        models = []
        legends = []
        for j in cosmetics['model'].split(";"):
            models.append(j)
        for i, icol, leg in zip(range(len(cosmetics["devcolour"].split(';'))), cosmetics["devcolour"].split(';'),
                                cosmetics["legend"].split(";")):
            c = icol.split(":")
            legends.append(leg)
            barhatch[c[0]] = hatches[i]
            try:
                barcols[c[0]] = ast.literal_eval(c[1])
            except:
                barcols[c[0]] = c[1]
        width = 1/(len(data_table[0]) + cosmetics["nbarsep"])
        rot = cosmetics['rot']
        n_data = len(data_table)
        ind = np.arange(n_data)
        nbars = len(barcols)
        barloc = barlocs(nbars, width=width)
        fig, ax = plt.subplots(figsize=(12, 6))
        if gridbool:
            plt.grid(axis=cosmetics['gridlines'])
        for column, i, colorkey, leg in zip(np.transpose(data_table),
                                            np.arange(-len(data_table[0]) // 2 + 1, len(data_table[0]) // 2 + 1, 1),
                                            barcols.keys(), legends):
            # ax.bar(ind + i * width, column, width, color=barcols[colorkey], label=leg, zorder=2,
            #        hatch=barhatch[colorkey])
            ax.bar(ind + barloc[i], column, width, color=barcols[colorkey], label=leg, zorder=2,
                   hatch=barhatch[colorkey])
        ax.plot((-1, len(ind) + 1), (0, 0), alpha=0.4, color='black', linewidth=2, zorder=1)
        limsbar = (np.min(data_table), np.max(data_table))
        if cosmetics["yrange"]:
            try:
                yrange = ast.literal_eval(cosmetics["yrange"])
            except:
                yrange = ((0, 1), (0, 1))
                print_string += f"\n\n The input yrange of {cosmetics['yrange']} was not interpreted properly." \
                                f"\n The range {yrange} has been employed.\n\n"
        else:
            yrange = lhal(limsbar)
        ax.set(xlim=(-0.5, len(ind) - 0.5), ylim=yrange)
        plt.xticks(ind, models, fontsize=cosmetics["axisfontsize"], rotation=rot, ha=cosmetics["tickalignment"])
        ax.yaxis.set_major_locator(plt.MultipleLocator(axstep(yrange[0], yrange[1])))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5 * axstep(yrange[0], yrange[1])))

        label_fig(ax, cosmetics['xaxis'], cosmetics['yaxis'], cosmetics["title"], cosmetics['axisfontsize'],
                  cosmetics['titlefontsize'], cosmetics['legendfontsize'], cosmetics['tickmarksize'],
                  loc=legloc, ncol=int(cosmetics['legcolumns']), tight=tightbool, legend=show_legend, legbox=legbox)


for name in plot_name:
    name, ext = name.split(sep='.')
    plt.savefig(f"{name}.{ext}", dpi=cosmetics["dpi"])

print_string += sep
print_string += "Printing content of input file:\n\n"

with open(filename, 'r') as inFile:
    for line in inFile:
        print_string += f"{line}"

print_string += sep

print_string += "\n\n\nExiting PlotPro. Have a nice day :)"

with open(output_name, 'w') as outFile:
    outFile.write(print_string)

plt.show()

