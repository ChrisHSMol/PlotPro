#!/opt/anaconda3.7/bin/python3.7
import numpy as np, matplotlib.pyplot as plt, os, sys
import regex as re, ast
import argparse
from setup_functions import *
from subroutines.parse_input import *
from subroutines.scatter import *
from subroutines.bar import *
from subroutines.dev import *

if len(sys.argv) == 1:
    print(f"\n\n\tERROR: This script must be called with an additional argument, ie: "
          f"\n\n\t\t\t plotpro input.txt (--suppress)\n\n")
    sys.exit()

filenames = sys.argv[1:]

# Set up the ArgumentParser to interpret the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("file", help="Name(s) of the file(s) to plot. If more than one input, all files will be processed.", nargs='+')
parser.add_argument("-s", "--suppress", help="Suppress the display of figures. Default: False")
args = parser.parse_args()
boolsuppress = False    # Default: Don't suppress the showing of figures.
if args.suppress:
    if args.suppress.lower() in ("y", "j", "yes", "ja", "on", "true") or args.suppress == True:
        boolsuppress = True

    filenames = filenames[:-2]  # Remove the "--suppress" and "arg" from the list of input arguments

(save_list, keys_list, keys, keys_default, cosmetics_list, cosmstr, cosmval, cosmcol, cosmetic_defaults,
 hatch_list) = definitions()

# print_string = "\n\nRunning PlotPro"
print_string = initiate_output()
sep2 = "-"*35 + "="*50 + "-"*35 + "\n"*2
sep = "\n"*2 + sep2 + "\n"*2

for filename in filenames:
    if os.path.isfile(filename):
        print("Working on file: "+filename)
    else:
        print(filename+" is not a filename.")
        continue

    output_name = f"{os.path.splitext(filename)[0]}.out"

    print_string += f"\n\n{'Input file':30} {filename:25}\n{'Current working directory':30} {os.getcwd()}{sep}"

    # PPPPPPP     AAAAAA    RRRRRRR      SSSSSSS    EEEEEEEE
    # P      P   A      A   R      R    S           E
    # P      P  A        A  R      R    S           E
    # PPPPPPP   AAAAAAAAAA  RRRRRRR      SSSSS      EEEEE
    # P         A        A  R R                S    E
    # P         A        A  R   R               S   E
    # P         A        A  R    R              S   E
    # P         A        A  R     R            S    E
    # P         A        A  R      R    SSSSSSS     EEEEEEEE

    data_table, keywords, cosmetics, plot_name, print_string = read_input(
        filename,
        save_list,
        keys_list,
        keys,
        cosmetics_list,
        print_string
    )

    if "devcolour" not in cosmetics.keys():
        if "bar" in keywords["plot"]:
            if "legend" not in cosmetics.keys():
                cosmetics["legend"] = ""

            if len(cosmetics["legend"].split(";")) <= 4:
                cosmetics["devcolour"] = update_default_bar_colours(legends=cosmetics["legend"].split(";"))

        elif "dev" in keywords["plot"]:
            if "legend" not in cosmetics.keys():
                ref = ""
                for i in keywords["deviation"]:
                    if i not in ("scatter", "ref", "type"):
                        ref += i+";"

                cosmetics["legend"] = ref[:-1]  # Remove the trailing ";"

            if sum([i not in ("scatter", "ref", "type") for i in keywords["deviation"]]) <= 4:
                cosmetics["devcolour"] = update_default_bar_colours(legends=cosmetics["legend"].split(";"))

    """
    if "bar" in keywords["plot"]:
        if "devcolour" not in cosmetics.keys():
            if "legend" not in cosmetics.keys():
                cosmetics["legend"] = ""
                cosmetics["devcolour"] = update_default_bar_colours(legends=cosmetics["legend"].split(";"))
    
            elif len(cosmetics["legend"].split(";")) <= 4:
                cosmetics["devcolour"] = update_default_bar_colours(legends=cosmetics["legend"].split(";"))
    
    elif "dev" in keywords["plot"]:
        if "devcolour" not in cosmetics.keys():
            if "legend" not in cosmetics.keys():
                ref = ""
                for i in keywords["deviation"]:
                    if i not in ("scatter", "ref"):
                        ref += i+";"
    
                cosmetics["legend"] = ref[:-1]
    
            if sum([i != "scatter" and i != "ref" for i in keywords["deviation"]]) <= 4:
                cosmetics["devcolour"] = update_default_bar_colours(legends=cosmetics["legend"].split(";"))
    """




    # Input default values

    cosmetics, keywords, gridbool, print_fit, figsize, legloc, print_string = add_defaults(
        cosmetics,
        cosmetic_defaults,
        cosmetics_list,
        keywords,
        keys_default,
        keys_list,
        hatch_list,
        print_string
    )

    ref = keywords["ref"]       # put the reference column in a variable for easy use in arrays

    # Sanitate data for bad reference values - only if fractional deviations are to be made

    if keywords["frac"]:
        data_backup = np.array(data_table)
        data_table, print_string = sanitate_data(
            data_table,
            keywords["ref"],
            output_name,
            print_string
        )
    else:
        data_table = np.array(data_table)

    # Print all parameters to output (that way, it is clear exactly which keywords exist, and which values have been used)
    print_string += "Used parameters".center(len(sep2))
    print_string += f"\n{sep2}".ljust(len(sep2))
    print_string += "routecard:\n"
    # for key in sorted(list(keywords.keys())):
    for key in sorted(keys_list):
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
    # for key in cosmetics:
    for key in sorted(cosmetics_list):
        if (key == "devpattern") and cosmetics[key] == "on":
            print_string += "\t{0:<20} {1}\t\t\"".format(key, cosmetics[key])
            for i, hat in enumerate(cosmetics[key]):
                if not i == len(cosmetics[key]) - 1:
                    print_string += "{0};".format(hat)
                else:
                    print_string += "{0}\"\n".format(hat)
        else:
            print_string += "\t{0:<20} {1}\n".format(key, cosmetics[key])
    print_string += sep

    data_table = rearrange_table(data_table, ref)

    # PPPPPPP   L           OOOO    TTTTTTTT
    # P      P  L         OO    OO     TT
    # P      P  L         O      O     TT
    # PPPPPPP   L         O      O     TT
    # P         L         O      O     TT
    # P         L         O      O     TT
    # P         L         O      O     TT
    # P         L         OO    OO     TT
    # P         LLLLLLLL    OOOO       TT

    if "plot" in keywords.keys():
        spac, dec = cosmetics["outdec"] + 5, cosmetics["outdec"]
        format_dict = {"spac": spac, "dec": dec, "sep2": sep2, "figsize": figsize, "legloc": legloc, "print_fit": print_fit}
        bool_dict = {"gridbool": gridbool, "diagonal": cosmetics["diagonal"], "show_legend": cosmetics["show_legend"]}

        if "scatter" in keywords["plot"]:
            print_string += "Scatter plot".center(len(format_dict["sep2"]))
            print_string += f"\n{format_dict['sep2']}".ljust(len(format_dict["sep2"]))
            fig, ax, print_string = plot_scatter(
                data_table,
                keywords,
                cosmetics,
                print_string,
                format_dict,
                bool_dict,
                plot_name
            )
            fig.patch.set_facecolor(cosmetics["background"])
            ax.set_facecolor(cosmetics["background"])

        if "dev" in keywords["plot"]:
            print_string += "Deviations".center(len(sep2))
            print_string += f"\n{sep2}".ljust(len(sep2))
            fig, ax, print_string = plot_dev(
                data_table,
                keywords,
                cosmetics,
                print_string,
                format_dict,
                bool_dict,
                plot_name,
                output_name
            )
            fig.patch.set_facecolor(cosmetics["background"])
            for a in list(ax):
                a.set_facecolor(cosmetics["background"])

        if "bar" in keywords["plot"]:
            print_string += "Pre-calculated bar plots".center(len(sep2))
            print_string += f"\n{sep2}".ljust(len(sep2))
            print_string += "This bar plot is made directly as a bar plot of the input data.\n" \
                            "The user is therefore responsible for the quality of the calculations.\n\n"
            fig, ax, print_string = plot_bars(
                data_table,
                keywords,
                cosmetics,
                print_string,
                format_dict,
                bool_dict,
                plot_name
            )
            fig.patch.set_facecolor(cosmetics["background"])
            ax.set_facecolor(cosmetics["background"])

    # for name in plot_name:
        # name, ext = name.split(sep='.')
        # plt.savefig(f"{name}.{ext}", dpi=cosmetics["dpi"])
    #     plt.savefig(name, dpi=cosmetics["dpi"])

    print_string += sep
    print_string += "Printing content of input file:\n\n"

    with open(filename, 'r') as inFile:
        for line in inFile:
            print_string += f"{line}"

    print_string += sep

    print_string += "\n\n\nExiting PlotPro. Have a nice day :)"

    with open(output_name, 'w') as outFile:
        outFile.write(print_string)

if boolsuppress:
    print("Output figures have been suppressed by the '--suppress' argument")
else:
    plt.show()
