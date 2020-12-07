    #!/opt/anaconda3.7/bin/python3.7
import numpy as np
import regex as re
import sys
import ast


def read_input(filename, save_list, keys_list, keys, cosmetics_list, print_string):
    """
    Read the input file given to PlotPro and fill in the containers such as print_string, keywords, and cosmetics,
    which are used constantly throughout the rest of the program.
    :param filename:            The name (and relative path) of the input file.
    :param save_list:           List of supported formats to save figures as (Matplotlib formats).
    :param keys_list:           List of accepted keywords used in plotting.
    :param keys:                Dictionary containing further specifications of the keys. These two parameters could
                                definitely be combined.
    :param cosmetics_list:      List of accepted keywords regarding the cosmetic aspects of the plots.
    :param print_string:        The output string, which is written into the output file.
                                I don't know how to log properly.
    :return:                    Returns the array of data, the updated print_string, the keywords and cosmetics read
                                from the input file.
                                Also returns a list of the desired names for the output figure file.
    """

    sep2 = "-" * 35 + "=" * 50 + "-" * 35 + "\n" * 2
    sep = "\n" * 2 + sep2 + "\n" * 2
    keywords = {}
    cosmetics = {}
    data_table = []
    for key in keys_list:
        keywords[key] = []

    with open(filename, 'r') as inFile:
        plot_name = []
        keywords["frac"] = False
        # Read each line in the file, line by line
        for line in inFile:

            if line.startswith("%"):  # Look for savecard specifications
                plot_name = read_savecard(line, save_list, plot_name)

            elif line.startswith("#"):  # Look for routecard specifications
                keywords = read_routecard(line, keys_list, keys, keywords)

            elif line.startswith("$"):  # Look for titlecard specifications
                cosmetics = read_titlecard(line, cosmetics_list, cosmetics)

            elif (line[0].isdigit()) or (line[0] == '-'):  # Look for data given by either a digit or a minus sign
                line = line.split()
                data_line = []
                for number in line:
                    data_line.append(float(number))
                data_table.append(data_line)

    # ---------------------------------------------------------------------------------------------------

    if len(plot_name) > 0:
        print_string += f"Figure will be saved as {plot_name[0]}"
        for iname in np.arange(len(plot_name) - 1) + 1:
            print_string += f", {plot_name[iname]}"
    else:
        print_string += f"Figure will not be saved by ths program."
    print_string += f"{sep}"

    # ---------------------------------------------------------------------------------------------------

    print_string += "Input parameters".center(len(sep2))
    print_string += f"\n{sep2}".ljust(len(sep2))
    print_string += "routecard:\n"
    # for key in keywords:
    for key in sorted(list(keywords.keys())):
        if type(keywords[key]) == type(keys_list):
            if len(keywords[key]) == 1:
                print_string += "\t{0:<20} {1}\n".format(key, keywords[key][0])
            elif len(keywords[key]) > 1:
                print_string += "\t{0:<20} ".format(key)
                for i in np.arange(len(keywords[key])):
                    if i == len(keywords[key]) - 1:
                        print_string += f"{keywords[key][i]}\n"
                    else:
                        print_string += f"{keywords[key][i]}, "
            else:
                continue
        else:
            print_string += "\t{0:<20} {1}\n".format(key, keywords[key])
    # ---------------------------------------------------------------------------------------------------
    print_string += "\n\ntitlecard:\n"
    # for key in cosmetics:
    for key in sorted(list(cosmetics.keys())):
        print_string += "\t{0:<20} {1}\n".format(key, cosmetics[key])
    print_string += sep

    return data_table, keywords, cosmetics, plot_name, print_string


def read_savecard(line, save_list, plot_name):
    """
    Read the savecard(s) provided by the input file - currently written to generate the list of
    output figure filenames.
    :param line:        Line in the input file
    :param save_list:   List of file extensions supported by pyplot
    :param plot_name:   List of already read filenames
    :return:            Returns an updated version of the plot_name list
    """
    for save in save_list:  # Look through the supported filetypes for sacing figures
        reg = re.compile(f"({save}) ?= ?(.*?\.{save})")
        match = reg.search(line.lower())

        if match:  # If a given filetype is specified, append the specified filename to the list "plot_name"
            plot_name.append(match.group(2))

    return plot_name


def read_routecard(line, keys_list, keys, keywords):
    """
    Read the routecard provided by the input file - the routecard contains information about the kind of
    work PlotPro is going to do, i.e. what kind of plot, whether data will be fitted to e.g. a linear function.
    :param line:        Line in the input file
    :param keys_list:   List of supported routecard keys - should be the same as keys.keys()
    :param keys:        Dict of supported variations of the keys_list list of keys
    :param keywords:    Dict of all job specifications, later to be filled with default values
                        for keys not defined in the input file
    :return:            Returns an updated version of the keywords dict
    """
    for key in keys_list:  # Iterate through all supported keys

        if key in line.lower():
            # If a supported key is found in the routecard, look through supported values

            for val in keys[key]:

                if key == "deviation":  # Special subroutine for deviation key due to the supported subkeys
                    reg = re.compile(f"({key}) ?= ?\((.*?)\)")
                    match = reg.search(line.lower())

                    if match:
                        devs = match.group(2)  # .split(',')

                        for dev in keys[key]:

                            if not dev in keywords[key]:
                                # If the deviation keyword is not already in keywords, look for the deviation keyword
                                reg = re.compile(f"({dev}) ?=? ?(\d*)")
                                match = reg.search(devs.lower())

                                if match:
                                    keywords[key].append(dev)

                                    if dev == "ref":
                                        # ref = int(match.group(2)) - 1  # Set the determined reference column
                                        keywords[dev] = int(match.group(2)) - 1
                                reg2 = re.compile(f"({dev}) ?=? ?(frac|rel)")
                                match = reg2.search(devs.lower())

                                if match and dev == "type":
                                        keywords[dev] = match.group(2)
                    break

                elif (key == "ref") or (key == "uncertainty"):
                    # Similarly, if the ref keyword is specifically given, set the ref to that
                    reg = re.compile(f" ({key}) ?= ?(\d*.?\d*)")
                    match = reg.search(line.lower())

                    if match:

                        if key == "ref":
                            keywords[key] = int(match.group(2)) - 1

                        else:
                            keywords[key] = float(match.group(2)) - 1

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

    return keywords


def read_titlecard(line, cosmetics_list, cosmetics):
    """
    Read the titlecard provided by the input file. The titlecard contains information about
    the visual aspects of the plots produced by PlotPro, e.g. the fontsize, colours and patterns
    :param line:                Line in the input file
    :param cosmetics_list:      List of supported cosmetic keywords
    :param cosmetics:           Dict of all cosmetic parameters to include in the processing of the plots.
                                Will later be filled with default values for undefined parameters
    :return:                    Returns an updated version of the cosmetics dict
    """
    for cos in cosmetics_list:
        if not cos in cosmetics.keys():  # Skip keywords that are already read in a different line
            if "color" in line.lower():
                line = re.sub('color', 'colour', line)
            if (cos == "legloc") or (
                    cos == "print_fit"):  # Legloc needs special parsing syntax due to 3 different input types
                reg = re.compile(f" ({cos}) ?= ?\"?(.*)\"?\s")
                match = reg.search(line)
                if match:
                    try:
                        cosmetics[cos] = ast.literal_eval(
                            match.group(2))  # Used to interpret tuples and floats
                    except:
                        cosmetics[cos] = match.group(2)  # Interpret strings
            elif cos == "yrange" or cos == "extrapolate":
                # reg = re.compile(f" ({cos}) ?= ?\"?(\(-?\d*, ?-?\d*\))\"?\s")
                reg = re.compile(f" ({cos}) ?= ?\"?(\(\(?-?\d*, ?-?\d*\)?,? ?\(?-?\d*,? ?-?\d*\)?\))\"?\s")
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
                continue
            regstr = re.compile(f" ({cos}) ?= ?\"(.*?)\"")  # Look for both string-specific and
            matchstr = regstr.search(line)
            regval = re.compile(f" ({cos}) ?= ?([+-]?\d*\.?\d*)")  # float-specific RegExs
            matchval = regval.search(line)
            if matchstr:
                cosmetics[cos] = matchstr.group(2)
            elif matchval:
                try:
                    cosmetics[cos] = int(matchval.group(2))
                except:
                    cosmetics[cos] = float(matchval.group(2))

    return cosmetics


def sanitate_data(data_table, ref, output_name, print_string, allow_exit=True):
    """
    Remove bad data points - here defined as data points with reference values below variable 'thresh' (currently
    hard-coded), since the process of calculating the fraction involves the reference value in the denominator, thus
    the fraction will blow up if the reference value is too small.
    :param data_table:      The array of data as acquired in the 'read_input()' function.
    :param print_string:    The output string.
    :param ref:             The reference column (0-indexed, of course)
    :param output_name:     The name of the output file - this is currently auto-generated early in the program.
    :return:                The sanitated array of data as a NumPy-array as well as the updated print_string (only
                            updated if at least one point is removed).
    """
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
    # ---------------------------------------------------------------------------------------------------
    if len(data_table) == 0 and allow_exit:
        print_string += f"Data table has been scrubbed as all reference values were below the threshold of {thresh}.\n" \
                        f"Exiting program."
        with open(output_name, 'w') as outFile:
            outFile.write(print_string)
        sys.exit()

    return data_table, print_string


def add_defaults(cosmetics, cosmetic_defaults, cosmetics_list, keywords, keys_default, keys_list,
                          hatch_list, print_string):
    """
    Add the default cosmetic values to the non-specified, and update some of the specified to easier-to-read Pythonic
    values. NOTE: This can definitely be optimised in some capacity, but I found it easier to just brute-force.
    :param cosmetics:           Dict of cosmetics obtained from 'read_input()' function.
    :param cosmetic_defaults:   Dict of default cosmetic values.
    :param cosmetics_list:      List of accepted cosmetic keywords.
    :param keywords:            Dict of keywords obtained from 'read_input()' function.
    :param keys_defaults:       Dict of default keyword values.
    :param keys_list:           List of accepted keywords.
    :param hatch_list:          List of possible hatches (bar patterns) understood by MatPlotLib.
    :param print_string:        The print_string with output - used in cases, where the input is not understood
                                and has been replaced by the default value.
    :return:                    The updated cosmetics and keywords dicts and print_string
                                as well as two new boolean variables, as the corresponding dict entries contain more
                                information than yes/no. Also returns the size of the figure based on the 'figtype'
                                cosmetic keyword and the location of the legend, 'legloc'.
    """
    gridbool, print_fit = False, False

    special_keys = ["devpattern", "show_legend", "legendbox", "diagonal", "gridlines", "plotall", "tight", "print_fit"]

    for key in cosmetics_list:

        if not key in cosmetics.keys():
            cosmetics[key] = cosmetic_defaults[key]

        if not key in special_keys:
            continue

        # ---------------------------------------------------------------------------------------------------

        if key == "devpattern":  # Specify the bar plot hatch patterns

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

                        for i in np.arange(len(hatch_list) - 1) + 1:
                            print_string += f", \"{hatch_list[i]}\""
                        print_string += "\n\n\n\n"

                while len(hatches) < 4:
                    hatches.append(None)

            else:
                print(f"Invalid value for the \"{key}\" titlecard keyword. "
                      f"The default value of \"{cosmetic_defaults[key]}\" is used.")
                hatches = [None, None, None, None]

            cosmetics[key] = hatches

        # ---------------------------------------------------------------------------------------------------

        elif key == "show_legend":

            if cosmetics[key] == "on":
                show_legend = True

            elif cosmetics[key] == "off":
                show_legend = False

            else:
                print(f"Invalid value for the \"{key}\" titlecard keyword. "
                      f"The default value of \"{cosmetic_defaults[key]}\" is used.")
                show_legend = True

            cosmetics[key] = show_legend

        # ---------------------------------------------------------------------------------------------------

        elif key == "legendbox":

            if cosmetics[key] == "on":
                legbox = True

            elif cosmetics[key] == "off":
                legbox = False

            else:
                print(f"Invalid value for the \"{key}\" titlecard keyword. "
                      f"The default value of \"{cosmetic_defaults[key]}\" is used.")
                legbox = True

            cosmetics[key] = legbox

        # ---------------------------------------------------------------------------------------------------

        elif key == "diagonal":

            if cosmetics[key] == "on":
                diagonal = True

            elif cosmetics[key] == "off":
                diagonal = False

            else:
                print(f"Invalid value for the \"{key}\" titlecard keyword. "
                      f"The default value of \"{cosmetic_defaults[key]}\" is used.")
                diagonal = False

            cosmetics[key] = diagonal

        # ---------------------------------------------------------------------------------------------------

        elif key == "gridlines":

            if cosmetics[key] == "off":
                gridbool = False

            elif (cosmetics[key] == "x") or (cosmetics[key] == "y") or (cosmetics[key] == "both"):
                gridbool = True

            else:
                print(f"Invalid value for the \"{key}\" titlecard keyword. "
                      f"The default value of \"{cosmetic_defaults[key]}\" is used.")
                gridbool = False

        # ---------------------------------------------------------------------------------------------------

        elif key == "plotall":

            if cosmetics[key] == "on":
                plotall = True

            elif cosmetics[key] == "off":
                plotall = False

            else:
                print(f"Invalid value for the \"{key}\" titlecard keyword. "
                      f"The default value of \"{cosmetic_defaults[key]}\" is used.")
                plotall = False

            cosmetics[key] = plotall

        # ---------------------------------------------------------------------------------------------------

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

        # ---------------------------------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------------------------------------

    if cosmetics["figtype"] == "square":  # Set the figure size based on input (square or rectangular)
        figsize = np.array((6, 6))

    else:
        figsize = np.array((12, 6))

    # -------------------------------------------------------------------------------------------------------

    # Input default values for keywords
    for key in keys_list:

        if not key in keywords:  # If not key is specified, add default value
            keywords[key] = keys_default[key]

        if not keywords[key]:  # If key contains a Python-False statement (0, "", [], etc. ), add default
            keywords[key] = keys_default[key]

    # -------------------------------------------------------------------------------------------------------

    # Cast the input legloc position into a variable for ease of use
    try:
        legloc = ast.literal_eval(cosmetics['legloc'])  # Literal evaluation of tuples and floats

    except:
        legloc = cosmetics['legloc'].lower()  # Force strings into lower case

    return cosmetics, keywords, gridbool, print_fit, figsize, legloc, print_string


