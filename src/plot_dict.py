
import matplotlib.pyplot as plt
import numpy as np
import warnings

# Magnify intervals where font size matters
MAGNIFY_INTERVALS = [0.1, 1]

# Min and max appropriate font sizes
FONT_RANGE = [10.5, 50]

# Maximum allowed magnify. This will get multiplied by 0 - 1 value.
MAX_MAGNIFY = 15

# Increase font for title ratio.
TITLE_FONT_RATIO = 1.8

def plot_dict(dict_arrays, start_step=0, step_size=1, use_title=None, points_values=False, points_round=3,
              use_xlabel=None, use_ylabel=None,
              style_sheet='ggplot', use_grid=True, use_linestyles=None, font_size=None, width=3, height=1, magnify=1.2,
              use_dpi=50, path=None, show_plot=True):
    r"""
    Create plot from a single array of values.
    Arguments:
        dict_arrays (:obj:`dict([list])`):
            Dictionary of arrays that will get plotted. The keys in dictionary are used as labels and the values as
            arrays that get plotted.
        start_step (:obj:`int`, `optional`, defaults to :obj:`0`):
            Starting value of plot.This argument is optional and it has a default value attributed inside
            the function.
        step_size (:obj:`int`, `optional`, defaults to :obj:`q`):
            Steps shows on x-axis. Change if each steps is different than 1.This argument is optional and it has a
            default value attributed inside the function.
        use_title (:obj:`int`, `optional`):
            Title on top of plot. This argument is optional and it will have a `None` value attributed
            inside the function.
        points_values (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Display each point value on the plot. This argument is optional and it has a default value attributed
            inside the function.
        points_round (:obj:`int`, `optional`, defaults to :obj:`1`):
            Round decimal valus for points values. This argument is optional and it has a default value attributed
            inside the function.
        use_xlabel (:obj:`str`, `optional`):
            Label to use for x-axis value meaning. This argument is optional and it will have a `None` value attributed
            inside the function.
        use_ylabel (:obj:`str`, `optional`):
            Label to use for y-axis value meaning. This argument is optional and it will have a `None` value attributed
            inside the function.
        style_sheet (:obj:`str`, `optional`, defaults to :obj:`ggplot`):
            Style of plot. Use plt.style.available to show all styles. This argument is optional and it has a default
            value attributed inside the function.
        use_grid (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Show grid on plot or not. This argument is optional and it has a default value attributed inside
            the function.
        use_linestyles (:obj:`str`, `optional`, defaults to :obj:`-`):
            Style to use on line from ['-', '--', '-.', ':']. This argument is optional and it has a default
            value attributed inside the function.
        font_size (:obj:`int` or `float`, `optional`):
            Font size to use across the plot. By default this function will adjust font size depending on `magnify`
            value. If this value is set, it will ignore the `magnify` recommended font size. The title font size is by
            default `1.8` greater than font-size. This argument is optional and it will have a `None` value attributed
            inside the function.
        width (:obj:`int`, `optional`, defaults to :obj:`3`):
            Horizontal length of plot. This argument is optional and it has a default value attributed inside
            the function.
        height (:obj:`int`, `optional`, defaults to :obj:`1`):
            Height length of plot in inches. This argument is optional and it has a default value attributed inside
            the function.
        magnify (:obj:`float`, `optional`, defaults to :obj:`0.1`):
            Ratio increase of both with and height keeping the same ratio size. This argument is optional and it has a
            default value attributed inside the function.
        use_dpi (:obj:`int`, `optional`, defaults to :obj:`50`):
            Print resolution is measured in dots per inch (or “DPI”). This argument is optional and it has a default
            value attributed inside the function.
        path (:obj:`str`, `optional`):
            Path and file name of plot saved as image. If want to save in current path just pass in the file name.
            This argument is optional and it will have a None value attributed inside the function.
        show_plot (:obj:`bool`, `optional`, defaults to :obj:`1`):
            if you want to call `plt.show()`. or not (if you run on a headless server). This argument is optional and
            it has a default value attributed inside the function.
    Raises:
        ValueError: If `dict_arrays` is not of type `dictionary`.
        ValueError: If `dict_arrays` doesn't have string keys.
        ValueError: If `dict_arrays` doesn't have array values.
        ValueError: If `style_sheet` is not valid.
        ValueError: If `use_linestyle` is not valid.
        ValueError: If `points_values`of type list don't have same length as `dict_arrays`.
        DeprecationWarning: If `magnify` is se to values that don't belong to [0, 1] values.
        ValueError: If `font_size` is not `None` and smaller or equal to 0.
    """

    # Check if `dict_arrays` is the correct format.
    if not isinstance(dict_arrays, dict):
        # Raise value error.
        raise ValueError("`dict_arrays` needs to be a dictionary of values!")

    # Check each label
    for label, array in dict_arrays.items():
        # Check if format is correct.
        if not isinstance(label, str):
            # Raise value error.
            raise ValueError("`dict_arrays` needs string keys!")
        if not isinstance(array, list) or isinstance(array, np.ndarray):
            # Raise value error.
            raise ValueError("`dict_arrays` needs lists values!")

    # Make sure style sheet is correct.
    if style_sheet in plt.style.available:
        # Set style of plot
        plt.style.use(style_sheet)
    else:
        # Style is not correct.
        raise ValueError("`style_sheet=%s` is not in the supported styles: %s" % (str(style_sheet),
                                                                                  str(plt.style.available)))

    # Make sure `magnify` is in right range.
    if magnify > 1 or magnify <= 0:
        # Deprecation warning from last time.
        warnings.warn(f'`magnify` needs to have value in [0,1]! `{magnify}` will be converted to `0.1` as default.',
                      DeprecationWarning)
        # Convert to regular value 0.1.
        magnify = 0.1

    # all linestyles.
    linestyles = ['-', '--', '-.', ':']

    # Make sure `font_size` is set right.
    if (font_size is not None) and (font_size <= 0):
        # Raise value error -  is not correct.
        raise ValueError(f'`font_size` needs to be positive number! Invalid value {font_size}')

    # Font size select custom or adjusted on `magnify` value.
    font_size = font_size if font_size is not None else np.interp(magnify, MAGNIFY_INTERVALS, FONT_RANGE)

    # Font variables dictionary. Keep it in this format for future updates.
    font_dict = dict(
        family='DejaVu Sans',
        color='black',
        weight='normal',
        size=font_size,
    )

    # If single style value is passed, use it on all arrays.
    if use_linestyles is None:
        use_linestyles = ['-'] * len(dict_arrays)

    else:
        # Check if linestyle is set right.
        for use_linestyle in use_linestyles:
            if use_linestyle not in linestyles:
                # Raise error.
                raise ValueError("`linestyle=%s` is not in the styles: %s!" % (str(use_linestyle), str(linestyles)))

    # Check `points_value` type - it can be bool or list(bool).
    if isinstance(points_values, bool):
        # Convert to list.
        points_values = [points_values] * len(dict_arrays)
    elif isinstance(points_values, list) and (len(points_values) != len(dict_arrays)):
        raise ValueError('`points_values` of type `list` must have same length as dictionary!')

    # Single plot figure.
    plt.subplot(1, 2, 1)

    # Plot each array.
    for index, (use_label, array) in enumerate(dict_arrays.items()):
        # Set steps plotted on x-axis - we can use step if 1 unit has different value.
        if start_step > 0:
            # Offset all steps by start_step.
            steps = np.array(range(0, len(array))) * step_size + start_step
        else:
            steps = np.array(range(1, len(array) + 1)) * step_size

        # Plot array as a single line.
        plt.plot(steps, array, linestyle=use_linestyles[index], label=use_label)

        # Plots points values.
        if points_values[index]:
            # Loop through each point and plot the label.
            for x, y in zip(steps, array):
                # Add text label to plot.
                plt.text(x, y, str(round(y, points_round)), fontdict=font_dict)

    # Set horizontal axis name.
    plt.xlabel(use_xlabel, fontdict=font_dict)

    # Set vertical axis name.
    plt.ylabel(use_ylabel, fontdict=font_dict)

    # Adjust both axis labels font size at same time.
    plt.tick_params(labelsize=font_dict['size'])

    # Place legend best position.
    plt.legend(loc='best', fontsize=font_dict['size'])

    # Adjust font for title.
    font_dict['size'] *= TITLE_FONT_RATIO

    # Set title of figure.
    plt.title(use_title, fontdict=font_dict)

    # Rescale `magnify` to be used on inches.
    magnify *= MAX_MAGNIFY

    # Display grid depending on `use_grid`.
    plt.grid(use_grid)

    # Make figure nice.
    plt.tight_layout()

    # Get figure object from plot.
    fig = plt.gcf()

    # Get size of figure.
    figsize = fig.get_size_inches()

    # Change size depending on height and width variables.
    figsize = [figsize[0] * width * magnify, figsize[1] * height * magnify]

    # Set the new figure size with magnify.
    fig.set_size_inches(figsize)

    # There is an error when DPI and plot size are too large!
    try:
        # Save figure to image if path is set.
        fig.savefig(path, dpi=use_dpi, bbox_inches='tight') if path is not None else None
    except ValueError:
        # Deprecation warning from last time.
        warnings.warn(f'`magnify={magnify // 15}` is to big in combination'
                      f' with `use_dpi={use_dpi}`! Try using lower values for'
                      f' `magnify` and/or `use_dpi`. Image was saved in {path}'
                      f' with `use_dpi=50 and `magnify={magnify // 15}`!', Warning)
        # Set DPI to smaller value and warn user to use smaller magnify or smaller dpi.
        use_dpi = 50
        # Save figure to image if path is set.
        fig.savefig(path, dpi=use_dpi, bbox_inches='tight') if path is not None else None

    # Show plot.
    plt.show() if show_plot is True else None

    plt.clf()
    return