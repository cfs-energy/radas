import numpy as np
from fortranformat import FortranRecordReader as FReader

def read_single_values(list_of_lines, format_spec, start_at_line):
    """
    Makes a 'generator', which is a special Python object that returns a value each time that it is called.

    This generator reads each line in the list of lines, and tries to format it according to format_spec
    For instance, the default format '5e16.9' will try to read 5 fortran-formatted floats in exponential format
    with a width of 16 characters and 9 characters after the decimal point
    Where there are less than 5 numbers on a line, the fortran reader will return 'None'. These are skipped (not
    returned by the generator)

    The generator returns the next value according to the specified format, and also the line number and position being read
    """
    for line_number, line in enumerate(list_of_lines):
        if line_number < start_at_line:
            continue
        
        try:
            values = FReader(f"({format_spec})").read(line)
        except ValueError as e:
            message = (
                f"fortranformat could not read line {line_number}: {line} with format ({format_spec})\n"
                "Make sure that your format_spec has the correct number of elements, the correct spacing and the correct format for the line.\n"
                "See https://docs.oracle.com/cd/E19957-01/805-4939/6j4m0vn9f/index.html for reference.\n"
                f"fortranformat error was: {e}"
            )
            raise ValueError(message) from e
            

        for value_position, value in enumerate(values):
            if value is not None:
                yield value, line_number, value_position

def read_1d_array(generator, number_of_values):
    """
    Take number_of_values from the generator and write it into a 1D array
    """
    array = np.zeros(number_of_values)
    line_number, value_position = 0, 0

    for i in range(number_of_values):
        value, line_number, value_position = next(generator)

        if i == 0:
            assert value_position == 0, \
                "Arrays are always defined as starting on a new line, but the first element in this array is not at position 0"

        array[i] = value

    return array, line_number, value_position

def read_2d_array(generator, rows, columns, fortran_order_arrays=False):
    """
    Take rows*columns from the generator and write it into a 2D array of
    shape (rows, columns), or (columns, rows) if transpose = True
    """
    number_of_values = rows * columns
    array, line_number, value_position = read_1d_array(generator, number_of_values)

    if fortran_order_arrays:
        array = array.reshape((rows, columns))
    else:
        array = array.reshape((columns, rows))

    return array, line_number, value_position
