# RADAS: radiation curves from OpenADAS

This Python library is intended to be a simple interface to ADAS ADF11 files.

## Installation

This project is installed using `poetry`. If you don't already have `poetry` installed on your machine, follow the installation instructions at [python-poetry.org/docs/](https://python-poetry.org/docs/).

Once `poetry` is installed, download this project with `git clone git@github.com:cfs-energy-internal/radas.git`.

Next, go into the `radas` directory and run `poetry install`. That's it!

## Downloading ADAS files

Due to the ADAS license, we can't distribute ADAS files with the repository. Instead, we've made it easy to download files directly from OpenADAS.

First of all, look in `data_to_fetch.yaml`. Make sure that there's an entry for the atomic species that you're interested in. If not, add a new entry with the new species, the ADF11 files you want to download, and which year you want to use (check on [open.adas.ac.uk/adf11](https://open.adas.ac.uk/adf11) to see which years are available).

Then, run
```
adas_data/fetch_adas_data.py
```

This will download the ADAS files as well as an ADF11 reader. It will then compile the ADF11 reader using `f2py` and the default Fortran compiler on your machine.

## Running an analysis

To run an analysis, run
```
./run_radas.py <case>
```
where `<case>` is corresponds to a folder in `cases` (for example, `neon`). This will look inside that folder for a `input.yaml` file, and will write the output into a `cases/<case>/output` subfolder. To extend or change the analysis, modify the `input.yaml` file.

## Developing radas

If you'd like to develop `radas`, most of the heavy lifting is done in `radas/computation` and the outputs are generated in `radas/write_output.py` and `radas/plotting.py`. Hopefully you won't need to do much to the ADAS file download or reader, but if you do these can be found in `adas_data` and `radas/adas_file_readers`.
