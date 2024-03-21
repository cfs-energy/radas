# RADAS: radiated power curves from OpenADAS

This Python library downloads atomic data from OpenADAS, performs simple calculations and stores the result as a NetCDF file for use in other programs.

## Installing via PyPi (quick start)

`radas` is on PyPi. You should be able to install it using
```
pip install radas
```

Once you've installed `radas` into your environment, you should be able to run
```
radas --help
```
to get an overview of the CLI options. Every option has a sensible default, so you can just run `radas` at the command line and it will generate a processed NetCDF for every species that has `data_files` defined in `config.yaml`. This can take some time (especially for high-Z impurities such as tungsten). If you'd just like to run a few species, you can specify these on the command line like
```
radas -s hydrogen -s helium -s lithium
```

If you want to change the years of the databases downloaded, you will need to change the `config.yaml` file. To get a copy of this, run `radas_config` to get a copy of the `config.yaml` file in your current working directory. You can edit this file (see the configuration section below) and then use it by passing it as the `-c` or `--config` argument to `radas` (i.e. `radas -s hydrogen -c ./config.yaml`).

## Development installation

If you want to develop `radas`, excellent! For contributing to `radas`, we ask that you
1. use Issues to ask questions, request features and discuss planned improvements,
2. use Pull Requests to merge code into `main` (mark work-in-progress with `draft` in the title),
3. write `pytest` tests for new functionality (ideally aiming to cover all new lines of code).

### Prerequisites

* Python 3.9 or later
* A Fortran compiler, such as `gfortran`
* The `poetry` packaging and dependency manager

### Installation

The project is installed using [poetry](https://python-poetry.org/). If you haven't already installed poetry, the installation instructions can be found [here](https://python-poetry.org/docs/#installing-with-the-official-installer).

Once you have poetry installed, you can install `radas` by running
```
poetry install --with dev
```

Because we've added `in-project = true` in `poetry.toml`, the project will install in the `.venv` in the repository directory.

### Usage

Once you have installed `radas`, you should be able to run the following snippet
```
poetry run radas --species=hydrogen --plot
```
where `--species` can be
* a specific species such as `hydrogen` or `tungsten`
* `all` which runs all species which have available data
* `none` which doesn't perform any analysis, but can be combined with `--plot` to generate the output plots from existing NetCDF files

If anything goes wrong, the script will drop into an `ipdb` interpreter so you can debug any issues. 

#### What's going on under the hood?

The above snippet executes `run_radas_cli` in `radas/cli.py`, which performs the following steps

1. Connect to [OpenADAS](https://open.adas.ac.uk/)
2. Download the datasets listed in `radas/config.yaml` under `species:hydrogen:data_files` (where the values are the years to download) and store them in `radas/.data_files`.
3. Download the fortran source for the dataset readers and store them in `radas/readers`.
4. Compile the readers using a fortran compiler with `f2py`.
5. Use the compiled readers to read the downloaded data files and store them in xarray Dataset (in `read_rate_coeffs.py`).
6. Calculate the fractional abundance of each charge state according to the coronal approximation (in `coronal_equilibrium.py`).
7. Calculate the coronal mean charge ($\langle Z \rangle$) and radiated power coefficient ($L_z$) as a function of the plasma temperature and density (in `cli.py` for the mean charge and in `radiated_power.py` for the radiated power).
8. Time-integrate equations for the abundance of each charge state to give the fractional abundance as a function of time $n_z(t)$ for different refuelling rates (characterized by $n_e \tau$ where $\tau$ is a particle residence time, in `time_evolution.py`).
9. Calculate the equilibrium ($t \to \infty$) mean charge ($\langle Z \rangle$) and radiated power coefficient ($L_z$) as a function of the plasma temperature and density (reusing the same functions as for the coronal values).
10. Store all of the results in a NetCDF in the `output` folder and make a figure comparing the computed curves to data from *Mavrin, J. Fus. Eng., 2017* (where available).

### Configuration

`radas` is configured using the `config.yaml` file provided in the `radas` source repository. You can edit this file directly, or can point the CLI to another configuration YAML file using the `--config` argument. Regardless of which approach you choose, the `config.yaml` file must have the following structure
```
globals:
  evolution_start:
    value: <Time to start time-evolution>
    units: "s"

  evolution_stop:
    value: <Time to stop time-evolution>
    units: "s"

  ne_tau:
    value: <Values of ne * tau to generate output for>
    units: "m^-3 s"

data_file_config:
  adf11: #or other reader class, but usually we want ADF11
    <what to call the dataset in the output>:
      prefix: <letters used to identify the dataset>
      code: <code used to identify the dataset for the ADAS reader>
      stored_units: "cm**3 / s"
      desired_units: "m**3 / s"

species:
  <species name>:
    atomic_symbol: <symbol>
    atomic_number: <atomic number>
    data_files:
      <dataset matching "what to call the dataset in the output" above>: <year to download>
```

### Testing

To make sure everything is working, run
```
poetry run pytest
```
to execute all of the tests in the `tests` folder.

### Pushing to PyPi

To publish the repository to the [PyPi](pypi.org) package index, you can follow the instructions at [RealPython](https://realpython.com/pypi-publish-python-package/), adapted slightly for the poetry project.

First, you should edit `pyproject.toml` and set `version="YYYY.MM.V"`, where `YYYY` is the year, `MM` is the month and `V` is a version tag (reset at zero each month). Then, you can publish to PyPi by running the following
```
# Clean up any previous distributions
rm -rf ./dist

# Install the project
poetry lock
poetry install --with dev
# Make sure the tests all pass!
poetry run pytest

# Build and check the distribution
poetry build
poetry run twine check dist/*

# Test on https://test.pypi.org/ first!
poetry run twine upload -r testpypi dist/*

# Publish the package to the real Python Package Index
poetry run twine upload dist/*
```
