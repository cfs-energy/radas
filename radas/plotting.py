import matplotlib.pyplot as plt
import xarray as xr
import sys
from radas.directories import cases_directory
from .get_git_hash import get_git_revision_short_hash

def make_plots(dataset: xr.Dataset, key: str, plot_params: dict, figsize, show_dpi, save_dpi):
    """Make a plot using the data from dataset as specified in the "plot" dictionary."""

    if plot_params["type"] == "skip": return

    fig, ax = plt.subplots(figsize=figsize, dpi=show_dpi,
                            nrows=plot_params.get("nrows", 1), ncols=plot_params.get("ncols", 1), sharex=True, sharey=True)

    ds_slice = dataset
    for variable, variable_slice in plot_params.get("slice", {}).items():
        ds_slice = ds_slice.sel({f"dim_{variable}": variable_slice}, method="nearest")

    if plot_params["type"] == "xrplot2d":
        make_xrplot2d(ds_slice, plot_params, fig, ax)
    elif plot_params["type"] == "xrplot1d":
        make_xrplot1d(ds_slice, plot_params, fig, ax)
    else:
        raise NotImplementedError(f"No implementation for plot type {plot_params['type']}")

    ax.set_xscale(plot_params.get("xscale", "linear"))
    ax.set_yscale(plot_params.get("yscale", "linear"))
    ax.set_xlabel(plot_params.get("xlabel", ""))
    ax.set_ylabel(plot_params.get("ylabel", ""))
    ax.set_title(f"{plot_params.get('title', '')} [{get_git_revision_short_hash()}]")
    if plot_params.get("legend", False):
        ax.legend()
    if plot_params.get("grid", False):
        ax.grid()

    save_name = f"{key}_{get_git_revision_short_hash()}" if plot_params.get("include_git_hash_in_name", False) else key
    if not "pytest" in sys.modules: #skip saving output if running tests
        plt.savefig(cases_directory / dataset.case / "output" / save_name, dpi=save_dpi, bbox_inches="tight")

    if not plot_params["show"]:
        plt.close(fig)

def make_xrplot2d(dataset, plot_params, fig, ax):

    im = dataset[plot_params["variable"]].plot(ax=ax, add_colorbar=False, add_labels=False)
    cbar = fig.colorbar(im, ax=ax)

def make_xrplot1d(dataset, plot_params, fig, ax):
    
    if "iterate_over" in plot_params:
        iteration_dim = plot_params["iterate_over"]
        in_legend = plot_params.get("in_legend", "none")
        for i in range(dataset.sizes[iteration_dim]):
            if in_legend == "none":
                legend_val = ""
            elif in_legend == "value":
                legend_val = dataset[iteration_dim].isel({iteration_dim: i}).values
            elif in_legend == "index":
                legend_val = i
            else:
                if in_legend in dataset:
                    legend_val = dataset[in_legend].isel({iteration_dim: i}).values
                else:
                    raise NotImplementedError(f"'in_legend' should be one of 'none', 'value', 'index' or a dataset variable, but was '{in_legend}'")
            
            legend_format = plot_params.get("legend_format", "4.3g")
            label = plot_params.get("legend_base", "#").replace("#", f"{legend_val:{legend_format}}")

            dataset[plot_params["variable"]].isel({iteration_dim: i}).plot(ax=ax, label=label)
    else:
        dataset[plot_params["variable"]].plot(ax=ax)
