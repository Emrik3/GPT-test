import argparse
import copy
import json
import os
from functools import reduce
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

from gptopt.plot_utils import (
    get_alpha_from_lr,
    get_lr_and_name,
    plot_data,
    plot_step_size_and_lr,
    smoothen_dict,
)
from gptopt.utils import get_default_config, load_config

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 1.5
plt.rc("text", usetex=False)
plt.rc("legend", fontsize=10)


def load_outputs(output_dir):
    """Load all individual output files from a directory."""
    outputs = []
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r") as file:
                output = json.load(file)
                outputs.append(output)
    return outputs


def load_output_folder(experiment_results_folder, exclude_runs=[]):
    outputs = []
    for root, _, files in os.walk(experiment_results_folder):
        for file_name in files:
            if any(exclude == file_name for exclude in exclude_runs):
                continue
            if file_name.startswith("logs") and file_name.endswith(".json"):
                file_path = os.path.join(root, file_name)
                with open(file_path, "r") as file:
                    logs = json.load(file)
                with open(os.path.join(root, ".hydra/config.yaml"), "r") as file:
                    config = yaml.safe_load(file)
                outputs.append(dict(config=config, logs=logs))
    return outputs


def plot_final_loss_vs_lr(
    outputs,
    colormap,
    outfilename,
    linestylemap,
    val=False,
    y_top_lim=None,
    y_bottom_lim=None,
    # Zoom region - tune these to your data; set to None to skip inset
    zoom_xlim=(0.004, 0.01),
    zoom_ylim=(3.75, 3.85),
):
    """Plot final loss versus learning rate as lines for each method."""
    fig, ax = plt.subplots(figsize=(6, 4))
    methods = {}

    for output in outputs:
        name, lr = get_lr_and_name(output)
        lr = float(lr)
        if val:
            if "val_losses" not in output["logs"]:
                continue
            final_loss = output["logs"]["val_losses"][-1]
        else:
            final_loss = output["logs"]["losses"][-1]
        if name not in methods:
            methods[name] = {"lrs": [], "losses": []}
        methods[name]["lrs"].append(lr)
        methods[name]["losses"].append(final_loss)

    for name, data in methods.items():
        sorted_indices = sorted(range(len(data["lrs"])), key=lambda i: data["lrs"][i])
        sorted_lrs = [data["lrs"][i] for i in sorted_indices]
        if len(set(sorted_lrs)) < len(sorted_lrs):
            print(f"Warning: Duplicate learning rates for method {name}.")
        sorted_losses = [data["losses"][i] for i in sorted_indices]
        ax.plot(
            sorted_lrs,
            sorted_losses,
            alpha=0.85,
            label=name,
            color=colormap.get(name, None),
            linestyle=linestylemap.get(name, None),
            linewidth=2,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    if val:
        ax.set_ylabel("Final Validation Loss")
        plotfile = "figures/" + outfilename + "-lr-sens-val.pdf"
    else:
        ax.set_ylabel("Final Loss")
        plotfile = "figures/" + outfilename + "-lr-sens.pdf"
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="both", lw=0.2, ls="--", zorder=0)
    if y_top_lim is not None:
        ax.set_ylim(top=y_top_lim)
    if y_bottom_lim is not None:
        ax.set_ylim(bottom=y_bottom_lim)

    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.95)

    # ---- Zoomed inset ----
    if zoom_xlim is not None and zoom_ylim is not None:
        # Force layout so get_position() returns final coords
        fig.canvas.draw()
        ax_pos = ax.get_position()

        # Place inset in lower-left of the axes, fully inside figure bounds
        inset_w = ax_pos.width * 0.36
        inset_h = ax_pos.height * 0.40
        inset_x = ax_pos.x0 + ax_pos.width * 0.05
        inset_y = ax_pos.y0 + ax_pos.height * 0.07
        axins = fig.add_axes([inset_x, inset_y, inset_w, inset_h])

        x1, x2 = zoom_xlim
        y1, y2 = zoom_ylim
        axins.set_xscale("log")
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticks([])
        axins.set_yticks([])
        axins.tick_params(
            which="both", left=False, bottom=False, labelleft=False, labelbottom=False
        )
        for spine in axins.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor("0.4")
        axins.patch.set_facecolor("white")
        axins.patch.set_alpha(1.0)
        axins.grid(axis="both", lw=0.15, ls="--", zorder=0, alpha=0.5)

        for name, data in methods.items():
            sorted_indices = sorted(
                range(len(data["lrs"])), key=lambda i: data["lrs"][i]
            )
            axins.plot(
                [data["lrs"][i] for i in sorted_indices],
                [data["losses"][i] for i in sorted_indices],
                color=colormap.get(name, None),
                linestyle=linestylemap.get(name, None),
                linewidth=1.5,
                alpha=0.9,
            )

        # loc1=2, loc2=1: upper corners of inset connect to zoom box
        # → connector lines go upward, staying inside the figure
        mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec="0.40", lw=0.7)

    # Do NOT use bbox_inches="tight" — it expands the figure to include
    # any connector lines that extend outside the axes, causing a ~38000px tall PDF
    fig.savefig(plotfile, format="pdf")


def main(
    outputs, outfilename, y_top_lim_lrs=None, y_bottom_lim_lrs=None, y_top_vs_time=None
):
    for output in outputs:  # Smoothing
        smoothen_dict(output["logs"], num_points=100, beta=0.05)

    colormap = {
        "muon-MachPolar5": "#A65900",
        "muon-MachPolar9": "#A65900",
        "muon-MachPolar17": "#A65900",
        "muon-MachPolar172": "#A65900",
        "muon-MachPolar23": "#A65900",
        "sgd-sch": "#B3CBB9",
        "adam": "#0D4A21",
        "adamw": "#0D4A21",
        "adam-sch": "#0D4A21",
        "momo": "#61ACE5",
        "muon-PolarExp": "#78001A",
        "muon-You": "#8A2BE2",  # Added a new color for "muon" (blue-violet)
        "muon-Jordan": "#4DA060",
    }
    linestylemap = {
        "momo": None,
        "muon-MachPolar5": None,
        "muon-MachPolar9": None,
        "muon-MachPolar17": None,
        "muon-MachPolar172": None,
        "muon-MachPolar23": None,
        "sgd-sch": "--",
        "muon-PolarExp": None,
        "adam": None,
        "adamw": None,
        "adam-sch": "--",
        "muon-You": ":",
        "muon-Jordan": None,
    }

    # Collect learning rate ranges for each method
    lr_ranges = {}
    for output in outputs:
        name, lr = get_lr_and_name(output)
        if name not in lr_ranges:
            lr_ranges[name] = [lr, lr]
        else:
            lr_ranges[name][0] = min(lr_ranges[name][0], lr)
            lr_ranges[name][1] = max(lr_ranges[name][1], lr)

    # Michael: Temparily resetting matplotlib settings to default so that latex doesn't
    # need to be used for plot formatting. Was giving me an error.
    # import matplotlib as mpl
    # mpl.rcParams.update(mpl.rcParamsDefault)

    best_outputs = {}
    best_lr = {}
    for output in outputs:
        name, lr = get_lr_and_name(output)
        if "val_losses" not in output["logs"]:
            continue
        final_val_loss = output["logs"]["val_losses"][-1]
        if (
            name not in best_outputs
            or final_val_loss < best_outputs[name]["logs"]["val_losses"][-1]
        ):
            best_outputs[name] = output
            lr = float(lr)
            best_lr[name] = [lr, lr]
    Path("figures/" + outfilename + ".pdf").parent.mkdir(parents=True, exist_ok=True)
    for name, output in best_outputs.items():
        print(
            f"Best {name}-{best_lr[name][0]} final val loss: {output['logs']['val_losses'][-1]}"
        )
    # print(f"Best {name} lr: {lr}")
    # Plot final loss vs learning rate
    plot_final_loss_vs_lr(
        outputs,
        colormap,
        outfilename,
        linestylemap,
        y_top_lim=y_top_lim_lrs,
        y_bottom_lim=y_bottom_lim_lrs,
    )
    plot_final_loss_vs_lr(
        outputs,
        colormap,
        outfilename,
        linestylemap,
        val=True,
        y_top_lim=y_top_lim_lrs,
        y_bottom_lim=y_bottom_lim_lrs,
    )
    # Plot loss
    selected_outputs = list(best_outputs.values())
    get_alpha_from_lr = lambda lr, lr_range: 0.85
    initial_loss = (
        selected_outputs[0]["logs"]["val_losses"][0]
        if selected_outputs and "val_losses" in selected_outputs[0]["logs"]
        else 1.0
    )  # Default to 1.0 if not available
    upper_bound = initial_loss * 1.0  # Set upper bound to 70% above the initial loss
    fig, ax = plt.subplots(figsize=(4, 3))
    plot_data(
        ax,
        selected_outputs,
        max(
            o["config"]["training_data"]["training_params"]["num_epochs"]
            for o in selected_outputs
        ),
        "val_losses",
        "Validation Loss",
        colormap,
        linestylemap,
        best_lr,
        get_alpha_from_lr,
    )
    lower_bound = min(
        min(output["logs"]["val_losses"])
        for output in selected_outputs
        if "val_losses" in output["logs"]
    )
    ax.set_ylim(lower_bound * 0.975, upper_bound)
    if y_top_vs_time is not None:
        ax.set_ylim(top=y_top_vs_time)
    ax.tick_params(axis="both", which="major", labelsize=8)  # Set tick label font size
    ax.set_xlabel("Epoch", fontsize=10)  # Set x-axis label font size
    ax.set_ylabel("Validation Loss", fontsize=10)
    # Set the upper bound
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=4, fontsize=10)
    ax.legend(
        loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10
    )  # Legend placed next to the figure
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    fig.subplots_adjust(
        top=0.99, bottom=0.155, left=0.12, right=0.8
    )  # Adjust right to make space for legend
    fig.savefig("figures/" + outfilename + ".pdf", format="pdf", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(4, 3))
    plot_data(
        ax,
        selected_outputs,
        max(
            o["config"]["training_data"]["training_params"]["num_epochs"]
            for o in selected_outputs
        ),
        "losses",
        "Loss",
        colormap,
        linestylemap,
        best_lr,
        get_alpha_from_lr,
        time=True,
    )
    ax.set_ylim(lower_bound * 0.975, upper_bound)  # Set the upper bound
    if y_top_vs_time is not None:
        ax.set_ylim(top=y_top_vs_time)
    ax.tick_params(axis="both", which="major", labelsize=8)  # Set tick label font size
    ax.set_xlabel("Time (s)", fontsize=10)  # Set x-axis label font size
    ax.set_ylabel("Loss", fontsize=10)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=4, fontsize=10)
    ax.legend(
        loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10
    )  # Legend placed next to the figure
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    fig.subplots_adjust(
        top=0.99, bottom=0.155, left=0.12, right=0.8
    )  # Adjust right to make space for legend
    fig.savefig(
        "figures/" + outfilename + "-time.pdf", format="pdf", bbox_inches="tight"
    )
    # initial_loss = outputs[0]['losses'][0] if outputs and 'losses' in outputs[0] else 1.0  # Default to 1.0 if not available
    # upper_bound = initial_loss * 1.2  # Set upper bound to 20% above the initial loss
    # fig, ax = plt.subplots(figsize=(4, 3))
    # plot_data(ax, outputs,  config['training_params']['num_epochs'], 'losses', 'Loss', colormap, linestylemap, lr_ranges, get_alpha_from_lr)
    # lower_bound = min(min(output['losses']) for output in outputs if 'losses' in output)
    # ax.set_ylim(lower_bound, upper_bound) # Set the upper bound
    # ax.legend(loc='upper right', fontsize=10)
    # fig.subplots_adjust(top=0.99, bottom=0.155, left=0.12, right=0.99)
    # fig.savefig('figures/' + outfilename + '.pdf', format='pdf', bbox_inches='tight')

    # # Plot learning rates
    # for method_subset in [['sgd-m', 'sgd-sch', 'momo'], ['adam', 'adam-sch', 'momo-adam']]:
    #     fig, ax = plt.subplots(figsize=(4, 3))
    #     subset_outputs = [output for output in outputs if output['name'].split('-lr-')[0] in method_subset]
    #     plot_data(ax, subset_outputs, config['training_params']['num_epochs'], 'learning_rates', 'Learning rate', colormap, linestylemap, lr_ranges,  get_alpha_from_lr)
    #     ax.legend(loc='upper right', fontsize=10)
    #     fig.subplots_adjust(top=0.935, bottom=0.03, left=0.155, right=0.99)
    #     name = 'figures/lr-' if 'sgd-m' in method_subset else 'figures/lr-adam-'
    #     fig.savefig(name + outfilename + '.pdf', format='pdf', bbox_inches='tight')

    # # Plot step size lists
    # fig, ax = plt.subplots(figsize=(4, 3))
    # plotted_methods = plot_step_size_and_lr(ax, outputs, colormap, linestylemap, lr_ranges, get_alpha_from_lr)
    # handles, labels = ax.get_legend_handles_labels()
    # legend_handles = [copy.copy(handle) for handle in handles]
    # for handle in legend_handles:
    #     handle.set_alpha(1.0)
    # ax.legend(legend_handles, labels, loc='upper right', fontsize=10)
    # ax.set_xlabel('Step')
    # ax.set_ylabel('Learning Rate')
    # fig.subplots_adjust(top=0.99, bottom=0.155, left=0.12, right=0.99)
    # fig.savefig('figures/step_size-' + outfilename + '.pdf', format='pdf', bbox_inches='tight')


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Plotting gpt_distill outputs.')
    # parser.add_argument('--results_folder', type=str, nargs='?', help='Path to results folder', default=None)

    # results_folder = "outputs/hydra-results/main_run"
    # lims = dict(y_top_lim_lrs=3.7, y_top_vs_time=4.5)
    # results_folder = "outputs/hydra-results/10b_data"
    # lims = dict(y_top_vs_time=3.5)
    results_folder = "outputs/hydra-results/main_run/2026-04-29"
    lims = dict(y_top_lim_lrs=4.6, y_top_vs_time=4.6)
    exclude_runs = [
        "logs_jobid_87335a44.json",
        "logs_jobid_e279af21.json",
        "logs_jobid_1b192fba.json",
        "logs_jobid_41c4f70d.json",
        "logs_jobid_4975d367.json",
        "logs_jobid_fd0e0465.json",
    ]
    outputs = load_output_folder(results_folder, exclude_runs=exclude_runs)
    print("Total num experiments:", len(outputs))

    for weight_decay in set(
        output["config"]["optimizer_params"]["args"]["weight_decay"]
        for output in outputs
    ):
        small_outputs = [
            output
            for output in outputs
            if output["config"]["optimizer_params"]["args"]["weight_decay"]
            == weight_decay
        ]
        for nlayer in set(
            output["config"]["gpt_model"]["n_layer"] for output in small_outputs
        ):
            smaller_outputs = [
                output
                for output in small_outputs
                if output["config"]["gpt_model"]["n_layer"] == nlayer
            ]
            assert len(small_outputs) > 0
            experiment_name = os.path.basename(results_folder.rstrip("/"))
            outfilename = experiment_name + f"-nl-{nlayer}" + "-wd-" + str(weight_decay)
            print("nlayer: ", nlayer, " weight_decay: ", weight_decay)
            print(f"Loaded {len(smaller_outputs)} outputs from {results_folder}")
            main(smaller_outputs, f"{experiment_name}/{outfilename}", **lims)
