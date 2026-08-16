from pathlib import Path

import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

import matplotlib.pyplot as plt
from style_thesis import (
    CATEGORY_COLORS,
    CATEGORY_LABELS,
    CATEGORY_ORDER,
    COLOR_AUX,
    savefig_thesis,
    style_axes,
)

COLOR_UA = "black"
EVENT_COLOR_RECAP = "dodgerblue"
EVENT_COLOR_RELOST = "crimson"


def _rising_event_counts(mask):
    events = np.zeros(mask.shape[0], dtype=int)
    events[0] = np.sum(mask[0])
    events[1:] = np.sum(mask[1:] & ~mask[:-1], axis=1)
    return events


def _falling_event_counts(mask):
    events = np.zeros(mask.shape[0], dtype=int)
    events[1:] = np.sum(mask[:-1] & ~mask[1:], axis=1)
    return events


def _window_around(t, center_idx, half_width):
    center = t[center_idx]
    lo = max(t[0], center - half_width)
    hi = min(t[-1], center + half_width)
    return lo, hi


def _plot_ux_ua_panel(ax, t, ux, ua, mask, title):
    ax.plot(t[mask], ux[mask], color=COLOR_AUX, lw=2.0, label="ux")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=11, labelcolor="black")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    style_axes(ax)

    ax2 = ax.twinx()
    ax2.plot(t[mask], ua[mask], color=COLOR_UA, lw=1.6, ls="--", label="ua")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(labelsize=11, labelcolor="black")
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax2.spines["top"].set_visible(False)

    return ax2


def _plot_control_inset(ax_parent, bbox, t, ux, ua, mask, title):
    axins = ax_parent.inset_axes(bbox)
    axins.set_facecolor("white")
    axins.patch.set_alpha(0.95)
    return _plot_ux_ua_panel(axins, t, ux, ua, mask, title)


def _event_rug(ax, t, times_mask_rising, times_mask_falling):
    for t_evt in t[times_mask_rising]:
        ax.axvline(t_evt, color=EVENT_COLOR_RECAP, alpha=0.6, lw=1.0, zorder=1)
    for t_evt in t[times_mask_falling]:
        ax.axvline(t_evt, color=EVENT_COLOR_RELOST, alpha=0.7, lw=1.2, zorder=2)


def _plot_stacked_counts(ax, t, b0, b1, b2, b3, n_shots):
    ax.fill_between(t, 0, b0, color=CATEGORY_COLORS["survived"], alpha=0.35)
    ax.fill_between(t, b0, b1, color=CATEGORY_COLORS["left_behind"], alpha=0.35)
    ax.fill_between(t, b1, b2, color=CATEGORY_COLORS["recaptured"], alpha=0.35)
    ax.fill_between(t, b2, b3, color=CATEGORY_COLORS["lost"], alpha=0.35)
    ax.plot(t, b0, color=CATEGORY_COLORS["survived"], lw=2.0)
    ax.plot(t, b1, color=CATEGORY_COLORS["left_behind"], lw=2.0)
    ax.plot(t, b2, color=CATEGORY_COLORS["recaptured"], lw=2.0)
    ax.plot(t, b3, color=CATEGORY_COLORS["lost"], lw=2.0)
    ax.axhline(n_shots, color="gray", ls=":", lw=1.0, alpha=0.5)
    ax.set_ylabel("Number of atoms", fontsize=13)
    ax.set_ylim(0, n_shots * 1.05)
    style_axes(ax)


def _mark_window(ax, window):
    for x in window:
        ax.axvline(x, color="black", ls="--", lw=1.0, alpha=0.6, zorder=4)


def _label_region(ax, window, label):
    x_center = 0.5 * (window[0] + window[1])
    ax.text(
        x_center,
        0.06,
        label,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        zorder=5,
    )


def plot_classification_thesis(
    data,
    classification,
    output_dir,
    top_only=False,
    transport_only=False,
    control_row=False,
    summary_row=False,
):
    t = data["t"]
    ux = data["ux"]
    ua = data["ua"]
    n_shots = data["x"].shape[1]
    transport_time = data["transport_time"]

    n_survived = np.sum(classification["survived"], axis=1)
    n_left_behind = np.sum(classification["left_behind"], axis=1)
    n_recaptured = np.sum(classification["recaptured"], axis=1)
    n_lost = np.sum(classification["lost"], axis=1)
    survival_pct = 100.0 * n_survived / n_shots

    b0 = n_survived
    b1 = b0 + n_left_behind
    b2 = b1 + n_recaptured
    b3 = b2 + n_lost

    ax_a = ax_b = ax_ctrl = ax_ctrl2 = None

    if transport_only:
        fig = plt.figure(figsize=(11, 7.5))
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.32, wspace=0.25)
        ax1 = fig.add_subplot(gs[0, :])
        ax_a = fig.add_subplot(gs[1, 0])
        ax_b = fig.add_subplot(gs[1, 1])
        _plot_stacked_counts(ax1, t, b0, b1, b2, b3, n_shots)
        ax1.set_xlim(t[0], transport_time)
        ax1.set_xlabel("Time [μs]", fontsize=13)
        legend_handles = [
            Patch(facecolor=CATEGORY_COLORS[c], alpha=0.35, label=CATEGORY_LABELS[c])
            for c in CATEGORY_ORDER
        ]
        ax1.legend(
            handles=legend_handles,
            loc="lower right",
            bbox_to_anchor=(0.97, 0.05),
            fontsize=10,
            framealpha=0.85,
        )
    elif summary_row:
        ctrl_mask = t <= transport_time
        recap_pct = 100.0 * n_recaptured[-1] / n_shots
        lost_pct = 100.0 * n_lost[-1] / n_shots
        fig = plt.figure(figsize=(11, 8.0))
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1.1], hspace=0.1)
        ax1 = fig.add_subplot(gs[0])
        ax_ctrl = fig.add_subplot(gs[1], sharex=ax1)
        _plot_stacked_counts(ax1, t, b0, b1, b2, b3, n_shots)
        ax1.set_xlim(t[0], transport_time)
        plt.setp(ax1.get_xticklabels(), visible=False)
        left_behind_pct = 100.0 * n_left_behind[-1] / n_shots
        summary_pct = {
            "survived": survival_pct[-1],
            "left_behind": left_behind_pct,
            "recaptured": recap_pct,
            "lost": lost_pct,
        }
        legend_handles = [
            Patch(
                facecolor=CATEGORY_COLORS[c],
                alpha=0.35,
                label=f"{CATEGORY_LABELS[c]}: {summary_pct[c]:.1f}%",
            )
            for c in ("survived", "left_behind", "recaptured", "lost")
        ]
        ax1.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(0.97, 0.5),
            fontsize=10,
            framealpha=0.85,
        )

        ax_ctrl.plot(t[ctrl_mask], ux[ctrl_mask], color="crimson", lw=2.0, label="ux")
        ax_ctrl.set_xlim(t[0], transport_time)
        ax_ctrl.set_xlabel("Time [μs]", fontsize=13)
        ax_ctrl.set_ylabel("Position [μm]", fontsize=13, color="crimson")
        ax_ctrl.tick_params(labelsize=11, labelcolor="crimson", axis="y")
        style_axes(ax_ctrl)

        ax_ctrl2 = ax_ctrl.twinx()
        ax_ctrl2.plot(
            t[ctrl_mask], ua[ctrl_mask], color="black", lw=1.6, ls=":", label="ua"
        )
        ax_ctrl2.set_ylim(0, 1.05)
        ax_ctrl2.set_ylabel("Amplitude", fontsize=13)
        ax_ctrl2.tick_params(labelsize=11)
        ax_ctrl2.spines["top"].set_visible(False)
    elif control_row:
        w0_um = data["scales"]["w0_um"]
        ctrl_mask = t <= transport_time
        fig = plt.figure(figsize=(11, 8.0))
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1.1], hspace=0.32)
        ax1 = fig.add_subplot(gs[0])
        ax_ctrl = fig.add_subplot(gs[1])
        _plot_stacked_counts(ax1, t, b0, b1, b2, b3, n_shots)
        ax1.set_xlim(t[0], transport_time)
        ax1.set_xlabel("Time [μs]", fontsize=13)
        legend_handles = [
            Patch(facecolor=CATEGORY_COLORS[c], alpha=0.35, label=CATEGORY_LABELS[c])
            for c in CATEGORY_ORDER
        ]
        ax1.legend(
            handles=legend_handles,
            loc="lower right",
            bbox_to_anchor=(0.97, 0.05),
            fontsize=10,
            framealpha=0.85,
        )

        ax_ctrl.plot(t[ctrl_mask], ux[ctrl_mask], color=COLOR_AUX, lw=2.0, label="ux")
        ax_ctrl.set_xlim(t[0], transport_time)
        ax_ctrl.set_xlabel("Time [μs]", fontsize=13)
        ax_ctrl.set_ylabel(f"ux  [w₀ = {w0_um:.1f} μm]", fontsize=13)
        ax_ctrl.tick_params(labelsize=11)
        style_axes(ax_ctrl)

        ax_ctrl2 = ax_ctrl.twinx()
        ax_ctrl2.plot(
            t[ctrl_mask], ua[ctrl_mask], color=COLOR_UA, lw=1.6, ls="--", label="ua"
        )
        ax_ctrl2.set_ylim(0, 1.05)
        ax_ctrl2.set_ylabel("ua  [dimless]", fontsize=13)
        ax_ctrl2.tick_params(labelsize=11)
        ax_ctrl2.spines["top"].set_visible(False)
    elif top_only:
        fig, ax1 = plt.subplots(1, 1, figsize=(11, 5.5))
        _plot_stacked_counts(ax1, t, b0, b1, b2, b3, n_shots)
        ax1.set_xlabel("Time [μs]", fontsize=13)
        legend_handles = [
            Patch(facecolor=CATEGORY_COLORS[c], alpha=0.35, label=CATEGORY_LABELS[c])
            for c in CATEGORY_ORDER
        ]
        ax1.legend(
            handles=legend_handles,
            loc="lower right",
            bbox_to_anchor=(0.97, 0.05),
            fontsize=10,
            framealpha=0.85,
        )
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
        _plot_stacked_counts(ax1, t, b0, b1, b2, b3, n_shots)
        ax1.set_title(
            f"Atom classification over time — {n_shots} shots, "
            f"final survival: {int(n_survived[-1])}/{n_shots} ({survival_pct[-1]:.1f}%)",
            fontsize=14,
            fontweight="bold",
        )

        ax2.plot(t, survival_pct, color=CATEGORY_COLORS["survived"], lw=2.5)
        ax2.fill_between(
            t, 0, survival_pct, color=CATEGORY_COLORS["survived"], alpha=0.25
        )
        ax2.axhline(100, color="gray", ls="--", lw=1.0, alpha=0.5)
        ax2.set_xlabel("Time [μs]", fontsize=13)
        ax2.set_ylabel("Survival rate [%]", fontsize=13)
        ax2.set_ylim(0, 105)
        style_axes(ax2)

        recap_rising = _rising_event_counts(classification["is_caught_again"]) > 0
        recap_falling = _falling_event_counts(classification["is_caught_again"]) > 0
        _event_rug(ax2, t, recap_rising, recap_falling)

        legend_handles = [
            Patch(facecolor=CATEGORY_COLORS[c], alpha=0.35, label=CATEGORY_LABELS[c])
            for c in CATEGORY_ORDER
        ]
        legend_handles.append(
            Patch(facecolor=EVENT_COLOR_RECAP, alpha=0.6, label="Recaptured event")
        )
        legend_handles.append(
            Patch(
                facecolor=EVENT_COLOR_RELOST, alpha=0.7, label="Relapsed-to-lost event"
            )
        )
        ax2.legend(
            handles=legend_handles, loc="lower left", fontsize=10, framealpha=0.85
        )

    if not summary_row:
        loss_events = _rising_event_counts(classification["is_lost"])
        recap_events = _rising_event_counts(classification["is_caught_again"])
        half_width = 0.025 * transport_time

        loss_idx = int(np.argmax(loss_events))
        recap_idx = int(np.argmax(recap_events)) if np.any(recap_events) else None

        labelled = top_only or transport_only or control_row
        loss_bbox = [0.32, 0.06, 0.28, 0.38] if top_only else [0.1, 0.08, 0.26, 0.36]
        loss_title = "(a) Highest loss" if labelled else "Highest loss"
        recap_title = "(b) Highest recapture" if labelled else "Highest recapture"

        loss_window = _window_around(t, loss_idx, half_width)
        ax1.axvspan(*loss_window, color=CATEGORY_COLORS["lost"], alpha=0.15)
        loss_mask = (t >= loss_window[0]) & (t <= loss_window[1])
        if transport_only:
            assert ax_a is not None
            _plot_ux_ua_panel(ax_a, t, ux, ua, loss_mask, loss_title)
        elif control_row:
            assert ax_ctrl is not None
            _mark_window(ax_ctrl, loss_window)
        else:
            _plot_control_inset(ax1, loss_bbox, t, ux, ua, loss_mask, loss_title)
        if labelled:
            _mark_window(ax1, loss_window)
            _label_region(ax1, loss_window, "(a)")

        if recap_idx is not None:
            recap_window = _window_around(t, recap_idx, half_width)
            ax1.axvspan(*recap_window, color=CATEGORY_COLORS["recaptured"], alpha=0.15)
            recap_mask = (t >= recap_window[0]) & (t <= recap_window[1])
            if transport_only:
                assert ax_b is not None
                _plot_ux_ua_panel(ax_b, t, ux, ua, recap_mask, recap_title)
            elif control_row:
                assert ax_ctrl is not None
                _mark_window(ax_ctrl, recap_window)
            else:
                _plot_control_inset(
                    ax1, [0.65, 0.48, 0.26, 0.36], t, ux, ua, recap_mask, recap_title
                )
            if labelled:
                _mark_window(ax1, recap_window)
                _label_region(ax1, recap_window, "(b)")
        elif transport_only:
            assert ax_b is not None
            ax_b.axis("off")

    plt.tight_layout()
    if transport_only:
        suffix = "_transport_only"
    elif control_row:
        suffix = "_control_row"
    elif summary_row:
        suffix = "_summary_row"
    elif top_only:
        suffix = "_top_only"
    else:
        suffix = ""
    savefig_thesis(
        fig, Path(output_dir) / f"survival_classification_thesis{suffix}.png"
    )
