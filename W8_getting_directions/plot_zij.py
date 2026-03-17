"""
Improved Zijderveld (vector component) plot function.

Based on pmagplotlib.plot_zij with the following differences:
- label_steps / label_list: annotate with actual treatment levels, optionally
  selecting which steps to show
- ax: plot onto an existing matplotlib Axes (for subplots)
- unit: label string appended to step annotations (e.g. 'mT', '°C')
- pad: fractional padding around the data extent
- title: custom or suppressed title (set to '' to hide)
- Uses adjustText (if installed) to automatically resolve overlapping labels;
  falls back to simple offset placement otherwise
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pmagpy import pmag

try:
    from adjustText import adjust_text
    _HAS_ADJUSTTEXT = True
except ImportError:
    _HAS_ADJUSTTEXT = False


def plot_zij(fignum, datablock, angle=0, s='', norm=True,
             label_steps=False, label_list=None, label_map=None,
             unit='', ax=None, pad=0.1, title=None):
    """
    Make a Zijderveld (vector component) diagram.

    Parameters
    ----------
    fignum : int
        Matplotlib figure number. Ignored if ax is provided.
    datablock : list
        Nested list of [step, dec, inc, M, type, quality].
    angle : float
        Declination rotation for the horizontal plane (0 puts North on x-axis).
    s : str
        Specimen name for the title.
    norm : bool
        If True, normalize to initial magnetization = 1.
    label_steps : bool
        If True, annotate points with actual treatment step values
        instead of sequential indices.
    label_list : list, optional
        Specific treatment step values to label. If provided, only these
        steps are annotated (implies label_steps=True). If None and
        label_steps is True, all steps are labeled.
    label_map : dict, optional
        Mapping of step values to custom label strings. For example,
        ``{0: 'NRM'}`` labels the 0 mT step as 'NRM' instead of '0 mT'.
        Steps not in the map use the default numeric format.
    unit : str
        Unit string appended to step labels (e.g., 'mT', '°C').
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    pad : float
        Fractional padding added around the data extent (default 0.1 = 10%).
    title : str or None
        Custom title string. If None (default), uses the pmagplotlib-style
        's: NRM = ...' title. Set to '' to suppress the title entirely.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    if label_list is not None:
        label_steps = True
    if label_map is None:
        label_map = {0: 'NRM'}

    if ax is None:
        plt.figure(num=fignum)
        plt.clf()
        ax = plt.gca()

    # Normalization factor
    if norm:
        try:
            fact = 1.0 / datablock[0][3]
        except ZeroDivisionError:
            fact = 1.0
    else:
        fact = 1.0

    # Build DataFrame from datablock
    data = pd.DataFrame(datablock)
    if len(data.columns) == 5:
        data.columns = ['treat', 'dec', 'inc', 'int', 'quality']
    elif len(data.columns) == 6:
        data.columns = ['treat', 'dec', 'inc', 'int', 'type', 'quality']
    elif len(data.columns) == 7:
        data.columns = ['treat', 'dec', 'inc', 'int', 'type', 'quality', 'y']

    data['int'] = data['int'] * fact
    data['dec'] = (data['dec'] - angle) % 360

    gdata = data[data['quality'].str.contains('g')]
    bdata = data[data['quality'].str.contains('b')]

    forVDS = gdata[['dec', 'inc', 'int']].values
    gXYZ = pd.DataFrame(pmag.dir2cart(forVDS), columns=['X', 'Y', 'Z'])

    # Compute axis limits with padding
    all_vals = np.concatenate([gXYZ['X'], gXYZ['Y'], gXYZ['Z']])
    amax = np.max(all_vals)
    amin = np.min(all_vals)
    # Ensure origin is included
    if amin > 0:
        amin = 0
    if amax < 0:
        amax = 0
    span = amax - amin
    if span == 0:
        span = 1.0
    amin -= pad * span
    amax += pad * span

    # Plot bad-quality points
    if len(bdata) > 0:
        bXYZ = pd.DataFrame(
            pmag.dir2cart(bdata[['dec', 'inc', 'int']].values),
            columns=['X', 'Y', 'Z'],
        )
        ax.scatter(bXYZ['X'], bXYZ['Y'], marker='d', c='y', s=30)
        ax.scatter(bXYZ['X'], bXYZ['Z'], marker='d', c='y', s=30)

    # Horizontal projection (filled red circles)
    horiz_pts = ax.plot(gXYZ['X'], gXYZ['Y'], 'ro')
    ax.plot(gXYZ['X'], gXYZ['Y'], 'r-')

    # Vertical projection (open blue squares)
    vert_pts = ax.plot(gXYZ['X'], gXYZ['Z'], 'ws', markeredgecolor='blue')
    ax.plot(gXYZ['X'], gXYZ['Z'], 'b-')

    # Collect all plotted point coordinates for overlap avoidance
    all_x = list(gXYZ['X']) + list(gXYZ['X'])
    all_y = list(gXYZ['Y']) + list(gXYZ['Z'])

    # Annotate points
    treat_vals = gdata['treat'].values
    texts = []
    for k in range(len(gXYZ)):
        val = treat_vals[k]
        # Decide whether to label this point
        if label_list is not None and val not in label_list:
            continue
        if val in label_map:
            lbl = label_map[val]
        elif not label_steps and label_list is None:
            lbl = str(k)
        else:
            if val == int(val):
                lbl = f"{int(val)}"
            else:
                lbl = f"{val:g}"
            if unit:
                lbl += f" {unit}"
        txt = ax.text(gXYZ['X'].iloc[k], gXYZ['Z'].iloc[k], lbl,
                      fontsize=8, ha='left', va='bottom')
        texts.append(txt)

    # Set axis limits (inverted y-axis is the Zij convention)
    ax.axis([amin, amax, amax, amin])

    # Axis cross-hairs spanning the full plot range
    ax.axhline(0, color='k', linewidth=0.8)
    ax.axvline(0, color='k', linewidth=0.8)

    # Labels
    ax.set_xlabel(f"X: rotated to Dec = {angle:7.1f}")
    ax.set_ylabel('Circles: Y; Squares: Z')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    if title is None:
        tstring = s + ': NRM = ' + '%9.2e' % (datablock[0][3])
        ax.set_title(tstring)
    elif title:
        ax.set_title(title)

    # Use adjustText to resolve label overlaps if available
    if _HAS_ADJUSTTEXT and texts:
        adjust_text(texts, ax=ax,
                    only_move={'text': 'xy'},
                    force_text=(0.5, 0.8),
                    force_points=(0.3, 0.5))

    return ax


def overlay_components(ax, comp_A_cart, comp_B_cart, angle=0, norm_factor=1.0,
                       color_A='#D55E00', color_B='#0072B2',
                       label_A='$M_A$', label_B='$M_B$',
                       arrow_kw=None):
    """
    Overlay component vectors on a Zijderveld diagram showing how they
    sum to produce the NRM.

    Draws arrows for both the horizontal and vertical projections:
      Origin → M_B (blue) and M_B → NRM (orange, representing M_A).
    This shows the vector addition NRM = M_A + M_B.

    Must be called on an axes object returned by plot_zij(), using the
    same ``angle`` and normalization.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes from a previous plot_zij() call.
    comp_A_cart : array-like, shape (3,)
        Cartesian vector [X, Y, Z] for Component A (overprint).
    comp_B_cart : array-like, shape (3,)
        Cartesian vector [X, Y, Z] for Component B (primary).
    angle : float
        Same declination rotation used in the plot_zij() call.
    norm_factor : float
        Normalization factor (1 / NRM intensity) — same as used by plot_zij
        when norm=True.  Typically ``1.0 / datablock[0][3]``.
    color_A, color_B : str
        Colors for the A and B component arrows.
    label_A, label_B : str
        Text labels placed at the midpoint of each arrow.
    arrow_kw : dict, optional
        Extra keyword arguments passed to ax.annotate's arrowprops.
    """
    a = np.radians(angle)

    def _project(cart):
        """Apply rotation and normalization, return (x_rot, y_rot, z)."""
        x, y, z = cart * norm_factor
        x_rot = x * np.cos(a) + y * np.sin(a)
        y_rot = -x * np.sin(a) + y * np.cos(a)
        return x_rot, y_rot, z

    A = _project(np.asarray(comp_A_cart))
    B = _project(np.asarray(comp_B_cart))
    # NRM = A + B in projected space (projection is linear)
    NRM = (A[0] + B[0], A[1] + B[1], A[2] + B[2])

    alpha = 0.7
    shaft_width = 0.025  # as fraction of axis extent
    head_width = 0.072
    head_length = 0.042
    if arrow_kw:
        alpha = arrow_kw.get('alpha', alpha)
        shaft_width = arrow_kw.get('width', shaft_width)
        head_width = arrow_kw.get('head_width', head_width)
        head_length = arrow_kw.get('head_length', head_length)

    for y_idx in ('horiz', 'vert'):
        idx = 1 if y_idx == 'horiz' else 2

        b_tip = (B[0], B[idx])
        nrm_tip = (NRM[0], NRM[idx])

        # Arrow: Origin → B
        ax.arrow(0, 0, b_tip[0], b_tip[1],
                 width=shaft_width, head_width=head_width,
                 head_length=head_length, fc=color_B, ec='none',
                 alpha=alpha, length_includes_head=True, zorder=10)
        # Arrow: B tip → NRM (this is the A component)
        dx = nrm_tip[0] - b_tip[0]
        dy = nrm_tip[1] - b_tip[1]
        ax.arrow(b_tip[0], b_tip[1], dx, dy,
                 width=shaft_width, head_width=head_width,
                 head_length=head_length, fc=color_A, ec='none',
                 alpha=alpha, length_includes_head=True, zorder=10)

    # Labels offset from the midpoint of each arrow (horizontal projection)
    # The Zij y-axis is inverted, so positive offset points move visually up
    # B label: up and to the left
    b_mid_h = (B[0] / 2, B[1] / 2)
    ax.annotate(label_B, xy=b_mid_h, xytext=(-18, 12),
                textcoords='offset points', fontsize=11, fontweight='bold',
                color=color_B, ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1),
                zorder=11)
    # A label: up and to the right
    a_mid_h = ((B[0] + NRM[0]) / 2, (B[1] + NRM[1]) / 2)
    ax.annotate(label_A, xy=a_mid_h, xytext=(18, 12),
                textcoords='offset points', fontsize=11, fontweight='bold',
                color=color_A, ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1),
                zorder=11)
