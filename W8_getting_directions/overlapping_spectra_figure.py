"""
Generate a summary figure showing the effect of overlapping coercivity spectra
on Zijderveld diagrams, after Dunlop (1979).

Three rows show progressively increasing overlap between two log-Gaussian
coercivity spectra (J_A and J_B), with equal-intensity components (50/50).
Left column: coercivity spectra. Right column: Zijderveld diagrams.

Usage:
    mamba activate pmagpy-dev
    python overlapping_spectra_figure.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from pmagpy import pmag
from plot_zij import plot_zij, overlay_components

# ---------------------------------------------------------------------------
# Component directions (same as tutorial notebook)
# ---------------------------------------------------------------------------
comp_A_dir = [0.0, 60.0]     # overprint: North, steeply down
comp_B_dir = [270.0, 20.0]   # ancient: WNW, shallowly down

# Equal intensities for illustrative clarity
intensity = 0.50
A_cart = np.array(pmag.dir2cart([comp_A_dir[0], comp_A_dir[1], intensity]))
B_cart = np.array(pmag.dir2cart([comp_B_dir[0], comp_B_dir[1], intensity]))
nrm_dec = pmag.cart2dir(A_cart + B_cart)[0]

# ---------------------------------------------------------------------------
# AF demagnetization steps and smooth curve for spectra
# ---------------------------------------------------------------------------
af_steps = np.array([0, 2, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                     60, 70, 80, 90, 100, 120, 140, 160, 180, 200])
af_fine = np.linspace(0.1, 200, 500)

# ---------------------------------------------------------------------------
# Three cases with progressively increasing overlap
# ---------------------------------------------------------------------------
cases = [
    {'median_A': 12, 'dp_A': 0.4, 'median_B': 100, 'dp_B': 0.3,
     'row_title': 'Non-overlapping spectra'},
    {'median_A': 20, 'dp_A': 0.5, 'median_B': 70,  'dp_B': 0.4,
     'row_title': 'Small overlap'},
    {'median_A': 30, 'dp_A': 0.6, 'median_B': 50,  'dp_B': 0.5,
     'row_title': 'Large overlap'},
]


def generate_demag_data(steps, comp_A_cart, comp_B_cart,
                        median_A, dp_A, median_B, dp_B):
    """Generate synthetic demagnetization data with log-Gaussian spectra."""
    frac_A = 1.0 - lognorm.cdf(steps, s=dp_A, scale=median_A)
    frac_B = 1.0 - lognorm.cdf(steps, s=dp_B, scale=median_B)
    demag_data = []
    for i, step in enumerate(steps):
        total_cart = frac_A[i] * comp_A_cart + frac_B[i] * comp_B_cart
        total_dir = pmag.cart2dir(total_cart)
        dec, inc, mag = total_dir[0], total_dir[1], total_dir[2]
        if np.isnan(dec):
            dec = 0.0
        if np.isnan(inc):
            inc = 0.0
        demag_data.append([step, dec, inc, mag, 'DA', 'g'])
    return demag_data


# ---------------------------------------------------------------------------
# Build the figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(10, 9))

for row, case in enumerate(cases):
    ax_spec = axes[row, 0]
    ax_zij = axes[row, 1]

    data = generate_demag_data(
        af_steps, A_cart, B_cart,
        case['median_A'], case['dp_A'], case['median_B'], case['dp_B'])

    # --- Left: coercivity spectra ---
    spec_A = intensity * lognorm.pdf(af_fine, s=case['dp_A'], scale=case['median_A'])
    spec_B = intensity * lognorm.pdf(af_fine, s=case['dp_B'], scale=case['median_B'])
    ax_spec.fill_between(af_fine, spec_A, alpha=0.3, color='#D55E00')
    ax_spec.fill_between(af_fine, spec_B, alpha=0.3, color='#0072B2')
    ax_spec.plot(af_fine, spec_A, color='#D55E00', linewidth=1.5)
    ax_spec.plot(af_fine, spec_B, color='#0072B2', linewidth=1.5)

    # Labels inside the curves at the peak
    for median, dp, label, color in [
        (case['median_A'], case['dp_A'], '$M_A$', '#D55E00'),
        (case['median_B'], case['dp_B'], '$M_B$', '#0072B2'),
    ]:
        peak_y = intensity * lognorm.pdf(median, s=dp, scale=median)
        ax_spec.text(median, peak_y * 0.45, label, ha='center', va='center',
                     fontsize=13, fontweight='bold', color=color)

    ax_spec.set_ylabel('dM/dB', fontsize=11)
    ax_spec.set_xlim(0.1, 200)
    ax_spec.set_ylim(bottom=0)
    ax_spec.set_title(case['row_title'], fontsize=12, loc='left')
    if row == 2:
        ax_spec.set_xlabel('AF level (mT)', fontsize=11)
    else:
        ax_spec.set_xticklabels([])

    # --- Right: Zijderveld diagram ---
    plot_zij(None, data, angle=nrm_dec, s='',
             label_list=[0, 15, 30, 60], unit='mT',
             ax=ax_zij, title='')

    # Show vector addition on all cases
    nrm_intensity = data[0][3]
    overlay_components(ax_zij, A_cart, B_cart,
                       angle=nrm_dec, norm_factor=1.0 / nrm_intensity)

# Force all Zij panels to share the same axis limits (use the first panel's range)
zij_lims = axes[0, 1].axis()
for row in range(1, 3):
    axes[row, 1].axis(zij_lims)

fig.tight_layout()
fig.savefig('overlapping_spectra_figure.png', dpi=200, bbox_inches='tight')
fig.savefig('overlapping_spectra_figure.pdf', bbox_inches='tight')
plt.show()
print("Saved: overlapping_spectra_figure.png, overlapping_spectra_figure.pdf")
