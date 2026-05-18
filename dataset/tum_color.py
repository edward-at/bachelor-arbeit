# TUM Corporate Design Colors
# Source: https://gist.github.com/lnksz/51e3566af2df5c7aa678cd4dfc8305f7
# Usage: from tum_colors import TUM

TUM = {
    # ── Main colors ───────────────────────────────────────────────────────────
    'blue':         '#0065bd',   # TUMBlue — primary color, use for main bars/lines
    'black':        '#000000',
    'white':        '#ffffff',

    # ── Secondary blues ───────────────────────────────────────────────────────
    'dark_blue':    '#005293',   # DarkBlue
    'light_blue':   '#64a0c8',   # LightBlue — secondary elements
    'lighter_blue': '#98c6ea',   # LighterBlue — background/fill

    # ── Emphasis ──────────────────────────────────────────────────────────────
    'orange':       '#e37222',   # Orange — contrast/highlight, second category
    'green':        '#a2ad00',   # Green — third category
    'gray':         '#999999',   # Gray — neutral/background bars

    # ── Extended palette (2022 web colors) ───────────────────────────────────
    'tum_blue':     '#3070B3',   # tum-blue-brand
    'yellow':       '#FED702',   # tum-yellow
    'pink':         '#B55CA5',   # tum-pink
    'red':          '#EA7237',   # tum-red
    'light_gray':   '#dad7cb',   # LightGray
}

# ── Suggested color assignments per chart type ────────────────────────────────

# For Real / Synthetic / Mixed (charts 01a-01d, 02, 06)
REAL_SYNTH_MIX = [TUM['blue'], TUM['orange'], TUM['green']]

# For single-category bar/line charts
SINGLE = TUM['blue']

# For pie charts (up to 4 slices)
PIE_COLORS = [TUM['blue'], TUM['orange'], TUM['green'], TUM['light_blue']]

# For background bars (e.g. dataset count in chart 03)
BACKGROUND_BAR = TUM['light_gray']

# For heatmaps
HEATMAP_CMAP = 'Blues'  # works well with TUM blue theme