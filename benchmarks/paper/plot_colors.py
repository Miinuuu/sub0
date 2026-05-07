"""Shared color palette for paper figures.

Wong / Okabe–Ito colorblind-safe palette (Nature Methods 2011, "Points of
View: Color blindness"). This is the de-facto standard for top-tier
journals because every pair remains distinguishable under deuteranopia,
protanopia, and grayscale conversion. CDA is mapped to vermillion so it
stays the hero color across every figure; the remaining methods are
spread over the palette's blue / orange / green / purple / sky hues for
separability when multiple baselines coexist (Fig 3).
"""

METHOD_COLORS = {
    "FA2":         "#0072B2",  # blue
    "DEQUANT_FA2": "#E69F00",  # orange
    "INT4_RTN":    "#009E73",  # bluish green
    "KIVI_K2V2":   "#CC79A7",  # reddish purple
    "COMMVQ":      "#56B4E9",  # sky blue
    "CDA":         "#D55E00",  # vermillion (hero)
}

REFERENCE_COLORS = {
    "PRIMARY":   "#6B7280",
    "SECONDARY": "#9CA3AF",
}
