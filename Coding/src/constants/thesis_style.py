PALETTE = {
    "Professional": "#1F4E79",
    "Personal": "#2A9D8F",
}

COLORS = {
    "background": "#FFFFFF",
    "text": "#1F2933",
    "secondary_text": "#52616B",
    "grid": "#E5E7EB",
    "axis": "#D1D5DB",
    "accent": "#D99A2B",
    "highlight": "#9B2C2C",
}

FONT = "DejaVu Sans"


SEABORN_THEME = {
    "style": "whitegrid",
    "context": "paper",
    "font": FONT,
    "rc": {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.facecolor": COLORS["background"],
        "figure.facecolor": COLORS["background"],
        "axes.edgecolor": COLORS["axis"],
        "axes.labelcolor": COLORS["text"],
        "text.color": COLORS["text"],
        "xtick.color": COLORS["secondary_text"],
        "ytick.color": COLORS["secondary_text"],
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "legend.frameon": False,
    },
}