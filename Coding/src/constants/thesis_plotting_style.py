IN_FIGURE_TEXT_SIZE_MULTIPLIER: float = 0.8

PALETTE = {
    "Professional": "#2C6E8F",  # blue
    "Personal":     "#4C956C",  # muted green
    "Neutral":      "#8D99AE",
}

COLORS = {
    "background":     "#FFFFFF",
    "text":           "#22333B",  # deep blue-charcoal, pairs with the blue/green
    "secondary_text": "#5A6B73",
    "grid":           "#E7EBED",  # very light cool grey
    "axis":           "#C7D0D5",
    "accent":         "#E9A23B",  # warm amber — peak-bar highlight, pops against blue & green
    "highlight":      "#A23B47",  # muted brick red — reference / quantile lines
}


FONT: str = "DejaVu Sans"

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