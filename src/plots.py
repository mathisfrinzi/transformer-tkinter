# Petits outils pour les graphiques et les widgets
import numpy as np
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (active la projection 3d)

# couleurs du dark mode
BG = "#1e1e1e"
BG2 = "#2a2a2a"
BG3 = "#3a3a3a"
FG = "#ffffff"
SEL = "#4a6fa5"

plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG2,
    "savefig.facecolor": BG,
    "font.size": 8,
    "axes.titlesize": 9,
})


def pca3(X):
    # PCA faite à la main avec la SVD, on garde les 3 premières composantes
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(3, Vt.shape[0])
    P = Xc @ Vt[:k].T
    if k < 3:
        P = np.hstack([P, np.zeros((P.shape[0], 3 - k))])
    return P


def style3d(ax):
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((0.16, 0.16, 0.16, 1.0))
    ax.set_facecolor(BG2)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")


def heat(fig, ax, M, title, ylabels=None, xlabels=None, cmap="viridis", annotate=False):
    M = np.array(M, dtype=float)
    M[M < -1e8] = np.nan  # positions masquées en gris
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#555555")
    im = ax.imshow(M, aspect="auto", cmap=cm)
    ax.set_title(title)
    if ylabels is not None:
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
    if xlabels is not None:
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=60, ha="right")
    if annotate and M.size <= 150:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if M[i, j] < np.nanmax(M) * 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def make_canvas(parent):
    fig = Figure(figsize=(9, 6))
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    return fig, canvas


def finish(fig, canvas):
    try:
        fig.tight_layout()
    except Exception:
        pass
    canvas.draw_idle()


def dark_text(parent, height=10, width=40):
    return tk.Text(parent, height=height, width=width, bg=BG2, fg=FG, insertbackground=FG,
                   relief="flat", font="TkFixedFont", wrap="word", highlightthickness=0)


def put_text(widget, s):
    widget.delete("1.0", "end")
    widget.insert("1.0", s)


def make_tree(parent, cols, widths, height=8):
    tree = ttk.Treeview(parent, columns=cols, show="headings", height=height)
    for c, w in zip(cols, widths):
        tree.heading(c, text=c)
        tree.column(c, width=w, anchor="center")
    return tree


def make_combo(parent, label, on_select):
    ttk.Label(parent, text=label).pack(anchor="w", pady=(6, 0))
    cb = ttk.Combobox(parent, state="readonly")
    cb.pack(fill="x")
    cb.bind("<<ComboboxSelected>>", lambda e: on_select())
    return cb
