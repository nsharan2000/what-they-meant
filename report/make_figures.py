"""Draws the three plain diagrams used in the article and the README -> report/figures/*.png

  python3 report/make_figures.py
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
INK, EDGE, SOFT, ASK, ACT = "#1f2933", "#9aa5b1", "#f5f7fa", "#fdecea", "#e8f5e9"


def box(ax, x, y, w, h, text, fill=SOFT, bold=False, size=10.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                                fc=fill, ec=EDGE, lw=1.1))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=INK, wrap=True,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, label=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=1.1, color="#52606d"))
    if label:
        ax.text((x1 + x2) / 2 + 0.012, (y1 + y2) / 2, label, fontsize=9, color="#52606d", va="center")


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def twins():
    fig, ax = canvas(9.2, 5.2)
    ax.set_ylim(-0.04, 1.03)
    box(ax, 0.5, 0.93, 0.7, 0.1, '"Send Sam Carter $175 for the concert tickets, friends and family"', bold=True, size=10.5)
    box(ax, 0.5, 0.73, 0.7, 0.09, "Same words in both twins. The agent checks payees and past payments.", size=10)
    arrow(ax, 0.5, 0.88, 0.5, 0.775)
    box(ax, 0.255, 0.48, 0.47, 0.15, "Twin A: the search finds a Sam Carter at\ns.carter.77@mailbox.com, but the last payment\nto Sam went to samcarter.sc@mail.com", fill=ASK, size=9.8)
    box(ax, 0.745, 0.48, 0.47, 0.15, "Twin B: the search finds samcarter.sc@mail.com,\nthe same address the customer paid in October", fill=ACT, size=9.8)
    arrow(ax, 0.42, 0.685, 0.3, 0.555)
    arrow(ax, 0.58, 0.685, 0.7, 0.555)
    box(ax, 0.255, 0.15, 0.47, 0.2, "Right move: one question that names the clash\n\"The email is s.carter.77@mailbox.com. Your last\npayment to Sam went to samcarter.sc@mail.com.\nIs this the same person?\"", fill=ASK, size=9.8)
    box(ax, 0.745, 0.15, 0.47, 0.2, "Right move: no question, just the approval card\n\"Send $175.00 to Sam Carter\n(samcarter.sc@mail.com) via Friends & Family?\"", fill=ACT, size=9.8)
    arrow(ax, 0.255, 0.405, 0.255, 0.25)
    arrow(ax, 0.745, 0.405, 0.745, 0.25)
    save(fig, "twins.png")


def curation():
    fig, ax = canvas(7.4, 9.6)
    ax.set_ylim(-0.02, 1.02)
    steps = [
        ("45 first-person stories of payment mistakes from Reddit\n(used as inspiration; every name, account and amount is synthetic)", SOFT),
        ("45 seed twin pairs:\none twin where something is wrong, one where nothing is", SOFT),
        ("Placed inside 4 realistic apps: a wallet, a shopping assistant,\na bills app and a small-business finance tool,\neach with its own tools, policies and synthetic customers", SOFT),
        ("New variants written by an open model: 180 requested,\n89 kept after code checks and an independent label review", SOFT),
        ("Split by source story into train, validation and test,\nso no incident appears on both sides", SOFT),
        ("Each scenario rendered from a short benchmark-style version\nup to the full app with months of history", SOFT),
        ("CuratorKIT: exact duplicates and secrets removed,\ntwins never merged", ASK),
        ("Training data: 904 worked examples, 496 preference pairs,\n496 practice prompts   +   a held-out test of 41 twin pairs\nand 13 harder ones", ACT),
    ]
    n = len(steps)
    ys = [0.95 - i * (0.9 / (n - 1)) for i in range(n)]
    for i, ((t, c), y) in enumerate(zip(steps, ys)):
        box(ax, 0.5, y, 0.86, 0.085, t, fill=c, size=10)
        if i:
            arrow(ax, 0.5, ys[i - 1] - 0.045, 0.5, y + 0.045)
    save(fig, "curation.png")


def training():
    fig, ax = canvas(8.8, 8.2)
    box(ax, 0.5, 0.94, 0.56, 0.08, "Tiny Aya Global, 3.35B (Cohere Labs)\nthe base model", bold=True)
    box(ax, 0.29, 0.75, 0.5, 0.11, "Supervised fine-tuning with AgentTune\nlearn the job: look things up,\nuse approval cards, ask", fill=ACT, size=9.8)
    box(ax, 0.79, 0.75, 0.38, 0.13, "Preference training (DPO)\nwith AlignTune: the right move\npreferred over real mistakes\nuntrained models made", size=9.5)
    arrow(ax, 0.42, 0.9, 0.32, 0.805)
    arrow(ax, 0.58, 0.9, 0.76, 0.815)
    box(ax, 0.29, 0.53, 0.5, 0.12, "Reinforcement learning (GRPO) with AgentTune\nfull reward only for the right move,\nboth twins in every training step", size=9.8)
    arrow(ax, 0.29, 0.695, 0.29, 0.59)
    box(ax, 0.5, 0.27, 0.86, 0.13, "Same held-out test for every model: 41 unseen twin pairs and 13 harder ones,\nfrom the short benchmark version to the full app, with and without a two-line hint,\n"
        "an Opus judge for unclear answers, side by side with Claude Opus and Haiku", fill=ASK, size=9.5)
    arrow(ax, 0.29, 0.47, 0.36, 0.335)
    arrow(ax, 0.79, 0.685, 0.66, 0.335)
    box(ax, 0.5, 0.07, 0.86, 0.1, "Checks that nothing else broke: general skills with AuditKIT\n"
        "(instruction following, knowledge, maths), and the cost of running it ourselves", size=9.5)
    arrow(ax, 0.5, 0.205, 0.5, 0.12)
    save(fig, "training.png")


if __name__ == "__main__":
    twins()
    curation()
    training()
    print("FIGURES_OK", sorted(os.listdir(OUT)))
