"""Draw the system block diagram (paper Fig. 1). Usage: python experiments/block_diagram.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

W, H = 17.0, 6.6
fig, ax = plt.subplots(figsize=(W, H), dpi=200)
ax.set_xlim(0, W); ax.set_ylim(-0.3, H); ax.axis("off")
INK, ACC = "#222", "#1f5fa8"
FT, FB, FL = 12.5, 10.3, 10   # title, body, edge-label font sizes

def box(x, y, w, h, title, body="", fs=FT):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="white", ec=INK, lw=1.2))
    if body:
        ax.text(x + w/2, y + h - 0.22, title, ha="center", va="top", fontsize=fs, weight="bold", color=INK)
        ax.text(x + w/2, y + 0.2, body, ha="center", va="bottom", fontsize=FB, color=INK, linespacing=1.3)
    else:
        ax.text(x + w/2, y + h/2, title, ha="center", va="center", fontsize=fs, weight="bold", color=INK)

def arrow(p, q, label="", dx=0, dy=0.1, rad=0.0, ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, lw=1.3, color=INK,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if label:
        ax.text((p[0]+q[0])/2 + dx, (p[1]+q[1])/2 + dy, label, ha=ha, va="bottom", fontsize=FL, color=INK)

# top row -------------------------------------------------------------------
box(0.2, 4.1, 1.7, 1.3, "Tokens\n$x_1 \\ldots x_T$", fs=12)
box(2.6, 3.8, 3.2, 1.9, "GLA backbone",
    "$L$ pre-norm blocks\ngated linear attention + MLP\nfixed-size recurrent state")
arrow((1.9, 4.75), (2.6, 4.75))
box(6.6, 3.8, 3.0, 1.9, "Projections",
    "$q_t = W_q h_t$,  $v_t = W_v h_t$\n$k_t = W_k\\, h_{t-1}$\n(key from previous position)")
arrow((5.8, 4.75), (6.6, 4.75), "$h_t, h_{t-1}$", dy=0.08)

# middle row ----------------------------------------------------------------
box(6.6, 0.5, 3.0, 2.5, "Slot memory",
    "$S$ slots: $(k_j, v_j)$ + age $a_j$,\nusage $c_j$, occupancy $f_j$\n\nread (Eq. 1): $r_t = \\sum_j \\alpha_{t,j} v_j$\nwrite (Eq. 2): convex update")
box(10.6, 0.5, 6.2, 2.5, "Writer  (one of five)",
    "Controller: per-slot features (Eq. 3): interference,\nnovelty, age, usage, occupancy $\\to$ MLP $\\to$ gates $g_{t,j}$\n"
    "(Eq. 4: hard forward, straight-through backward)\n"
    "H2 ablation: same controller, interference features zeroed\n"
    "Baselines: recency, residual cache, position activation")
arrow((7.4, 3.8), (7.4, 3.0), "$q_t$ (read first)", dx=-0.1, dy=-0.05, ha="right")
arrow((8.6, 3.8), (8.6, 3.0), "$(k_t, v_t)$ write", dx=0.1, dy=-0.05, ha="left")
arrow((9.2, 3.8), (13.7, 3.0), "$(k_t, v_t)$", dx=0.3, dy=0.62, rad=-0.22)
arrow((9.6, 2.2), (10.6, 2.2), "slot state", dy=0.06)
arrow((10.6, 1.2), (9.6, 1.2), "gates $g_{t,j}$", dy=-0.5)

# bottom-left ---------------------------------------------------------------
box(2.6, 0.7, 3.2, 1.7, "Answer head", "2-layer MLP on $[h_t ; r_t]$\nlogits over vocabulary")
arrow((6.6, 1.55), (5.8, 1.55), "$r_t$", dy=0.06)
arrow((4.2, 3.8), (4.2, 2.4), "$h_t$", dx=-0.1, dy=-0.1, ha="right")
box(0.2, 0.9, 1.7, 1.3, "Loss at answer\npositions only", fs=11)
arrow((2.6, 1.55), (1.9, 1.55))

# shared / varies -----------------------------------------------------------
ax.add_patch(FancyBboxPatch((10.45, 0.35), 6.5, 2.8, boxstyle="round,pad=0.02,rounding_size=0.15",
                            fc="none", ec=ACC, lw=1.2, ls="--"))
ax.text(16.95, 3.2, "varies", ha="right", va="bottom", fontsize=FL, color=ACC, style="italic")
ax.text(W/2, -0.25, "Shared by all writers: backbone, projections, memory, answer head, data, state budget.  "
                    "Only the writer block differs.", ha="center", va="bottom", fontsize=FL, color=ACC, style="italic")

fig.savefig("paper/figures/block-diagram.png", bbox_inches="tight", facecolor="white")
print("wrote paper/figures/block-diagram.png")
