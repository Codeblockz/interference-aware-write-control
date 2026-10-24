"""Draw the system block diagram (paper Fig. 1). Usage: python experiments/block_diagram.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(15.2, 5.4), dpi=200)
ax.set_xlim(0, 15.2); ax.set_ylim(0, 5.4); ax.axis("off")
INK, ACC, SOFT = "#222", "#1f5fa8", "#f3f6fb"

def box(x, y, w, h, title, body="", fc="white", ec=INK, lw=1.2, fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=lw))
    if body:
        ax.text(x + w/2, y + h - 0.28, title, ha="center", va="top", fontsize=fs, weight="bold", color=INK)
        ax.text(x + w/2, y + 0.18, body, ha="center", va="bottom", fontsize=8.6, color=INK, linespacing=1.35)
    else:
        ax.text(x + w/2, y + h/2, title, ha="center", va="center", fontsize=fs, weight="bold", color=INK)

def arrow(p, q, label="", color=INK, style="-|>", lw=1.3, dx=0, dy=0.12, fs=8.6, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if label:
        ax.text((p[0]+q[0])/2 + dx, (p[1]+q[1])/2 + dy, label, ha="center", va="bottom", fontsize=fs, color=color)

# --- top row: backbone and projections -------------------------------------
box(0.2, 3.5, 1.5, 1.1, "Tokens\n$x_1 \\ldots x_T$", fs=10)
box(2.3, 3.3, 2.6, 1.5, "GLA backbone",
    "$L$ pre-norm blocks\ngated linear attention + MLP\nfixed-size recurrent state")
arrow((1.7, 4.05), (2.3, 4.05))
box(5.6, 3.3, 2.4, 1.5, "Projections",
    "$q_t = W_q h_t$,  $v_t = W_v h_t$\n$k_t = W_k\\, h_{t-1}$  (key from\nthe preceding position)")
arrow((4.9, 4.05), (5.6, 4.05), "$h_t, h_{t-1}$", dy=0.08, fs=8.2)

# --- middle: memory and writer --------------------------------------------
box(5.6, 0.55, 2.4, 2.1, "Slot memory",
    "$S$ slots: $(k_j, v_j)$ + age $a_j$,\nusage $c_j$, occupancy $f_j$\n\nread (Eq. 1): $r_t = \\sum_j \\alpha_{t,j} v_j$\nwrite (Eq. 2): convex update")
box(9.2, 0.55, 5.6, 2.1, "Writer  (one of five)",
    "Controller: per-slot features (Eq. 3): interference,\nnovelty, age, usage, occupancy "
    "$\\to$ MLP $\\to$ gates $g_{t,j}$\n(Eq. 4: hard forward, straight-through backward)\n"
    "H2 ablation: same controller, interference features zeroed\n"
    "Baselines: recency, residual cache, position activation")
# q to memory (read)
arrow((6.3, 3.3), (6.3, 2.65), "$q_t$  (read first)", dx=-0.55, dy=-0.05)
# (k,v) to writer
arrow((7.4, 3.3), (12.0, 2.65), "$(k_t, v_t)$", dx=0.2, dy=0.12, rad=-0.22)
# memory state to writer, gates back
arrow((8.0, 1.95), (9.2, 1.95), "slot state", dy=0.06, fs=8.4)
arrow((9.2, 1.05), (8.0, 1.05), "gates $g_{t,j}$", dy=-0.4)
# (k,v) into memory for the write
arrow((7.0, 3.3), (7.0, 2.65), "$(k_t, v_t)$ write", dx=0.62, dy=-0.05)

# --- bottom-left: answer head ----------------------------------------------
box(2.3, 0.55, 2.6, 1.4, "Answer head",
    "2-layer MLP on $[h_t ; r_t]$\nlogits over vocabulary")
arrow((5.6, 1.25), (4.9, 1.25), "$r_t$", dy=0.06)
arrow((3.6, 3.3), (3.6, 1.95), "$h_t$", dx=-0.25, dy=-0.1)
box(0.2, 0.7, 1.5, 1.1, "Loss at answer\npositions only", fs=9.5)
arrow((2.3, 1.25), (1.7, 1.25))

# shared / varied annotation
ax.text(7.6, 0.08, "Shared by all writers: backbone, projections, memory, answer head, data, state budget.  "
                  "Only the writer block differs.", ha="center", va="bottom", fontsize=8.8, color=ACC, style="italic")
ax.add_patch(FancyBboxPatch((9.05, 0.4), 5.9, 2.4, boxstyle="round,pad=0.02,rounding_size=0.15",
                            fc="none", ec=ACC, lw=1.2, ls="--"))
ax.text(14.95, 2.85, "varies", ha="right", va="bottom", fontsize=8.8, color=ACC, style="italic")

fig.savefig("paper/figures/block-diagram.png", bbox_inches="tight", facecolor="white")
print("wrote paper/figures/block-diagram.png")
