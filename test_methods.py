import matplotlib.pyplot as plt
import numpy as np
import torch

from gptopt.optim.mach_polar import MachPolar
from gptopt.optim.muon import svd_exact_polar, zeropower_via_newtonschulz5
from gptopt.optim.polar_express import PolarExpress

# TODO: Make this and the machpolar file so that we can choose the one we want to use, so maybe add another file?
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 1.5
plt.rc("text", usetex=True)

# ── Load gradient matrix ──────────────────────────────────────────────────────
grads = torch.load("h3_c_attn_grads.pt", weights_only=True)  # shape: (3, 768, 2304)
G = grads[2].float()  # take first step, shape (768, 2304)

# ── Exact polar factor via SVD ────────────────────────────────────────────────
U_exact = svd_exact_polar(G, 1)  # shape (768, 2304)
norm_exact = U_exact.norm(p="fro").item()

# ── Sweep over number of steps ────────────────────────────────────────────────
max_steps = 10
steps_range = list(range(1, max_steps + 1))
mults_per_step = 3  # each iteration costs 3 matrix multiplications

errors_zeropower = []
errors_polarexpress = []
errors_machpolar = []

for ns in steps_range:
    print(ns)
    # ZeroPower (Newton-Schulz)
    U_zp = zeropower_via_newtonschulz5(G, steps=ns)
    errors_zeropower.append((U_zp - U_exact).norm(p="fro").item() / norm_exact)

    # PolarExpress
    U_pe = PolarExpress(G, steps=ns)
    errors_polarexpress.append((U_pe - U_exact).norm(p="fro").item() / norm_exact)

    # MachPolar
    if ns >= 4:
        continue
    U_mp = MachPolar(G, steps=ns)
    errors_machpolar.append((U_mp - U_exact).norm(p="fro").item() / norm_exact)

x = [s * mults_per_step for s in steps_range]  # matrix multiplications on x-axis
xMach = [5, 10, 15]  # matrix multiplications on x-axis

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))

ax.plot(x, errors_zeropower, label=r"Jordan", color="#FF6B35", linewidth=2)
ax.plot(x, errors_polarexpress, label=r"PolarExpress", color="k", linewidth=2)
ax.plot(xMach, errors_machpolar, label=r"MachPolar", color="#8A2BE2", linewidth=2)

ax.set_yscale("log")
ax.set_xlabel(r"Matrix Multiplications", fontsize=12)
ax.set_ylabel(r"Frobenius Error $\|U - U_{\mathrm{exact}}\|_F$", fontsize=12)
ax.legend(fontsize=10)
ax.grid(axis="both", lw=0.2, ls="--", zorder=0)

fig.subplots_adjust(top=0.97, bottom=0.14, left=0.15, right=0.97)
fig.savefig("figures/polar_convergenceEq.pdf", format="pdf", bbox_inches="tight")
print("Saved figures/polar_convergenceEq.pdf")
