"""Assemble side-by-side comparisons: paper panel (left) vs our reproduction (right).

Reads whatever data/*.npz files exist and writes figures/compare_*.png.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANELS = os.path.join(ROOT, "figures", "paper_panels")
OUT = os.path.join(ROOT, "figures")

# colours matching the paper's Delta T/T0 legend
DT_COLORS = {
    0.00: "#1f9e8b", 0.25: "#3b6fb6", 0.50: "#e79a3c", 0.75: "#6bbf3a",
    1.00: "#e2412a", 1.25: "#7a5aa8", 1.50: "#b5651d",
}
def dtcol(dT):
    return DT_COLORS.get(round(float(dT), 2), "#333333")


def _paper(ax, name, title="Paper (Kim et al. 2021)"):
    p = os.path.join(PANELS, name)
    if os.path.exists(p):
        ax.imshow(mpimg.imread(p))
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def fig3a():
    f = os.path.join(ROOT, "data", "fig3.npz")
    if not os.path.exists(f):
        return
    d = np.load(f)
    dT_list, tlag, msd, alpha = d["dT_list"], d["tlag"], d["msd"], d["alpha"]

    fig = plt.figure(figsize=(12, 5))
    axp = fig.add_subplot(1, 2, 1)
    _paper(axp, "crop_fig3a.png")
    ax = fig.add_subplot(1, 2, 2)
    for i, dT in enumerate(dT_list):
        ax.loglog(tlag[i], msd[i], "-", color=dtcol(dT), lw=2, label=f"{dT:.2f}")
    # guide lines
    ax.loglog([3e-3, 3e-1], [3e-6, 3e-4], "k:", lw=1)          # slope-1 (early)
    ax.text(0.02, 1.5, "Diffusion", fontsize=9)
    ax.plot([15, 100], [3e-3, 3e-3], "k--", lw=1)
    ax.text(30, 4e-3, "Caged", fontsize=9)
    ax.set_xlabel(r"$t/\tau_T$"); ax.set_ylabel(r"MSD$(t)/L_0^2$")
    ax.set_ylim(1e-6, 1e1); ax.set_xlim(3e-3, 1.2e2)
    ax.legend(title=r"$\Delta T/T_0$", fontsize=8, ncol=2, loc="lower right")
    ax.set_title("This work (Python)", fontsize=11)
    # inset alpha vs dT
    axi = ax.inset_axes([0.12, 0.62, 0.34, 0.34])
    axi.plot(dT_list, alpha, "o-", color="k", ms=5)
    for dT, a in zip(dT_list, alpha):
        axi.plot(dT, a, "o", color=dtcol(dT), ms=6)
    axi.axhline(1.0, color="r", lw=1)
    axi.set_xlabel(r"$\Delta T/T_0$", fontsize=8); axi.set_ylabel(r"$\alpha$", fontsize=8)
    axi.set_ylim(0, 1.15); axi.tick_params(labelsize=7)
    fig.suptitle("Fig. 3a  |  Mean squared displacement vs time "
                 r"($\rho=1,\ W/T_0=1$)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT, "compare_fig3a_msd.png"), dpi=120)
    plt.close(fig)
    print("wrote compare_fig3a_msd.png")


def _fit_jamming(phis, z, zc=4.0):
    """Robust fit z-zc = z0 sqrt(phi-phic) + z1 (phi-phic), z0,z1>=0."""
    from scipy.optimize import curve_fit
    mask = z > zc + 0.02
    pj, zj = phis[mask], z[mask]

    def model(phi, phic, z0, z1):
        dd = np.clip(phi - phic, 0, None)
        return zc + z0 * np.sqrt(dd) + z1 * dd
    best, berr = None, np.inf
    for phic0 in np.linspace(0.80, 0.86, 13):
        try:
            popt, _ = curve_fit(model, pj, zj, p0=[phic0, 1.45, 10.45],
                                bounds=([0.78, 0, 0], [0.88, 20, 40]), maxfev=40000)
            err = np.mean((model(pj, *popt) - zj) ** 2)
            if err < berr:
                berr, best = err, popt
        except Exception:
            pass
    return best if best is not None else [0.83, 1.45, 10.45]


def fig2f():
    f = os.path.join(ROOT, "data", "fig2f.npz")
    if not os.path.exists(f):
        return
    d = np.load(f)
    phis, z, zsd, zc = d["phis"], d["zmean"], d["zstd"], float(d["zc"])
    phic, z0, z1 = _fit_jamming(phis, z, zc)

    fig = plt.figure(figsize=(11, 4.6))
    axp = fig.add_subplot(1, 2, 1)
    _paper(axp, "crop_fig2f.png")
    ax = fig.add_subplot(1, 2, 2)
    ax.errorbar(phis, z, yerr=zsd, fmt="s", color="#c0392b", ms=5,
                capsize=2, label="simulation")
    pp = np.linspace(phic, 1.0, 100)
    ax.plot(pp, zc + z0 * np.sqrt(pp - phic) + z1 * (pp - phic), "k--",
            label=fr"$z-z_c=z_0\sqrt{{\phi-\phi_c}}+z_1(\phi-\phi_c)$")
    ax.axhline(zc, color="gray", lw=0.8, ls=":")
    ax.annotate(fr"$\phi_c\simeq{phic:.2f}$", (phic, zc), (phic+0.01, 4.2),
                fontsize=10, arrowprops=dict(arrowstyle="->"))
    ax.set_xlabel(r"$\phi$"); ax.set_ylabel(r"$z$")
    ax.set_xlim(0.82, 1.0); ax.set_ylim(3.8, 6.2)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"This work: $z_0$={z0:.2f}, $z_1$={z1:.2f}", fontsize=11)
    fig.suptitle("Fig. 2f  |  Jamming of the foam limit ($W=0$): "
                 "contact number vs volume fraction", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "compare_fig2f_jamming.png"), dpi=120)
    plt.close(fig)
    print("wrote compare_fig2f_jamming.png")


def fig4a():
    f = os.path.join(ROOT, "data", "fig4a.npz")
    if not os.path.exists(f):
        return
    d = np.load(f)
    dT_list, ts, sig, sig_sd, sA, fit = (d["dT_list"], d["ts"], d["sig"],
                                         d["sig_sd"], d["sigma_A"], d["fit"])

    def stretched(t, sinf, s0, tau, beta):
        return sinf + (s0 - sinf) * np.exp(-(np.clip(t, 0, None) / tau) ** beta)

    fig = plt.figure(figsize=(12, 5))
    axp = fig.add_subplot(1, 2, 1)
    _paper(axp, "crop_fig4a.png")
    ax = fig.add_subplot(1, 2, 2)
    for i, dT in enumerate(dT_list):
        ax.semilogx(ts, sig[i], "-", color=dtcol(dT), lw=2, label=f"{dT:.1f}")
        if np.all(np.isfinite(fit[i])):
            ax.semilogx(ts, stretched(ts, *fit[i]), "k--", lw=1)
    ax.set_xlabel(r"$t/\tau_R$"); ax.set_ylabel(r"$\sigma_{xy}(t)/\sigma_0$")
    ax.legend(title=r"$\Delta T/T_0$", fontsize=8, loc="upper right")
    ax.set_title("This work (dashed = stretched-exp fit)", fontsize=11)
    fig.suptitle(r"Fig. 4a  |  Shear-stress relaxation after a strain step "
                 r"($\rho=1,\ W/T_0=0.5$)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "compare_fig4a_stress.png"), dpi=120)
    plt.close(fig)
    print("wrote compare_fig4a_stress.png")


def shapefactor():
    f = os.path.join(ROOT, "data", "shapefactor.npz")
    if not os.path.exists(f):
        return
    d = np.load(f)
    W4, DT4, s4 = d["W4"], d["DT4"], d["s4"]
    W2, s2, s2sd = d["W2"], d["s2"], d["s2sd"]

    # Fig 4g
    fig = plt.figure(figsize=(11, 4.6))
    axp = fig.add_subplot(1, 2, 1); _paper(axp, "crop_fig4g.png")
    ax = fig.add_subplot(1, 2, 2)
    for i, dT in enumerate(DT4):
        ax.plot(W4, s4[i], "s-", color=dtcol(dT), ms=4, label=f"{dT:.2f}")
    ax.axhline(3.81, color="gray", ls="--", lw=0.8)
    ax.set_xlabel(r"$W/T_0$"); ax.set_ylabel(r"$\bar s$")
    ax.set_ylim(3.6, 4.3); ax.legend(title=r"$\Delta T/T_0$", fontsize=7, ncol=2)
    ax.set_title("This work", fontsize=11)
    fig.suptitle(r"Fig. 4g  |  Dynamic shape factor vs adhesion", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "compare_fig4g_shape.png"), dpi=120)
    plt.close(fig)
    print("wrote compare_fig4g_shape.png")

    # Fig 2g
    fig = plt.figure(figsize=(11, 4.6))
    axp = fig.add_subplot(1, 2, 1); _paper(axp, "crop_fig2g.png")
    ax = fig.add_subplot(1, 2, 2)
    ax.errorbar(W2, s2, yerr=s2sd, fmt="o-", color="#c0392b", ms=5, capsize=2)
    ax.axhline(3.81, color="gray", ls="--", lw=0.8)
    ax.set_xlabel(r"$W/T_0$"); ax.set_ylabel(r"$\bar s$")
    ax.set_title("This work (equilibrium, $\\Delta T=0$)", fontsize=11)
    fig.suptitle(r"Fig. 2g  |  Equilibrium shape factor: density-independent "
                 r"transition at $W/T_0=2$", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "compare_fig2g_shape.png"), dpi=120)
    plt.close(fig)
    print("wrote compare_fig2g_shape.png")


def fig2a():
    panel = os.path.join(OUT, "foam_fig2a_panel.png")
    if not os.path.exists(panel):
        return
    fig = plt.figure(figsize=(12, 5))
    axp = fig.add_subplot(1, 2, 1); _paper(axp, "crop_fig2a.png")
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(mpimg.imread(panel)); ax.axis("off")
    ax.set_title("This work (curved-edge foam model)", fontsize=11)
    fig.suptitle("Fig. 2a  |  Equilibrium configurations: extracellular spaces vs "
                 "adhesion & density", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "compare_fig2a_configs.png"), dpi=120)
    plt.close(fig)
    print("wrote compare_fig2a_configs.png")


def _phi_z_panel(npz, crop, title, outname, quantity):
    f = os.path.join(ROOT, "data", npz)
    if not os.path.exists(f):
        return
    d = np.load(f)
    WS, RHOS = d["WS"], d["RHOS"]
    Q = d[quantity]
    fig = plt.figure(figsize=(11, 4.6))
    axp = fig.add_subplot(1, 2, 1); _paper(axp, crop)
    ax = fig.add_subplot(1, 2, 2)
    cols = ["#2c7fb8", "#41ab5d", "#c0392b", "#7a5aa8"]
    for i, rho in enumerate(RHOS):
        ax.plot(WS, Q[i], "o-", color=cols[i % len(cols)], label=fr"$\rho={rho:.2f}$")
    ax.set_xlabel(r"$W/T_0$")
    ax.set_ylabel(r"$\phi$" if quantity == "phi" else r"$z$")
    ax.legend(fontsize=9)
    ax.set_title("This work", fontsize=11)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, outname), dpi=120)
    plt.close(fig)
    print("wrote", outname)


def fig2b():
    # deformable model gives phi across all three densities; overlay the faithful
    # foam-model rho=1 result as a cross-check
    f = os.path.join(ROOT, "data", "fig2bc_def.npz")
    if not os.path.exists(f):
        return
    d = np.load(f)
    WS, RHOS, phi = d["WS"], d["RHOS"], d["phi"]
    fig = plt.figure(figsize=(11, 4.6))
    axp = fig.add_subplot(1, 2, 1); _paper(axp, "crop_fig2b.png")
    ax = fig.add_subplot(1, 2, 2)
    cols = ["#c0392b", "#41ab5d", "#2c7fb8"]
    for i, rho in enumerate(RHOS):
        ax.plot(WS, phi[i], "o-", color=cols[i % 3], label=fr"$\rho={rho:.2f}$")
    ff = os.path.join(ROOT, "data", "fig2b_full.npz")
    if os.path.exists(ff):
        df = np.load(ff)
        ax.plot(df["WS"], df["phi"][0], "k^--", ms=5, label=r"foam model, $\rho=1$")
    ax.axhline(1.0, color="gray", lw=0.6, ls=":")
    ax.set_xlabel(r"$W/T_0$"); ax.set_ylabel(r"$\phi$")
    ax.set_ylim(0.6, 1.03); ax.legend(fontsize=8)
    ax.set_title("This work", fontsize=11)
    fig.suptitle(r"Fig. 2b  |  Volume fraction vs adhesion & density", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "compare_fig2b_phi.png"), dpi=120)
    plt.close(fig)
    print("wrote compare_fig2b_phi.png")


def fig2c():
    # unified foam model now produces z via the T1->T3-reverse contact-breaking
    # chain; fall back to the deformable model if the foam sweep isn't present.
    src = "fig2b_full.npz" if os.path.exists(
        os.path.join(ROOT, "data", "fig2b_full.npz")) else "fig2bc_def.npz"
    crop = "crop_fig2c.png"
    f = os.path.join(ROOT, "data", src)
    d = np.load(f)
    WS = d["WS"]; RHOS = d["RHOS"]; Q = d["z"]
    fig = plt.figure(figsize=(11, 4.6))
    axp = fig.add_subplot(1, 2, 1); _paper(axp, crop)
    ax = fig.add_subplot(1, 2, 2)
    cols = ["#c0392b", "#41ab5d", "#2c7fb8", "#7a5aa8"]
    for i, rho in enumerate(RHOS):
        ax.plot(WS, Q[i], "o-", color=cols[i % 4], label=fr"$\rho={rho:.2f}$")
    ax.axhline(6, color="gray", lw=0.6, ls=":")
    ax.set_xlabel(r"$W/T_0$"); ax.set_ylabel(r"$z$"); ax.set_ylim(2, 6.5)
    ax.legend(fontsize=9)
    model = "faithful foam model" if src.startswith("fig2b") else "deformable model"
    ax.set_title(f"This work ({model})", fontsize=11)
    fig.suptitle(r"Fig. 2c  |  Contact number vs adhesion", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "compare_fig2c_z.png"), dpi=120)
    plt.close(fig)
    print("wrote compare_fig2c_z.png  (source:", src, ")")


def fig2i():
    f = os.path.join(ROOT, "data", "fig2i.npz")
    if not os.path.exists(f):
        return
    d = np.load(f)
    WS, ts, sig = d["WS"], d["ts"], d["sig"]
    tr = np.clip(ts, 1e-3, None)
    fig = plt.figure(figsize=(11, 4.6))
    axp = fig.add_subplot(1, 2, 1); _paper(axp, "crop_fig2i.png")
    ax = fig.add_subplot(1, 2, 2)
    wcol = {0.0: "#3b6fb6", 0.5: "#1f9e8b", 1.0: "#7a5aa8",
            1.5: "#b5a11d", 2.0: "#b95a24"}
    for i, w in enumerate(WS):
        ax.semilogx(tr, sig[i], "-", lw=2,
                    color=wcol.get(round(float(w), 2), "k"), label=f"{w:.1f}")
    ax.set_xlabel(r"$t/\tau_R$"); ax.set_ylabel(r"$\sigma_{xy}(t)/\sigma_0$")
    ax.legend(title=r"$W/T_0$", fontsize=8)
    ax.set_title(r"This work ($\rho=1,\ \Delta T=0$)", fontsize=11)
    fig.suptitle(r"Fig. 2i  |  Shear-stress relaxation vs adhesion "
                 r"(paper Eq. 4/5 stress)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "compare_fig2i_stress.png"), dpi=120)
    plt.close(fig)
    print("wrote compare_fig2i_stress.png")


def fig2j():
    f = os.path.join(ROOT, "data", "fig2j.npz")
    if not os.path.exists(f):
        return
    d = np.load(f)
    WS, sY, err = d["WS"], d["sigmaY"], d["err"]
    m = np.isfinite(sY)
    fig = plt.figure(figsize=(11, 4.6))
    axp = fig.add_subplot(1, 2, 1); _paper(axp, "crop_fig2j.png")
    ax = fig.add_subplot(1, 2, 2)
    ax.axvspan(0.4, 0.6, color="#8fd18f", alpha=0.35)          # structural transition
    ax.errorbar(WS[m], sY[m], yerr=err[m], fmt="o-", color="k", ms=5, capsize=2)
    wpk = WS[m][np.argmax(sY[m])]
    ax.set_xlabel(r"$W/T_0$"); ax.set_ylabel(r"$\sigma_Y/\sigma_0$")
    ax.set_ylim(0, 0.45); ax.set_xlim(-0.05, 2.05)
    ax.set_title(fr"This work (peak at $W/T_0\approx{wpk:.1f}$; $\rho=1,\ \Delta T=0$)",
                 fontsize=11)
    fig.suptitle(r"Fig. 2j  |  Yield stress vs adhesion — maximal rigidity at the "
                 r"structural transition", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "compare_fig2j_yield.png"), dpi=120)
    plt.close(fig)
    print("wrote compare_fig2j_yield.png")


TAU_T = 10.0          # tau_T / tau_R
KNE_MAX = 0.5         # fully-fluid NE-rate ceiling (per cell per tau_R)


def _tausr(kne):
    """tau_SR/tau_R from the cellular NE rate: tau_SR = tau_T * (k_max / k_NE)."""
    return TAU_T * KNE_MAX / np.clip(kne, 1e-6, None)


def _de_grid(ax, DTS, WS, phi, values=None, cmap=None, norm=None,
             classify=None, sq=0.34):
    """Shared renderer for the Fig 4d/4e square grids over (W/T0, dT/T0)."""
    import matplotlib.patches as mpatches
    green = "#bfe3c0"
    confl = phi >= 0.999
    dW = WS[1] - WS[0]
    dD = DTS[1] - DTS[0]
    # green background: confluent cells (paper's structural transition)
    for i, d in enumerate(DTS):
        for j, w in enumerate(WS):
            if confl[i, j]:
                ax.add_patch(mpatches.Rectangle(
                    (w - dW / 2, d - dD / 2), dW, dD,
                    facecolor=green, edgecolor="none", zorder=0))
    for i, d in enumerate(DTS):
        for j, w in enumerate(WS):
            if classify is not None:
                c = classify(i, j)
            else:
                c = cmap(norm(values[i, j]))
            ax.add_patch(mpatches.Rectangle(
                (w - sq * dW, d - sq * dD), 2 * sq * dW, 2 * sq * dD,
                facecolor=c, edgecolor="0.35", lw=0.5, zorder=2))
    ax.set_xlim(WS[0] - dW, WS[-1] + dW)
    ax.set_ylim(DTS[0] - dD, DTS[-1] + dD)
    ax.set_xlabel(r"$W/T_0$"); ax.set_ylabel(r"$\Delta T/T_0$")
    ax.set_xticks([0, 0.4, 0.8, 1.2]); ax.set_yticks([0.5, 1.0, 1.5])


def fig4d():
    f = os.path.join(ROOT, "data", "fig4de.npz")
    if not os.path.exists(f):
        return
    import matplotlib.colors as mcolors
    from matplotlib.cm import ScalarMappable
    d = np.load(f)
    DTS, WS, kne, phi = d["DTS"], d["WS"], d["kne"], d["phi"]
    # tau_SR = tau_T * (k_max / k_NE): stress relaxes in ~one tension-persistence
    # time at the fully-fluid NE-rate ceiling, and diverges as the NE rate -> 0.
    tauSR = _tausr(kne)
    cmap = plt.get_cmap("RdYlBu")
    norm = mcolors.Normalize(vmin=1, vmax=6)          # log10(tau_SR/tau_R): 10^1..10^6
    logt = np.log10(np.clip(tauSR, 10, 1e6))
    fig = plt.figure(figsize=(11, 4.8))
    axp = fig.add_subplot(1, 2, 1); _paper(axp, "crop_fig4d.png")
    ax = fig.add_subplot(1, 2, 2)
    _de_grid(ax, DTS, WS, phi, values=logt, cmap=cmap, norm=norm)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
    cb.set_label(r"$\log_{10}(\tau_{SR}/\tau_R)$")
    ax.text(0.05, 1.42, "Non-confluent", color="0.5", fontsize=9)
    ax.text(0.78, 0.62, "Confluent", color="#2f8f4f", fontsize=9)
    ax.set_title("This work (foam model; " r"$\tau_{SR}=1/k_{NE}$)", fontsize=11)
    fig.suptitle(r"Fig. 4d  |  Stress-relaxation time $\tau_{SR}$ vs adhesion and "
                 r"activity (max at the structural transition)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "compare_fig4d_tauSR.png"), dpi=120)
    plt.close(fig)
    print("wrote compare_fig4d_tauSR.png")


def fig4e():
    f = os.path.join(ROOT, "data", "fig4de.npz")
    if not os.path.exists(f):
        return
    d = np.load(f)
    DTS, WS, kne, phi = d["DTS"], d["WS"], d["kne"], d["phi"]
    tauT = 10.0                                        # tau_T = 10 tau_R
    tauSR = _tausr(kne)
    solid = (tauSR / tauT) >= 1e2                       # paper's Fig 4e criterion
    fig = plt.figure(figsize=(11, 4.8))
    axp = fig.add_subplot(1, 2, 1); _paper(axp, "crop_fig4e.png")
    ax = fig.add_subplot(1, 2, 2)
    _de_grid(ax, DTS, WS, phi,
             classify=lambda i, j: ("#2b3fd6" if solid[i, j] else "#e3271f"))
    ax.text(0.05, 1.28, "Fluid", color="#e3271f", fontsize=11)
    ax.text(0.55, 0.62, "Solid", color="#2b3fd6", fontsize=11)
    ax.text(0.05, 1.42, r"$\tau_{SR}/\tau_T = 10^2$", fontsize=10)
    ax.set_title("This work (foam model)", fontsize=11)
    fig.suptitle(r"Fig. 4e  |  Fluid / solid phase diagram — solid states "
                 r"surround the structural transition", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "compare_fig4e_phase.png"), dpi=120)
    plt.close(fig)
    print("wrote compare_fig4e_phase.png")


if __name__ == "__main__":
    fig2a()
    fig2b()
    fig2c()
    fig2i()
    fig2j()
    fig3a()
    fig2f()
    fig4a()
    fig4d()
    fig4e()
    shapefactor()
    print("done")
