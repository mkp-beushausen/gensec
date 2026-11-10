import numpy as np
import matplotlib.pyplot as plt

from gensec.materials import ConcreteEC2Nonlinear
from gensec.geometry import RectSection, RebarLayer
from gensec.materials import Steel
from gensec.solver import FiberSolver


# ==========================================================
# 1. Materials
# ==========================================================

concrete = ConcreteEC2Nonlinear(
    fc=42.0,
    Ecm=33800.0,
    eps_c1=-0.0022,
    eps_cu=-0.0035,
    k=1.9961,

    # Quast tension branch
    fct=1.68,
    eps_cr=0.00011,
    n_tension=1.9961,

    # Quast VMB
    use_vmb=True,
    eps_vmb_u=0.0025,
    n_vmb=1.99990,
)


steel = Steel(
    fyk=500.0,
    gamma_s=1.0,
    Es=200000.0,
    k_hardening=1.0,
    eps_su=0.025,
)


# ==========================================================
# 2. Simple test section
# ==========================================================
#
# A simple section is sufficient because this test initially
# checks only the VMB helper functions.
#
# Governing reinforcement is placed in the tensile edge.
# ==========================================================

B = 300.0
H = 500.0

As = 1000.0


rebars = [
    RebarLayer(
        y=50.0,
        As=As,
        material=steel,
    ),
    RebarLayer(
        y=450.0,
        As=As,
        material=steel,
    ),
]


section = RectSection(
    B=B,
    H=H,
    bulk_material=concrete,
    rebars=rebars,
    n_fibers_y=50,
)


solver = FiberSolver(section)


# ==========================================================
# 3. Direct VMB test
# ==========================================================

print()
print("DIRECT VMB TEST")
print()

print(
    " eps_ref [mm/m] |"
    "      VMB |"
    " dVMB/deps_ref"
)

print(
    "-----------------+"
    "----------+"
    "----------------"
)


eps_ref_mm_m = np.array([
    0.000,
    0.050,
    0.100,
    0.110,
    0.200,
    0.500,
    1.000,
    1.305,
    1.500,
    2.000,
    2.400,
    2.500,
    2.600,
])


for eps_mm_m in eps_ref_mm_m:

    eps_ref = (
        eps_mm_m
        / 1000.0
    )

    vmb, dvmb = (
        solver._quast_vmb_from_reference_strain(
            bulk_material=concrete,
            eps_c_ref=eps_ref,
        )
    )

    print(
        f"{eps_mm_m:16.4f} |"
        f"{vmb:9.6f} |"
        f"{dvmb:16.6f}"
    )


# ==========================================================
# 4. Expected characteristic values
# ==========================================================

print()
print("CHECKPOINTS")
print()


eps_cr = concrete.eps_cr
eps_u = concrete.eps_vmb_u


# ----------------------------------------------------------
# Below cracking
# ----------------------------------------------------------

vmb_0, dvmb_0 = (
    solver._quast_vmb_from_reference_strain(
        concrete,
        0.0,
    )
)

assert np.isclose(
    vmb_0,
    1.0,
)

assert np.isclose(
    dvmb_0,
    0.0,
)


# ----------------------------------------------------------
# Exactly at cracking
# ----------------------------------------------------------

vmb_cr, dvmb_cr = (
    solver._quast_vmb_from_reference_strain(
        concrete,
        eps_cr,
    )
)

assert np.isclose(
    vmb_cr,
    1.0,
)


# ----------------------------------------------------------
# Halfway between cracking and end strain
# ----------------------------------------------------------

eps_half = (
    eps_cr
    + 0.5
    * (
        eps_u
        - eps_cr
    )
)

vmb_half, dvmb_half = (
    solver._quast_vmb_from_reference_strain(
        concrete,
        eps_half,
    )
)


# ----------------------------------------------------------
# Expected VMB at half of the reduction interval
# ----------------------------------------------------------

expected_vmb_half = (
    0.5 ** concrete.n_vmb
)

assert np.isclose(
    vmb_half,
    expected_vmb_half,
    atol=1e-12,
)


expected_dvmb = (
    -concrete.n_vmb
    / (
        eps_u
        - eps_cr
    )
    * 0.5 ** (
        concrete.n_vmb - 1.0
    )
)
assert np.isclose(
    dvmb_half,
    expected_dvmb,
    atol=1e-9,
)


# ----------------------------------------------------------
# Exactly at end strain
# ----------------------------------------------------------

vmb_u, dvmb_u = (
    solver._quast_vmb_from_reference_strain(
        concrete,
        eps_u,
    )
)

assert np.isclose(
    vmb_u,
    0.0,
)

assert np.isclose(
    dvmb_u,
    0.0,
)


# ----------------------------------------------------------
# Beyond end strain
# ----------------------------------------------------------

vmb_after, dvmb_after = (
    solver._quast_vmb_from_reference_strain(
        concrete,
        eps_u + 0.001,
    )
)

assert np.isclose(
    vmb_after,
    0.0,
)

assert np.isclose(
    dvmb_after,
    0.0,
)


print("All direct VMB checks passed.")


# ==========================================================
# 5. Check reference-rebar selection
# ==========================================================

print()
print("REFERENCE REBAR TEST")
print()


# Artificial rebar strains:
#
# Rebar 1 = 0.60 mm/m
# Rebar 2 = 1.30 mm/m
#
# Rebar 2 must govern.
eps_rebar = np.array([
    0.00060,
    0.00130,
])


state = solver._compute_quast_state(
    eps_rebar=eps_rebar,
    bulk_material=concrete,
)


print(
    "governing_index =",
    state["governing_index"],
)

print(
    "eps_c_ref [mm/m] =",
    state["eps_c_ref"] * 1000.0,
)

print(
    "VMB =",
    state["vmb"],
)

print(
    "dVMB/deps_ref =",
    state["dvmb_deps_ref"],
)


assert (
    state["governing_index"]
    == 1
)


print()
print(
    "Reference-rebar selection passed."
)

print()
print(
    "ALL QUAST VMB TESTS PASSED."
)

# ==========================================================
# 6. Plot VMB over reference concrete strain
# ==========================================================

eps_plot_mm_m = np.linspace(
    0.0,
    3.0,
    500,
)

eps_plot = (
    eps_plot_mm_m
    / 1000.0
)

vmb_plot = np.zeros_like(
    eps_plot
)

dvmb_plot = np.zeros_like(
    eps_plot
)


for i, eps_ref in enumerate(eps_plot):

    (
        vmb_plot[i],
        dvmb_plot[i],
    ) = solver._quast_vmb_from_reference_strain(
        bulk_material=concrete,
        eps_c_ref=eps_ref,
    )


# ==========================================================
# Plot
# ==========================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    eps_plot_mm_m,
    vmb_plot,
    label="VMB",
)

# Crack strain
ax.axvline(
    concrete.eps_cr * 1000.0,
    linestyle="--",
    label=(
        "eps_cr = "
        f"{concrete.eps_cr * 1000.0:.3f} mm/m"
    ),
)

# End strain
ax.axvline(
    concrete.eps_vmb_u * 1000.0,
    linestyle="--",
    label=(
        "eps_u = "
        f"{concrete.eps_vmb_u * 1000.0:.3f} mm/m"
    ),
)

ax.set_xlabel(
    "Betondehnung am Referenzstahl eps_ref [mm/m]"
)

ax.set_ylabel(
    "VMB [-]"
)

ax.set_title(
    "Abminderungsfaktor VMB nach Quast"
)

ax.set_xlim(
    eps_plot_mm_m.min(),
    eps_plot_mm_m.max(),
)

ax.set_ylim(
    -0.05,
    1.05,
)

ax.grid(True)

ax.legend()

fig.tight_layout()


# ==========================================================
# Save plot
# ==========================================================

output_file = (
    "quast_vmb_verlauf.png"
)

fig.savefig(
    output_file,
    dpi=200,
)

plt.close(fig)


print()
print(
    f"VMB plot saved as: {output_file}"
)