import numpy as np

from gensec.materials import ConcreteEC2Nonlinear


# ----------------------------------------------------------
# Material
# ----------------------------------------------------------

mat = ConcreteEC2Nonlinear(
    fc=42,
    Ecm=33800,
    eps_c1=-0.0022,
    eps_cu=-0.0035,
    fct=1.68,
    eps_tu=0.00011,
    #k=1.9961
)


# ----------------------------------------------------------
# Dehnungsbereich
# ----------------------------------------------------------
# von -3.7 mm/m bis 0.0 mm/m
# 66 Werte

eps_mm_m = np.linspace(0, 0.15, 20)

# GenSec benötigt dimensionslose Dehnung
eps = eps_mm_m / 1000.0


# ----------------------------------------------------------
# Material auswerten
# ----------------------------------------------------------

sigma = mat.stress_array(eps)
Et = mat.tangent_array(eps)


# ----------------------------------------------------------
# Wertetabelle ausgeben
# ----------------------------------------------------------

print()
print(" Nr. | eps [mm/m] | sigma [N/mm²] | Et [N/mm²]")
print("----------------------------------------------------")

for i, (e, s, et) in enumerate(
    zip(eps_mm_m, sigma, Et),
    start=1
):
    print(
        f"{i:4d} | "
        f"{e:10.4f} | "
        f"{s:14.4f} | "
        f"{et:11.4f}"
    )