import numpy as np

#from gensec.materials import ConcreteEC2Nonlinear
from gensec.materials.ec2_bridge import prestress_from_class




# ----------------------------------------------------------
# Material
# ----------------------------------------------------------

""" mat = ConcreteEC2Nonlinear(
    fc=42,
    Ecm=33800,
    eps_c1=-0.0022,
    eps_cu=-0.0035,
    fct=1.68,
    eps_cr=0.00011,
    n_tension=1.9961,
    #eps_tu=0.00011,
    #k=1.9961
) """

mat=prestress_from_class(
    ps_class="St140/160",
    ls="F",
    NA="EC2",
    eps_ud_factor=0.9,
    diagram="inclined",
    works_in_compression=True,
)

# ----------------------------------------------------------
# Dehnungsbereich
# ----------------------------------------------------------
# von -3.7 mm/m bis 0.0 mm/m
# 66 Werte

eps_mm_m = np.linspace(-25, 25,150)

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
print("Prestressing steel parameters")
print("-----------------------------")

print(f"f_p01d   = {mat.f_p01d:.4f} MPa")
print(f"sigma_ud = {mat.sigma_ud:.4f} MPa")
print(f"eps_ud   = {mat.eps_ud:.8f}")
print(f"Ep       = {mat.Ep:.4f} MPa")

print()


print("EC2 bridge parameters")
print("---------------------")

print(f"f_p01k            = {mat.ec2.f_p01k:.4f} MPa")
print(f"f_pk              = {mat.ec2.f_pk:.4f} MPa")
print(f"eps_uk            = {mat.ec2.eps_uk:.8f}")
print(f"Ep                = {mat.ec2.Ep:.4f} MPa")

print(f"f_p01d            = {mat.ec2.f_p01d:.4f} MPa")
print(f"eps_ud            = {mat.ec2.eps_ud:.8f}")
print(
    f"sigma_ud_inclined = "
    f"{mat.ec2.sigma_ud_inclined:.4f} MPa"
)

print()


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