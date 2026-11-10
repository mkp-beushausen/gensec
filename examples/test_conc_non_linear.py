from gensec.materials.steel import Steel
import numpy as np

from gensec.materials import ConcreteEC2Nonlinear
from gensec.materials.ec2_bridge import prestress_from_class




# ----------------------------------------------------------
# Material
# ----------------------------------------------------------

# mat = ConcreteEC2Nonlinear(
#     fc=42,
#     Ecm=33800,
#     eps_c1=-0.0022,
#     eps_cu=-0.0035,
#     fct=1.68,
#     eps_cr=0.00011,
#     use_vmb=True,
#     eps_vmb_u=0.0025,
#     n_vmb= 1.0,
    
#     #n_tension=1.9961,
#     #eps_tu=0.00011,
#     #k=1.9961
# )

mat=prestress_from_class(
    ps_class="St1570/1770",
    ls="F",
    NA="EC2",
    eps_ud_factor=0.9,
    diagram="inclined",
    works_in_compression=True,
)

# mat= Steel(
#     fyk=500.0,
#     gamma_s=1.15,
#     Es=200000,
#     k_hardening=1.0,
#     eps_su=0.01,
#     works_in_compression=True
# )
    


# ----------------------------------------------------------
# Dehnungsbereich
# ----------------------------------------------------------
# von -3.7 mm/m bis 0.0 mm/m
# 66 Werte

eps_mm_m = np.linspace(-35, 35,150)

# GenSec benötigt dimensionslose Dehnung
eps = eps_mm_m / 1000.0


# ----------------------------------------------------------
# Material auswerten
# ----------------------------------------------------------

sigma = mat.stress_array(eps)
Et = mat.tangent_array(eps)


""" # ----------------------------------------------------------
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

print() """


print()
#print(f"Faktor k = {mat.k}")
#print(f"n_tension = {mat.n_tension}")
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


print()
print("SPANNSTAHL EC2")
print("----------------------------")
print(f"ls        = {mat.ec2.ls}")
print(f"NA        = {mat.ec2.NA}")
print(f"gamma_s   = {mat.ec2.gamma_s:.4f}")
print(f"f_p01k    = {mat.ec2.f_p01k:.4f} MPa")
print(f"f_pk      = {mat.ec2.f_pk:.4f} MPa")
print(f"f_p01d    = {mat.ec2.f_p01d:.4f} MPa")
print(f"f_pd      = {mat.ec2.f_pd:.4f} MPa")
print(f"eps_uk    = {mat.ec2.eps_uk:.6f}")
print(f"eps_ud    = {mat.ec2.eps_ud:.6f}")
print(f"Ep        = {mat.ec2.Ep:.4f} MPa")

print()
print("SPANNSTAHL-PARAMETER")
print("--------------------")

# Grundwerte Material
print(f"Ep                  = {mat.Ep:.8f} N/mm²")
print(f"f_p01d              = {mat.f_p01d:.8f} N/mm²")
print(f"sigma_ud            = {mat.sigma_ud:.8f} N/mm²")
print(f"eps_el              = {mat.eps_el:.8f} [-]")
print(f"eps_el              = {mat.eps_el * 1000:.8f} mm/m")
print(f"eps_ud              = {mat.eps_ud:.8f} [-]")
print(f"eps_ud              = {mat.eps_ud * 1000:.8f} mm/m")
print(f"E_hard              = {mat.E_hard:.8f} N/mm²")
print(f"works_in_compression = {mat.works_in_compression}")

# EC2-Ausgangswerte
if hasattr(mat, "ec2"):

    print()
    print("EC2-PARAMETER")
    print("-------------")

    print(f"f_p01k              = {mat.ec2.f_p01k:.8f} N/mm²")
    print(f"f_pk                = {mat.ec2.f_pk:.8f} N/mm²")
    print(f"gamma_s             = {mat.ec2.gamma_s:.8f}")
    print(f"f_p01d              = {mat.ec2.f_p01d:.8f} N/mm²")
    print(f"f_pd                = {mat.ec2.f_pd:.8f} N/mm²")
    print(f"Ep                   = {mat.ec2.Ep:.8f} N/mm²")

    print(f"eps_uk              = {mat.ec2.eps_uk:.8f} [-]")
    print(f"eps_uk              = {mat.ec2.eps_uk * 1000:.8f} mm/m")

    print(f"eps_el              = {mat.ec2.eps_el:.8f} [-]")
    print(f"eps_el              = {mat.ec2.eps_el * 1000:.8f} mm/m")

    print(f"eps_ud              = {mat.ec2.eps_ud:.8f} [-]")
    print(f"eps_ud              = {mat.ec2.eps_ud * 1000:.8f} mm/m")

    # if hasattr(mat.ec2, "eps_ud_ec2"):
    #     print(
    #         f"eps_ud_ec2          = "
    #         f"{mat.ec2.eps_ud_ec2:.8f} [-]"
    #     )
    #     print(
    #         f"eps_ud_ec2          = "
    #         f"{mat.ec2.eps_ud_ec2 * 1000:.8f} mm/m"
    #     )

    # if hasattr(mat.ec2, "eps_ud_explicit"):
    #     print(
    #         f"eps_ud_explicit     = "
    #         f"{mat.ec2.eps_ud_explicit}"
    #     )

    # if hasattr(mat.ec2, "sigma_ud_inclined"):
    #     print(
    #         f"sigma_ud_inclined   = "
    #         f"{mat.ec2.sigma_ud_inclined:.8f} N/mm²"
    #     )

    # if hasattr(mat.ec2, "E_hard_characteristic"):
    #     print(
    #         f"E_hard_characteristic = "
    #         f"{mat.ec2.E_hard_characteristic:.8f} N/mm²"
    #     )
