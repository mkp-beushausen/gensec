import numpy as np
eps0_test = np.array([
    -0.0010,
    0.0000,
    0.0005,
    0.0010,
    0.0015,
    0.0020,
    0.0025,
    0.0030,
])

chi_x_test = np.array([
    1.0e-5,
    1.0e-5,
    1.0e-5,
    1.0e-5,
    1.0e-5,
    1.0e-5,
    1.0e-5,
    1.0e-5,
])

chi_y_test = np.zeros_like(
    eps0_test
)

N_batch, Mx_batch, My_batch = (
    solver.integrate_batch(
        eps0_test,
        chi_x_test,
        chi_y_test,
    )
)

for i in range(
    len(eps0_test)
):

    N_scalar, Mx_scalar, My_scalar = (
        solver.integrate(
            eps0_test[i],
            chi_x_test[i],
            chi_y_test[i],
        )
    )

    print(
        f"{i:3d} | "
        f"dN={N_batch[i] - N_scalar: .6e} | "
        f"dMx={Mx_batch[i] - Mx_scalar: .6e} | "
        f"dMy={My_batch[i] - My_scalar: .6e}"
    )

    assert np.isclose(
        N_batch[i],
        N_scalar,
        rtol=1e-10,
        atol=1e-5,
    )

    assert np.isclose(
        Mx_batch[i],
        Mx_scalar,
        rtol=1e-10,
        atol=1e-3,
    )

    assert np.isclose(
        My_batch[i],
        My_scalar,
        rtol=1e-10,
        atol=1e-3,
    )

print(
    "Batch and scalar integration agree."
)