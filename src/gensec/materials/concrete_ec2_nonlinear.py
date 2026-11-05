"""
Nonlinear concrete material for GenSec.

Sign convention
---------------
Compression:
    strain < 0
    stress < 0

Tension:
    strain > 0
    stress > 0

Compression law
---------------
For eps_cu < eps < 0:

    eta = eps / eps_c1

    sigma = -fc * (k * eta - eta**2)
                  / (1 + (k - 2) * eta)

For eps <= eps_cu:

    sigma = sigma(eps_cu)
    tangent = 0

Tension law
-----------
For 0 <= eps <= eps_cr:

    sigma = Et * eps

with:

    eps_cr = fct / Et

If eps_tu is not specified:

    sigma = 0 for eps > eps_cr

If eps_tu > eps_cr is specified:

    linear tensile softening from fct at eps_cr
    to zero stress at eps_tu.
"""

import numpy as np

from .base import Material


class ConcreteEC2Nonlinear(Material):
    """
    Nonlinear concrete material with optional tensile branch.

    Parameters
    ----------
    fc : float
        Maximum compressive stress [MPa].

        Enter as positive value, for example:

            fc = 42.0

    Ecm : float
        Concrete modulus used for automatic calculation of k [MPa].

    eps_c1 : float
        Compressive strain at maximum compressive stress [-].

        May be entered positive or negative. It is internally
        stored as a negative value.

        Example:

            eps_c1 = -0.0022

    eps_cu : float
        Start of the constant compression plateau [-].

        May be entered positive or negative. It is internally
        stored as a negative value.

        For eps <= eps_cu:

            sigma = sigma(eps_cu)
            tangent = 0

        Example:

            eps_cu = -0.0035

    k : float, optional
        Shape parameter of the compression curve.

        If not specified:

            k = 1.05 * Ecm * abs(eps_c1) / fc

    fct : float, optional
        Concrete tensile strength [MPa].

        Default:

            fct = 0.0

        If fct = 0, concrete tension is neglected.

    Et : float, optional
        Elastic modulus for the tensile branch [MPa].

        If not specified, Ecm is used.

    eps_tu : float, optional
        Ultimate tensile strain [-].

        If eps_tu is not specified, tensile stress drops to zero
        immediately after cracking.

        If eps_tu > eps_cr, linear tensile softening is used:

            sigma = fct at eps_cr
            sigma = 0 at eps_tu

    name : str, optional
        Material label.
    """

    def __init__(
        self,
        fc,
        Ecm,
        eps_c1,
        eps_cu,
        k=None,
        fct=0.0,
        eps_cr=None,
        n_tension=None,
        name="ConcreteEC2Nonlinear",
    ):
        # ==========================================================
        # Compression parameters
        # ==========================================================

        self.fc = abs(float(fc))
        self.Ecm = float(Ecm)
        self.E = self.Ecm

        # GenSec sign convention:
        # compression is negative
        self.eps_c1 = -abs(float(eps_c1))
        self.eps_cu = -abs(float(eps_cu))

        # ==========================================================
        # Compression shape parameter
        # ==========================================================

        if k is None:
            self.k = (
                1.10
                * self.Ecm
                * abs(self.eps_c1)
                / self.fc
            )
        else:
            self.k = float(k)

        # ------------------------------------------------------
        # Tension according to Quast
        # ------------------------------------------------------

        self.fct = max(
            0.0,
            float(fct),
        )

        if eps_cr is None:
            self.eps_cr = 0.0
        else:
            self.eps_cr = abs(
                float(eps_cr)
            )

        if n_tension is None:
            self.n_tension = self.k
        else:
            self.n_tension = float(
                n_tension
            )

        self._name = str(name)

        # ==========================================================
        # Validation
        # ==========================================================

        if self.fc <= 0.0:
            raise ValueError(
                "fc must be greater than zero."
            )

        if self.Ecm <= 0.0:
            raise ValueError(
                "Ecm must be greater than zero."
            )

        if abs(self.eps_c1) <= 0.0:
            raise ValueError(
                "eps_c1 must not be zero."
            )

        if abs(self.eps_cu) <= abs(self.eps_c1):
            raise ValueError(
                "abs(eps_cu) must be greater than "
                "abs(eps_c1)."
            )

        if self.k <= 0.0:
            raise ValueError(
                "k must be greater than zero."
            )

        if self.fct > 0.0:
            if self.eps_cr <= 0.0:
                raise ValueError(
                    "eps_cr must be greater than zero if fct > 0."
                )
            if self.n_tension <= 0.0:
                raise ValueError(
                    "n_tension must be greater than zero if fct > 0."
                )

        # ==========================================================
        # Precomputed values
        # ==========================================================

        # Constant compression stress beyond eps_cu
        self._sigma_cu = float(
            self._stress_compression(
                self.eps_cu
            )
        )


    # ==============================================================
    # Properties required by GenSec
    # ==============================================================

    @property
    def eps_min(self):
        """
        Lowest formally admissible strain.

        Compression remains defined beyond eps_cu because the
        stress is kept constant.
        """

        return -1.0

    @property
    def eps_max(self):
        """
        Highest formally admissible strain.

        Tensile strains beyond cracking are handled either by
        immediate failure or by the optional softening branch.
        """

        return 1.0

    @property
    def name(self):
        """Material label."""

        return self._name

    @name.setter
    def name(self, value):
        """
        Allow the YAML loader to assign the YAML material name.

        GenSec calls:

            mat.name = mat_name
        """

        self._name = str(value)

    # ==============================================================
    # Internal compression functions
    # ==============================================================

    def _stress_compression(self, eps):
        """
        Nonlinear compression stress.

        Parameters
        ----------
        eps : float or numpy.ndarray
            Negative compressive strain.

        Returns
        -------
        float or numpy.ndarray
            Stress [MPa].
        """

        eta = eps / self.eps_c1

        numerator = (
            self.k * eta
            - eta**2
        )

        denominator = (
            1.0
            + (self.k - 2.0) * eta
        )

        return (
            -self.fc
            * numerator
            / denominator
        )

    def _tangent_compression(self, eps):
        """
        Analytical tangent modulus of the compression curve.

        Parameters
        ----------
        eps : float or numpy.ndarray
            Negative compressive strain.

        Returns
        -------
        float or numpy.ndarray
            Tangent modulus [MPa].
        """

        eta = eps / self.eps_c1

        numerator = (
            self.k * eta
            - eta**2
        )

        denominator = (
            1.0
            + (self.k - 2.0) * eta
        )

        dnumerator_deta = (
            self.k
            - 2.0 * eta
        )

        ddenominator_deta = (
            self.k
            - 2.0
        )

        df_deta = (
            (
                dnumerator_deta
                * denominator
            )
            -
            (
                numerator
                * ddenominator_deta
            )
        ) / denominator**2

        # sigma = -fc * f(eta)
        # eta = eps / eps_c1

        return (
            -self.fc
            * df_deta
            / self.eps_c1
        )

    # ==============================================================
    # Internal tension functions
    # ==============================================================

    def _stress_tension(self, eps):
        """
        Concrete tension branch according to Quast.
        For:
            0 <= eps < eps_cr
        sigma = fct * [
            1 - (1 - eps / eps_cr)**k oder **n_tension    k kann vom Druckbereich übernommen werden, da nach den selben Vorschriften berechnet
        ]
        For:
            eps >= eps_cr
        sigma = fct
        Notes
        -----
        This is the unscaled parabola-rectangle target curve.
        A strain-dependent VMB reduction is not applied here.
        """
        eps = float(eps)
        # Tension disabled
        if self.fct <= 0.0:
            return 0.0
        # Origin
        if eps <= 0.0:
            return 0.0
        # Parabolic branch
        if eps < self.eps_cr:
            eta = ( eps/ self.eps_cr)
            return (self.fct* (1.0-(1.0 - eta)** self.n_tension))
        # Rectangle branch
        return self.fct

    def _tangent_tension(self, eps):
        """
        Tangent modulus of the Quast tension branch according to n_tension.
        """
        eps = float(eps)
        if self.fct <= 0.0:
            return 0.0
        if eps < 0.0:
            return 0.0
        if eps < self.eps_cr:
            eta = (eps/ self.eps_cr)
            return (self.fct* self.n_tension/ self.eps_cr* (1.0 - eta)** (self.n_tension - 1.0))
        # Rectangle branch
        return 0.0

    # ==============================================================
    # Scalar interface
    # ==============================================================

    def stress(self, eps):
        """
        Scalar stress evaluation.

        Compression:
            eps <= eps_cu:
                constant compression plateau

            eps_cu < eps < 0:
                nonlinear compression law

        Tension:
            0 <= eps <= eps_cr:
                linear elastic tension

            eps > eps_cr:
                brittle failure or linear softening
        """

        eps = float(eps)

        # ----------------------------------------------------------
        # Compression plateau
        # ----------------------------------------------------------

        if eps <= self.eps_cu:
            return self._sigma_cu

        # ----------------------------------------------------------
        # Nonlinear compression
        # ----------------------------------------------------------

        if eps < 0.0:
            return float(
                self._stress_compression(eps)
            )

        # ----------------------------------------------------------
        # Tension
        # ----------------------------------------------------------

        return float(
            self._stress_tension(eps)
        )

    def tangent(self, eps):
        """
        Scalar tangent modulus.

        Compression plateau:
            tangent = 0

        Nonlinear compression:
            analytical tangent modulus

        Linear tension:
            tangent = Et

        Tensile softening:
            negative constant tangent modulus

        Fully cracked tension:
            tangent = 0
        """

        eps = float(eps)

        # Compression plateau
        if eps <= self.eps_cu:
            return 0.0

        # Nonlinear compression
        if eps < 0.0:
            return float(
                self._tangent_compression(eps)
            )

        # Tension
        return float(
            self._tangent_tension(eps)
        )

    # ==============================================================
    # Vectorized interface
    # ==============================================================

    def stress_array(self, eps):
        """
        Vectorized stress evaluation.

        Accepts NumPy arrays of any shape.
        """

        eps = np.asarray(
            eps,
            dtype=float,
        )

        sigma = np.zeros_like(
            eps,
            dtype=float,
        )

        # ----------------------------------------------------------
        # Compression plateau
        # ----------------------------------------------------------

        compression_plateau = (
            eps <= self.eps_cu
        )

        sigma[compression_plateau] = (
            self._sigma_cu
        )

        # ----------------------------------------------------------
        # Nonlinear compression branch
        # ----------------------------------------------------------

        compression_nonlinear = (
            (eps > self.eps_cu)
            &
            (eps < 0.0)
        )

        if np.any(compression_nonlinear):
            sigma[compression_nonlinear] = (
                self._stress_compression(
                    eps[compression_nonlinear]
                )
            )

        # ==========================================================
        # TENSION ACCORDING TO QUAST
        # ==========================================================

        if self.fct > 0.0:
            # ------------------------------------------------------
            # Parabolic tension branch
            # ------------------------------------------------------
            tension_parabola = ((eps >= 0.0)& (eps < self.eps_cr))
            if np.any(tension_parabola):
                eta = (eps[tension_parabola]/ self.eps_cr)
                sigma[tension_parabola] = (
                    self.fct* (1.0-(1.0 - eta)** self.n_tension))
            # ------------------------------------------------------
            # Rectangle branch
            # ------------------------------------------------------
            tension_plateau = (eps >= self.eps_cr)
            sigma[tension_plateau] = (self.fct)

        return sigma

    def tangent_array(self, eps):
        """
        Vectorized tangent modulus evaluation.

        Accepts NumPy arrays of any shape.
        """

        eps = np.asarray(eps,dtype=float,)
        tangent = np.zeros_like(eps,dtype=float,)
        # ----------------------------------------------------------
        # Nonlinear compression branch
        # ----------------------------------------------------------
        compression_nonlinear = (
            (eps > self.eps_cu)
            &
            (eps < 0.0)
        )
        if np.any(compression_nonlinear):
            tangent[compression_nonlinear] = (
                self._tangent_compression(
                    eps[compression_nonlinear]))

        # Compression plateau:
        #
        # eps <= eps_cu
        # tangent remains zero.

        # ==========================================================
        # TENSION ACCORDING TO QUAST
        # ==========================================================

        if self.fct > 0.0:

            tension_parabola = (
                (eps >= 0.0)
                &
                (eps < self.eps_cr)
            )

            if np.any(tension_parabola):

                eta = (
                    eps[tension_parabola]
                    / self.eps_cr
                )

                tangent[tension_parabola] = (
                    self.fct
                    * self.n_tension
                    / self.eps_cr
                    * (1.0 - eta)
                    ** (self.n_tension - 1.0)
                )

            # eps >= eps_cr:
            #
            # tangent remains zero because
            # sigma = fct = constant

        return tangent

    # ==============================================================
    # Representation
    # ==============================================================

    def __repr__(self):
        return (
            f"ConcreteEC2Nonlinear("
            f"fc={self.fc:g}, "
            f"Ecm={self.Ecm:g}, "
            f"eps_c1={self.eps_c1:g}, "
            f"eps_cu={self.eps_cu:g}, "
            f"k={self.k:g}, "
            f"fct={self.fct:g}, "
            #f"Et={self.Et:g}, "
            f"eps_cr={self.eps_cr:g}, "
            f"n_tension={self.n_tension:g}, "
            #f"eps_tu={self.eps_tu!r}, "
            f"sigma_cu={self._sigma_cu:g}, "
            f"name={self.name!r}"
            f")"
        )