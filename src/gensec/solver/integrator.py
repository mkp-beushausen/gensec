# ---------------------------------------------------------------------------
# GenSec — Copyright (c) 2026 Andrea Albero
#
# This file is part of GenSec.
#
# GenSec is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# GenSec is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public
# License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with GenSec.  If not, see <https://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------

r"""
Fiber integrator and equilibrium solver — biaxial bending.

Supports both single-material and multi-material sections. When
the section has ``mat_indices`` (from :class:`GenericSection` with
``bulk_materials`` zones), the integrator groups fibers by material
to call each constitutive law on its own subset. For single-material
sections, the fast vectorized path is used unchanged.

The strain plane has 3 parameters
:math:`(\varepsilon_0, \chi_x, \chi_y)` and the internal forces
are :math:`(N, M_x, M_y)`.

Uniaxial bending is the special case :math:`\chi_y = 0`.

Performance features
--------------------
- **Batch integration** (:meth:`integrate_batch`): evaluate many
  strain configurations in a single vectorized call, eliminating
  Python-loop overhead in capacity-surface generators.
- **Analytical tangent stiffness** (:meth:`integrate_with_tangent`):
  compute internal forces *and* the 3×3 tangent matrix in one pass,
  halving the cost of each Newton-Raphson iteration compared to a
  finite-difference Jacobian.
"""

import numpy as np


class FiberSolver:
    r"""
    Material-agnostic fiber solver for biaxial bending + axial force.

    Strain plane:

    .. math::

        \varepsilon(x, y) = \varepsilon_0
            + \chi_x \, (y - y_{\text{ref}})
            + \chi_y \, (x - x_{\text{ref}})

    Internal forces:

    .. math::

        N   &= \sum_i \sigma_i \, A_i \\
        M_x &= \sum_i \sigma_i \, A_i \, (y_i - y_{\text{ref}}) \\
        M_y &= \sum_i \sigma_i \, A_i \, (x_i - x_{\text{ref}})

    Convention:

    - :math:`M_x > 0` compresses the bottom edge (:math:`y = 0`),
      tensions the top edge. Right-hand rule: thumb along +x,
      fingers curl from +y toward observer.
    - :math:`M_y > 0` compresses the right edge (:math:`x = x_{\max}`),
      tensions the left edge. Right-hand rule: thumb along +y,
      fingers curl from +x toward observer.

    Parameters
    ----------
    section : GenericSection or RectSection
        Any section exposing ``x_fibers``, ``y_fibers``, ``A_fibers``,
        ``x_rebars``, ``y_rebars``, ``A_rebars``, ``embedded_rebars``,
        ``bulk_material``, ``rebars``, ``x_centroid``, ``y_centroid``.
    x_ref : float or None, optional
        Reference x-coordinate [mm]. Default: x-centroid.
    y_ref : float or None, optional
        Reference y-coordinate [mm]. Default: y-centroid.
    """

    def __init__(self, section, x_ref=None, y_ref=None):
        self.sec = section
        self.x_ref = x_ref if x_ref is not None else section.x_centroid
        self.y_ref = y_ref if y_ref is not None else section.y_centroid
        self._rebar_groups = self._build_rebar_groups()

        # Pre-compute lever arms (constant for a given section)
        self._ly_bulk = self.sec.y_fibers - self.y_ref
        self._lx_bulk = self.sec.x_fibers - self.x_ref
        self._ly_rebar = self.sec.y_rebars - self.y_ref
        self._lx_rebar = self.sec.x_rebars - self.x_ref

        # Tendon lever arms and locked-in initial strains.  Sections
        # without tendons (legacy or non-prestressed) expose empty
        # arrays from ``_setup_tendons``; ``getattr`` keeps the solver
        # tolerant of sections built before the prestress phase.
        self._y_tendon = getattr(self.sec, 'y_tendons',
                                 np.empty(0, dtype=float))
        self._x_tendon = getattr(self.sec, 'x_tendons',
                                 np.empty(0, dtype=float))
        self._ly_tendon = self._y_tendon - self.y_ref
        self._lx_tendon = self._x_tendon - self.x_ref
        self._eps_init_tendon = getattr(
            self.sec, 'eps_init_tendons', np.empty(0, dtype=float))
        self._has_tendons = len(self._y_tendon) > 0
        self._tendon_groups = (self._build_tendon_groups()
                               if self._has_tendons else [])

        # Build bulk material groups for multi-material support
        self._bulk_groups = self._build_bulk_groups()
        self._is_multi_material = len(self._bulk_groups) > 1

        # ---- Bulk imposed-strain offset (coazione) ----
        # Uniform locked-in bulk strain (shrinkage / differential
        # thermal), read through getattr so sections built before the
        # prestress phase are bit-identical (offset 0.0 -> argument
        # unchanged everywhere).  The bulk law is evaluated at
        # (eps + offset), exactly as a bonded tendon's law is
        # evaluated at (eps_section + eps_pe).
        self._bulk_eps_init = float(
            getattr(self.sec, 'bulk_eps_init', 0.0))
        # Per-fiber / per-element imposed-strain offset fields
        # (Phase 8): the uniform scalar above plus the per-zone
        # locked-in datum planes carried by the view
        # (``bulk_planes_active``).  ``None`` on the fast path — the
        # consumption sites then add the plain scalar, keeping the
        # legacy numeric stream unchanged (never ``eb + zeros(n)``).
        self._build_offset_fields()

    def _build_rebar_groups(self):
        r"""
        Group rebar layers by the pair
        ``(rebar material, bulk-zone material)``.

        For multi-material sections (a confined core inside an
        unconfined cover, for example), an embedded rebar must
        subtract the bulk stress evaluated with the constitutive
        law of the **zone the rebar physically occupies**, not the
        primary ``bulk_material``.  Splitting the rebar groups by
        this pair lets the integrator vectorise the lookup with no
        per-rebar branching at evaluation time.

        For single-material sections (no zones, or every rebar in
        zone ``0``) this collapses to the legacy "group by rebar
        material" behaviour, with the primary ``bulk_material``
        attached to every group.

        Returns
        -------
        list of tuple
            Each tuple is
            ``(rebar_material, bulk_zone_material,
            ndarray_of_rebar_indices)``.
        """
        sec = self.sec
        # Index-aligned list of candidate bulk-zone materials.
        all_bulk_mats = (sec.get_all_bulk_materials()
                         if hasattr(sec, 'get_all_bulk_materials')
                         else [sec.bulk_material])
        # Per-rebar zone index.  Sections without
        # ``mat_indices_rebar`` (legacy ``RectSection``) place every
        # rebar in zone 0.
        mat_idx_r = getattr(sec, 'mat_indices_rebar', None)
        if mat_idx_r is None:
            mat_idx_r = np.zeros(len(sec.rebars), dtype=int)

        groups = {}
        for i, r in enumerate(sec.rebars):
            key = (id(r.material), int(mat_idx_r[i]))
            if key not in groups:
                bulk_mat = all_bulk_mats[mat_idx_r[i]]
                groups[key] = (r.material, bulk_mat, [])
            groups[key][2].append(i)
        return [(rm, bm, np.array(ix))
                for rm, bm, ix in groups.values()]

    def _build_tendon_groups(self):
        r"""
        Group tendons by the pair
        ``(tendon material, bulk-zone material)``.

        Identical zoning logic to :meth:`_build_rebar_groups`: a
        bonded tendon inside a confined core must subtract the bulk
        stress evaluated with the constitutive law of the zone it
        physically occupies.  Splitting by ``(tendon mat, zone mat)``
        vectorises the lookup with no per-tendon branching.

        Unlike rebars, the tendon's own law is evaluated at the
        **offset** total strain (section strain + locked-in initial
        strain), while the displaced-bulk subtraction uses the
        **section** strain — the two arguments differ.  The grouping
        carries the per-tendon initial-strain offset alongside the
        indices so the integrator can apply it group-wise.

        Returns
        -------
        list of tuple
            Each tuple is
            ``(tendon_material, bulk_zone_material,
            ndarray_of_tendon_indices, ndarray_of_eps_init)``.
        """
        sec = self.sec
        all_bulk_mats = (sec.get_all_bulk_materials()
                         if hasattr(sec, 'get_all_bulk_materials')
                         else [sec.bulk_material])
        mat_idx_t = getattr(sec, 'mat_indices_tendon', None)
        if mat_idx_t is None:
            mat_idx_t = np.zeros(len(sec.tendons), dtype=int)

        groups = {}
        for i, t in enumerate(sec.tendons):
            key = (id(t.material), int(mat_idx_t[i]))
            if key not in groups:
                bulk_mat = all_bulk_mats[mat_idx_t[i]]
                groups[key] = (t.material, bulk_mat, [])
            groups[key][2].append(i)

        out = []
        for rm, bm, ix in groups.values():
            ix_arr = np.array(ix)
            out.append((rm, bm, ix_arr,
                        self._eps_init_tendon[ix_arr]))
        return out

    def _build_bulk_groups(self):
        r"""
        Group bulk fibers by material index.

        For single-material sections (no ``mat_indices`` attribute or
        all indices == 0), returns a single group covering all fibers.

        Returns
        -------
        list of tuple
            Each tuple is ``(Material, ndarray_of_fiber_indices)``.
        """
        sec = self.sec

        # Check if section has multi-material zones
        mat_indices = getattr(sec, 'mat_indices', None)
        if mat_indices is None or np.all(mat_indices == 0):
            # Single material: one group covering all fibers
            idx = np.arange(sec.n_fibers)
            return [(sec.bulk_material, idx)]

        # Multi-material: group by index
        all_mats = sec.get_all_bulk_materials()
        groups = []
        for mi in np.unique(mat_indices):
            idx = np.where(mat_indices == mi)[0]
            groups.append((all_mats[mi], idx))
        return groups

    def _build_offset_fields(self):
        r"""
        Precompute the imposed-strain offset fields (Phase 8).

        The bulk constitutive law of fiber *i* (zone
        :math:`z(i) = \texttt{mat\_indices}[i]`) is evaluated at the
        offset argument
        :math:`\varepsilon_{\mathrm{sec},i} +
        \varepsilon^{\mathrm{off}}_i` with

        .. math::

           \varepsilon^{\mathrm{off}}_i \;=\; \varepsilon_{b,0}
           \; + \; \varepsilon_{0,z(i)}
           \; + \; \chi_{x,z(i)}\,(y_i - y_{\mathrm{ref}})
           \; - \; \chi_{y,z(i)}\,(x_i - x_{\mathrm{ref}})

        — the exact sign convention of :meth:`strain_field` — where
        :math:`\varepsilon_{b,0}` is the legacy uniform scalar and
        :math:`(\varepsilon_{0,z}, \chi_{x,z}, \chi_{y,z})` the
        per-zone locked-in plane from the view attribute
        ``bulk_planes_active`` (attached by
        :func:`~gensec.solver.section_state.materialize_view`).
        Companion fields are evaluated at the rebar and tendon
        coordinates for the displaced-bulk subtraction: the displaced
        concrete of zone *z* carries zone *z*'s plane at the element
        point, with the zone given by the **geometric** containment
        (``mat_indices_rebar`` / ``mat_indices_tendon`` — a staging
        ``parent`` override never touches the physics).

        **Fast path**: with no planes (or all-zero planes) every field
        is ``None`` and the consumption sites add the plain scalar
        ``_bulk_eps_init`` — byte-identity with the pre-Phase-8
        pipeline by construction (the NumPy instruction stream is
        unchanged), not by IEEE luck.

        Raises
        ------
        ValueError
            ``bulk_planes_active`` present with a shape inconsistent
            with the section's zone indices.
        """
        planes = getattr(self.sec, 'bulk_planes_active', None)
        if planes is None or not np.any(
                np.asarray(planes, dtype=float)):
            self._bulk_eps_field = None
            self._rebar_eps_field = None
            self._tendon_eps_field = None
            return
        planes = np.asarray(planes, dtype=float)
        mi = getattr(self.sec, 'mat_indices', None)
        if mi is None:
            mi = np.zeros(int(self.sec.n_fibers), dtype=int)
        mi = np.asarray(mi, dtype=int)
        mir = getattr(self.sec, 'mat_indices_rebar', None)
        if mir is None:
            mir = np.zeros(self._ly_rebar.size, dtype=int)
        mir = np.asarray(mir, dtype=int)
        mit = getattr(self.sec, 'mat_indices_tendon', None)
        if mit is None:
            mit = np.zeros(self._ly_tendon.size, dtype=int)
        mit = np.asarray(mit, dtype=int)
        zmax = max([int(mi.max(initial=0)),
                    int(mir.max(initial=0)),
                    int(mit.max(initial=0))])
        if planes.ndim != 2 or planes.shape[1] != 3 \
                or planes.shape[0] <= zmax:
            raise ValueError(
                f"bulk_planes_active has shape {planes.shape}; "
                f"expected (n_zones >= {zmax + 1}, 3) for this "
                f"section's zone indices."
            )
        eb0 = self._bulk_eps_init
        self._bulk_eps_field = (eb0 + planes[mi, 0]
                                + planes[mi, 1] * self._ly_bulk
                                - planes[mi, 2] * self._lx_bulk)
        self._rebar_eps_field = (eb0 + planes[mir, 0]
                                 + planes[mir, 1] * self._ly_rebar
                                 - planes[mir, 2] * self._lx_rebar)
        self._tendon_eps_field = (eb0 + planes[mit, 0]
                                  + planes[mit, 1] * self._ly_tendon
                                  - planes[mit, 2] * self._lx_tendon)

    def _bulk_offset(self, idx=None):
        r"""
        Imposed-strain offset at bulk fibers.

        Parameters
        ----------
        idx : numpy.ndarray or None, optional
            Fiber subset; ``None`` for all fibers.

        Returns
        -------
        float or numpy.ndarray
            The scalar ``_bulk_eps_init`` on the fast path (fields
            ``None``), else the per-fiber field (sliced).  Either
            broadcasts transparently against 1-D and batch strain
            arrays at every consumption site.
        """
        if self._bulk_eps_field is None:
            return self._bulk_eps_init
        return (self._bulk_eps_field if idx is None
                else self._bulk_eps_field[idx])

    def _rebar_offset(self, idx):
        r"""
        Displaced-bulk offset at rebar locations (fiber-subset *idx*).

        Returns
        -------
        float or numpy.ndarray
        """
        if self._rebar_eps_field is None:
            return self._bulk_eps_init
        return self._rebar_eps_field[idx]

    def _tendon_offset(self, idx):
        r"""
        Displaced-bulk offset at tendon locations (subset *idx*).

        Returns
        -------
        float or numpy.ndarray
        """
        if self._tendon_eps_field is None:
            return self._bulk_eps_init
        return self._tendon_eps_field[idx]

    # ==================================================================
    #  Single-configuration integration
    # ==================================================================

    def strain_field(self, eps0, chi_x, chi_y=0.0):
        r"""
        Compute strains at all fibers.

        .. math::

            \varepsilon_i = \varepsilon_0
                + \chi_x (y_i - y_{\text{ref}})
                - \chi_y (x_i - x_{\text{ref}})

        The minus sign on :math:`\chi_y` ensures that positive
        :math:`M_y` compresses the right edge (:math:`x = x_{\max}`),
        consistent with the right-hand rule around the y-axis.

        Parameters
        ----------
        eps0 : float
            Strain at the reference point.
        chi_x : float
            Curvature about the x-axis [1/mm].
        chi_y : float, optional
            Curvature about the y-axis [1/mm]. Default 0.

        Returns
        -------
        eps_bulk : numpy.ndarray
        eps_rebars : numpy.ndarray
        """
        eb = eps0 + chi_x * self._ly_bulk - chi_y * self._lx_bulk
        er = eps0 + chi_x * self._ly_rebar - chi_y * self._lx_rebar
        return eb, er

    def _quast_vmb_from_reference_strain(
        self,
        bulk_material,
        eps_c_ref,
    ):
        """
        Compute Quast VMB and its analytical derivative.

        Parameters
        ----------
        bulk_material : Material
            Concrete material containing:
                eps_cr
                eps_vmb_u
                n_vmb
                use_vmb

        eps_c_ref : float
            Concrete strain at the position of the governing
            reinforcing-steel fiber.

            Positive strain = tension.

        Returns
        -------
        vmb : float
            Reduction factor in [0, 1].

        dvmb_deps_ref : float
            Analytical derivative of VMB with respect to eps_c_ref.
        """

        # ------------------------------------------------------
        # VMB disabled
        # ------------------------------------------------------

        if not getattr(
            bulk_material,
            "use_vmb",
            False,
        ):
            return 1.0, 0.0

        eps_cr = float(
            bulk_material.eps_cr
        )

        eps_u = float(
            bulk_material.eps_vmb_u
        )

        n_vmb = float(
            bulk_material.n_vmb
        )

        eps_c_ref = float(
            eps_c_ref
        )

        # ------------------------------------------------------
        # Before cracking
        # ------------------------------------------------------

        if eps_c_ref <= eps_cr:
            return 1.0, 0.0

        # ------------------------------------------------------
        # End of tension stiffening
        # ------------------------------------------------------

        if eps_c_ref >= eps_u:
            return 0.0, 0.0

        # ------------------------------------------------------
        # Quast transition region
        # ------------------------------------------------------

        ratio = (
            (eps_u - eps_c_ref)
            / (eps_u - eps_cr)
        )

        ratio = float(
            np.clip(
                ratio,
                0.0,
                1.0,
            )
        )

        vmb = (
            ratio ** n_vmb
        )

        dvmb_deps_ref = (
            -n_vmb
            / (eps_u - eps_cr)
            * ratio ** (
                n_vmb - 1.0
            )
        )

        return (
            float(vmb),
            float(dvmb_deps_ref),
        )

    def _compute_quast_state(
        self,
        eps_rebar,
        bulk_material=None,):
        """
        Determine the complete Quast VMB state.

        The most strongly tensioned embedded reinforcing-steel fiber
        of the considered concrete material determines the reference
        position.

        VMB itself is calculated from the concrete strain at exactly
        this reinforcement position.
        """

        if bulk_material is None:
            bulk_material = self.sec.bulk_material

        default_state = {
            "vmb": 1.0,
            "dvmb_deps_ref": 0.0,
            "eps_c_ref": 0.0,
            "governing_index": None,
        }

        if not getattr(
                bulk_material,
                "use_vmb",
                False):
            return default_state

        eps_rebar = np.asarray(
            eps_rebar,
            dtype=float,
        )

        if eps_rebar.size == 0:
            return default_state

        embedded = np.asarray(
            self.sec.embedded_rebars,
            dtype=bool,
        )

        governing_index = None
        governing_steel_strain = -np.inf

        # ==========================================================
        # 1. Maßgebende Zugbewehrung im GESAMTEN Querschnitt suchen
        # ==========================================================

        for (
            rebar_material,
            zone_bulk_material,
            idx,
        ) in self._rebar_groups:

            if zone_bulk_material is not bulk_material:
                continue

            idx = np.asarray(
                idx,
                dtype=int,
            )

            active_idx = idx[
                embedded[idx]
            ]

            if active_idx.size == 0:
                continue

            local_position = int(
                np.argmax(
                    eps_rebar[active_idx]
                )
            )

            candidate_index = int(
                active_idx[local_position]
            )

            candidate_strain = float(
                eps_rebar[candidate_index]
            )

            if (
                candidate_strain
                > governing_steel_strain
            ):
                governing_steel_strain = (
                    candidate_strain
                )

                governing_index = (
                    candidate_index
                )

        # ==========================================================
        # 2. Erst NACH allen Gruppen prüfen
        # ==========================================================

        if governing_index is None:
            return default_state

        # ==========================================================
        # 3. Betondehnung an der maßgebenden Bewehrung
        # ==========================================================

        idx_ref = np.array(
            [governing_index],
            dtype=int,
        )

        rebar_offset = self._rebar_offset(
            idx_ref
        )

        if np.ndim(rebar_offset) == 0:
            eps_offset_ref = float(
                rebar_offset
            )

        else:
            eps_offset_ref = float(
                np.asarray(
                    rebar_offset,
                    dtype=float,
                ).ravel()[0]
            )

        eps_c_ref = float(
            eps_rebar[governing_index]
            + eps_offset_ref
        )

        # ==========================================================
        # 4. VMB bestimmen
        # ==========================================================

        (
            vmb,
            dvmb_deps_ref,
        ) = self._quast_vmb_from_reference_strain(
            bulk_material=bulk_material,
            eps_c_ref=eps_c_ref,
        )

        return {
            "vmb": float(vmb),
            "dvmb_deps_ref": float(
                dvmb_deps_ref
            ),
            "eps_c_ref": eps_c_ref,
            "governing_index": (
                governing_index
            ),
        }

    def _compute_quast_state_batch(
        self,
        eps_rebar,
        bulk_material=None,):
        """
        Determine the Quast VMB state for multiple strain configurations.

        Parameters
        ----------
        eps_rebar : numpy.ndarray
            Rebar strains with shape (n_configs, n_rebars).

        bulk_material : Material or None
            Concrete material for which VMB is evaluated.

        Returns
        -------
        dict
            vmb : ndarray, shape (n_configs,)
            dvmb_deps_ref : ndarray, shape (n_configs,)
            eps_c_ref : ndarray, shape (n_configs,)
            governing_index : ndarray, shape (n_configs,)
        """

        if bulk_material is None:
            bulk_material = self.sec.bulk_material

        eps_rebar = np.asarray(
            eps_rebar,
            dtype=np.float64,
        )

        if eps_rebar.ndim != 2:
            raise ValueError(
                "eps_rebar must have shape "
                "(n_configs, n_rebars)."
            )

        n_configs = eps_rebar.shape[0]

        default_state = {
            "vmb": np.ones(
                n_configs,
                dtype=np.float64,
            ),
            "dvmb_deps_ref": np.zeros(
                n_configs,
                dtype=np.float64,
            ),
            "eps_c_ref": np.zeros(
                n_configs,
                dtype=np.float64,
            ),
            "governing_index": np.full(
                n_configs,
                -1,
                dtype=int,
            ),
        }

        if not getattr(
            bulk_material,
            "use_vmb",
            False,
        ):
            return default_state

        if eps_rebar.shape[1] == 0:
            return default_state

        embedded = np.asarray(
            self.sec.embedded_rebars,
            dtype=bool,
        )

        # ======================================================
        # Candidate rebars belonging to this concrete material
        # ======================================================

        candidate_indices = []

        for (
            rebar_material,
            zone_bulk_material,
            idx,
        ) in self._rebar_groups:

            if zone_bulk_material is not bulk_material:
                continue

            idx = np.asarray(
                idx,
                dtype=int,
            )

            active_idx = idx[
                embedded[idx]
            ]

            if active_idx.size:
                candidate_indices.extend(
                    active_idx.tolist()
                )

        if not candidate_indices:
            return default_state

        candidate_indices = np.asarray(
            candidate_indices,
            dtype=int,
        )

        # ======================================================
        # Governing rebar for every configuration
        # ======================================================

        candidate_strains = eps_rebar[
            :,
            candidate_indices,
        ]

        local_governing = np.argmax(
            candidate_strains,
            axis=1,
        )

        governing_index = candidate_indices[
            local_governing
        ]

        config_index = np.arange(
            n_configs,
            dtype=int,
        )

        # ======================================================
        # Concrete strain at governing rebar position
        # ======================================================

        all_rebar_indices = np.arange(
            eps_rebar.shape[1],
            dtype=int,
        )

        rebar_offset = self._rebar_offset(
            all_rebar_indices
        )

        if np.ndim(rebar_offset) == 0:

            eps_offset_ref = np.full(
                n_configs,
                float(rebar_offset),
                dtype=np.float64,
            )

        else:

            rebar_offset = np.asarray(
                rebar_offset,
                dtype=np.float64,
            )

            eps_offset_ref = rebar_offset[
                governing_index
            ]

        eps_c_ref = (
            eps_rebar[
                config_index,
                governing_index,
            ]
            + eps_offset_ref
        )

        # ======================================================
        # Vectorized VMB evaluation
        # ======================================================

        eps_cr = float(
            bulk_material.eps_cr
        )

        eps_u = float(
            bulk_material.eps_vmb_u
        )

        n_vmb = float(
            bulk_material.n_vmb
        )

        if eps_u <= eps_cr:
            raise ValueError(
                "eps_vmb_u must be greater than eps_cr."
            )

        if n_vmb <= 0.0:
            raise ValueError(
                "n_vmb must be greater than zero."
            )

        vmb = np.ones(
            n_configs,
            dtype=np.float64,
        )

        dvmb_deps_ref = np.zeros(
            n_configs,
            dtype=np.float64,
        )

        transition = (
            (eps_c_ref > eps_cr)
            & (eps_c_ref < eps_u)
        )

        ended = (
            eps_c_ref >= eps_u
        )

        ratio = (
            (eps_u - eps_c_ref[transition])
            / (eps_u - eps_cr)
        )

        ratio = np.clip(
            ratio,
            0.0,
            1.0,
        )

        vmb[transition] = (
            ratio ** n_vmb
        )

        dvmb_deps_ref[transition] = (
            -n_vmb
            / (eps_u - eps_cr)
            * ratio ** (
                n_vmb - 1.0
            )
        )

        vmb[ended] = 0.0

        return {
            "vmb": vmb,
            "dvmb_deps_ref": dvmb_deps_ref,
            "eps_c_ref": eps_c_ref,
            "governing_index": governing_index,
        }

    @staticmethod
    def _apply_quast_vmb(
        material,
        strain,
        stress,
        tangent=None,
        vmb=1.0,
    ):
        """
        Apply VMB only to the tensile concrete response.

        Compression remains unchanged. The tangent returned here only
        contains the local term VMB * E_target. The global VMB coupling
        term is added separately to the section tangent matrix.
        """

        strain = np.asarray(
            strain,
            dtype=float,
        )

        stress_reduced = np.asarray(
            stress,
            dtype=float,
        ).copy()

        if tangent is not None:
            tangent_reduced = np.asarray(
                tangent,
                dtype=float,
            ).copy()
        else:
            tangent_reduced = None

        if not getattr(
            material,
            "use_vmb",
            False,
        ):
            if tangent_reduced is None:
                return stress_reduced

            return (
                stress_reduced,
                tangent_reduced,
            )

        vmb = float(
            np.clip(
                vmb,
                0.0,
                1.0,
            )
        )

        tension = strain > 0.0

        stress_reduced[tension] *= vmb

        if tangent_reduced is not None:
            tangent_reduced[tension] *= vmb

            return (
                stress_reduced,
                tangent_reduced,
            )

        return stress_reduced

    # ==================================================================
    #  Tendon contribution (prestress) — decoupled-argument core
    # ==================================================================
    @staticmethod
    def _tendon_eps_ud_limits(
        material,
        eps_init,
        use_german_eps_ud=False,
    ):
        """
        Return the permitted total tensile strain for each tendon.

        Priority:
        1. Explicit eps_ud from the material definition.
        2. German rule:
        min(0.9 * eps_uk, eps_init + 0.025)
        3. EC2 base value 0.9 * eps_uk.
        """

        eps_init = np.asarray(
            eps_init,
            dtype=float,
        )

        ec2 = getattr(
            material,
            "ec2",
            None,
        )

        # No attached EC2 property object
        if ec2 is None:
            return np.full_like(
                eps_init,
                float(material.eps_ud),
            )

        # Explicit user input has highest priority
        if getattr(
            ec2,
            "eps_ud_explicit",
            False,
        ):
            return np.full_like(
                eps_init,
                float(ec2.eps_ud),
            )

        eps_ud_ec2 = float(
            getattr(
                ec2,
                "eps_ud_ec2",
                ec2.eps_ud,
            )
        )

        if not use_german_eps_ud:
            return np.full_like(
                eps_init,
                eps_ud_ec2,
            )

        return np.minimum(
            eps_ud_ec2,
            eps_init + 0.025,
        )



    def _tendon_section_strain(self, eps0, chi_x, chi_y=0.0):
        r"""
        Section strain at each tendon location (no prestrain offset).

        .. math::

            \varepsilon_{\text{sec},i} = \varepsilon_0
                + \chi_x (y_i - y_{\text{ref}})
                - \chi_y (x_i - x_{\text{ref}})

        This is the strain the **displaced bulk** sees.  The tendon's
        own law is evaluated at this value plus the locked-in initial
        strain; see :meth:`_tendon_forces`.

        Parameters
        ----------
        eps0, chi_x : float
        chi_y : float, optional

        Returns
        -------
        numpy.ndarray
            Section strain at tendons, shape ``(n_tendons,)``.
        """
        return (eps0 + chi_x * self._ly_tendon
                - chi_y * self._lx_tendon)

    def _tendon_forces(self, et_sec, want_tangent=False, quast_state_by_material=None):
        r"""
        Accumulate tendon force/moment contributions (and tangent).

        Implements the hard correctness invariant for bonded
        prestress: the tendon evaluates its own constitutive law at
        the **offset total strain**

        .. math::

            \varepsilon_{\text{tot},i} =
                \varepsilon_{\text{sec},i} + \varepsilon_{\text{init},i}

        while the **displaced bulk** is evaluated at the section
        strain :math:`\varepsilon_{\text{sec},i}` alone.  The net
        embedded-tendon stress is therefore evaluated at two different
        arguments:

        .. math::

            \sigma^{\text{net}}_i =
                \sigma_p(\varepsilon_{\text{sec},i}
                         + \varepsilon_{\text{init},i})
                - \sigma_{\text{bulk}}(\varepsilon_{\text{sec},i})

        Because :math:`\varepsilon_{\text{init},i}` is a constant in
        this phase, the chain rule passes it through unchanged, so the
        net tangent modulus is

        .. math::

            E^{\text{net}}_{t,i} =
                E^{p}_t(\varepsilon_{\text{sec},i}
                        + \varepsilon_{\text{init},i})
                - E^{\text{bulk}}_t(\varepsilon_{\text{sec},i})

        evaluated at the same two arguments; the
        :math:`\boldsymbol{\varphi}\boldsymbol{\varphi}^T` outer
        product structure of the stiffness is unaffected.

        Parameters
        ----------
        et_sec : numpy.ndarray
            Section strain at tendons, shape ``(n_tendons,)`` (1-D) or
            ``(n_configs, n_tendons)`` (batch).
        want_tangent : bool, optional
            If ``True``, also accumulate the 3×3 tangent-stiffness
            contribution.  Only valid for the 1-D case.  Default
            ``False``.

        Returns
        -------
        N, Mx, My : float or numpy.ndarray
            Tendon force/moment contributions.  Scalars for the 1-D
            case, arrays of shape ``(n_configs,)`` for the batch case.
        K : numpy.ndarray, optional
            3×3 tangent-stiffness contribution.  Returned only when
            ``want_tangent`` is ``True``.

        Notes
        -----
        ``embedded`` is honoured exactly as for rebars: a
        non-embedded tendon does not displace bulk, so the bulk
        subtraction is skipped for those entries.
        """
        if quast_state_by_material is None:
            quast_state_by_material = {}
        batch = et_sec.ndim == 2
        embedded = getattr(self.sec, 'embedded_tendons',
                           np.ones(et_sec.shape[-1], dtype=bool))

        if batch:
            N = np.zeros(et_sec.shape[0])
            Mx = np.zeros(et_sec.shape[0])
            My = np.zeros(et_sec.shape[0])
        else:
            N = Mx = My = 0.0
            if want_tangent:
                K = np.zeros((3, 3))

        for mat, bulk_mat, idx, eps_init in self._tendon_groups:
            a = self.sec.A_tendons[idx]
            ly_t = self._ly_tendon[idx]
            lx_t = self._lx_tendon[idx]
            emb = embedded[idx]

            if batch:
                e_sec_g = et_sec[:, idx]
                e_tot_g = e_sec_g + eps_init[None, :]
                # Individual permissible tensile strain
                eps_ud_limit = (self._tendon_eps_ud_limits(material=mat,eps_init=eps_init,use_german_eps_ud=getattr(
                            mat, "use_german_eps_ud",False,),))
                e_tot_eval = np.minimum(
                e_tot_g, eps_ud_limit[None, :]
                )
                s_tendon = mat.stress_array(e_tot_eval)
                failed_tension = (
                    e_tot_g
                    > eps_ud_limit[None, :]
                )
                # s_tendon = s_tendon.copy()
                # s_tendon[
                #     failed_tension
                # ] = 0.0
                # Displaced bulk response
                bulk_eps_at_tendons = (
                    e_sec_g
                    + self._tendon_offset(idx)
                )
                s_bulk = bulk_mat.stress_array(
                    bulk_eps_at_tendons
                )
                # Apply the same VMB that is used for the corresponding
                # concrete fiber zone.
                state = quast_state_by_material.get(
                    id(bulk_mat)
                )
                if (
                    state is not None
                    and getattr(
                        bulk_mat,
                        "use_vmb",
                        False,
                    )
                ):
                    tensile_bulk = (
                        bulk_eps_at_tendons > 0.0
                    )
                    s_bulk = np.where(
                        tensile_bulk,
                        s_bulk
                        * state["vmb"][:, None],
                        s_bulk,
                    )
                s_net = s_tendon.copy()
                s_net[:, emb] -= (
                    s_bulk[:, emb]
                )
                fa = (
                    s_net
                    * a[None, :]
                )
                N += fa.sum(
                    axis=1
                )
                Mx += (
                    fa
                    * ly_t[None, :]
                ).sum(
                    axis=1
                )
                My -= (
                    fa
                    * lx_t[None, :]
                ).sum(
                    axis=1
                )
                continue

            e_sec_g = et_sec[idx]
            e_tot_g = e_sec_g + eps_init
            eps_ud_limit = self._tendon_eps_ud_limits(
                material=mat,
                eps_init=eps_init,
                use_german_eps_ud=getattr(
                    mat,
                    "use_german_eps_ud",
                    False,
                ),
            )
            #valid_tension = ( e_tot_g<= eps_ud_limit)
            e_tot_eval = np.minimum(e_tot_g, eps_ud_limit,)
            s_tendon = mat.stress_array(e_tot_eval)
            failed_tension = (e_tot_g > eps_ud_limit)
            bulk_eps_at_tendons = (e_sec_g+ self._tendon_offset(idx))
            s_bulk_target = bulk_mat.stress_array(
                bulk_eps_at_tendons
            )
            state = quast_state_by_material.get(
                id(bulk_mat)
            )
            if (
                state is not None
                and getattr(
                    bulk_mat,
                    "use_vmb",
                    False,
                )
            ):
                s_bulk = self._apply_quast_vmb(
                    material=bulk_mat,
                    strain=bulk_eps_at_tendons,
                    stress=s_bulk_target,
                    vmb=state["vmb"],
                )
            else:
                s_bulk = s_bulk_target
            s_net = s_tendon.copy()
            s_net[emb] -= s_bulk[emb]
            fa = s_net * a
            N += float(np.sum(fa))
            Mx += float(np.sum(fa * ly_t))
            My -= float(np.sum(fa * lx_t))

            if want_tangent:
                Et_tendon = mat.tangent_array(e_tot_eval)
                Et_tendon = Et_tendon.copy()
                Et_tendon[failed_tension] = 0.0
                Et_bulk_target = bulk_mat.tangent_array(
                    bulk_eps_at_tendons
                )
                if (
                    state is not None
                    and getattr(
                        bulk_mat,
                        "use_vmb",
                        False,
                    )
                ):
                    _, Et_bulk = self._apply_quast_vmb(
                        material=bulk_mat,
                        strain=bulk_eps_at_tendons,
                        stress=s_bulk_target,
                        tangent=Et_bulk_target,
                        vmb=state["vmb"],
                    )
                else:
                    Et_bulk = Et_bulk_target
                Et_net = Et_tendon.copy()
                Et_net[emb] -= Et_bulk[emb]
                EtA = Et_net * a
                K[0, 0] += float(np.sum(EtA))
                K[0, 1] += float(np.sum(EtA * ly_t))
                K[0, 2] -= float(np.sum(EtA * lx_t))
                K[1, 1] += float(np.sum(EtA * ly_t * ly_t))
                K[1, 2] -= float(np.sum(EtA * ly_t * lx_t))
                K[2, 2] += float(np.sum(EtA * lx_t * lx_t))

        if not batch and want_tangent:
            K[1, 0] = K[0, 1]
            K[2, 0] = K[0, 2]
            K[2, 1] = K[1, 2]
            return N, Mx, My, K
        return N, Mx, My

    def integrate(self, eps0, chi_x, chi_y=0.0):
        r"""
        Direct problem: strain parameters to internal forces.

        For embedded rebars, the bulk material contribution at the
        rebar location is subtracted to avoid double-counting:

        .. math::

            F_i = \bigl[\sigma_{\text{rebar},i}(\varepsilon_i)
                  - \sigma_{\text{bulk}}(\varepsilon_i)\bigr] \, A_{s,i}

        For multi-material sections, each fiber group is evaluated
        with its own constitutive law.

        Parameters
        ----------
        eps0 : float
        chi_x : float
            Curvature about x-axis [1/mm].
        chi_y : float, optional
            Curvature about y-axis [1/mm]. Default 0.

        Returns
        -------
        N : float
            Axial force [N]. Positive = tension.
        Mx : float
            Bending moment about x-axis [N*mm].
        My : float
            Bending moment about y-axis [N*mm].
        """
        eb, er = self.strain_field(eps0, chi_x, chi_y)

        # ==========================================================
        # Quast VMB state per concrete material
        # ==========================================================

        quast_state_by_material = {}

        for bulk_mat, _ in self._bulk_groups:

            material_id = id(
                bulk_mat
            )

            if material_id not in quast_state_by_material:

                quast_state_by_material[material_id] = (
                    self._compute_quast_state(
                        eps_rebar=er,
                        bulk_material=bulk_mat,
                    )
                )

        # ---- Bulk contribution ----
        if not self._is_multi_material:
            bulk_mat = self.sec.bulk_material
            eb_arg = (eb+ self._bulk_offset())
            sb_target = bulk_mat.stress_array(eb_arg)
            state = quast_state_by_material[id(bulk_mat)]
            # print(
            # "\nVMB DEBUG:",
            # f"eps_c_ref={state['eps_c_ref']:.8e}",
            # f"vmb={state['vmb']:.8f}",
            # f"gov_idx={state['governing_index']}",
            # f"eps_bulk_max={np.max(eb_arg):.8e}",
            # f"sigma_target_max={np.max(sb_target):.8f}",)
            sb = self._apply_quast_vmb(
                material=bulk_mat,
                strain=eb_arg,
                stress=sb_target,
                vmb=state["vmb"],)
            # print(
            #     "VMB RESULT:",
            #     f"sigma_after_max={np.max(sb):.8f}"
            # )
        else:
            sb = np.zeros_like(eb)
            for mat, idx in self._bulk_groups:
                eb_arg = (
                    eb[idx]
                    + self._bulk_offset(idx))
                sb_target = mat.stress_array(
                    eb_arg)
                state = quast_state_by_material[
                    id(mat)]

                sb[idx] = self._apply_quast_vmb(
                    material=mat,
                    strain=eb_arg,
                    stress=sb_target,
                    vmb=state["vmb"],)

        fA = sb * self.sec.A_fibers
        N = float(np.sum(fA))
        Mx = float(np.sum(fA * self._ly_bulk))
        My = -float(np.sum(fA * self._lx_bulk))

        # ---- Rebar contribution ----
        # For each (rebar material, bulk-zone material) group, subtract
        # the bulk stress evaluated with the **zone's** constitutive
        # law from the rebar stress for embedded rebars, then accumulate
        # forces and moments.  Splitting by (rebar mat, zone mat) is
        # what makes this correct for multi-material sections.
        embedded = self.sec.embedded_rebars

        for mat, bulk_mat, idx in self._rebar_groups:
            er_g = er[idx]
            s_rebar = mat.stress_array(er_g)
            bulk_eps_at_rebars = (er_g+ self._rebar_offset(idx))
            sb_target_at_rebars = (
                bulk_mat.stress_array(
                    bulk_eps_at_rebars))
            state = quast_state_by_material[
                id(bulk_mat) ]
            sb_at_rebars = self._apply_quast_vmb(
                material=bulk_mat,
                strain=bulk_eps_at_rebars,
                stress=sb_target_at_rebars,
                vmb=state["vmb"],)
            a = self.sec.A_rebars[idx]
            emb = embedded[idx]

            s_net = s_rebar.copy()
            s_net[emb] -= sb_at_rebars[emb]

            fa = s_net * a
            N += float(np.sum(fa))
            Mx += float(np.sum(fa * self._ly_rebar[idx]))
            My -= float(np.sum(fa * self._lx_rebar[idx]))

        # ---- Tendon contribution (prestress) ----
        if self._has_tendons:
            et = self._tendon_section_strain(eps0, chi_x, chi_y)
            Nt, Mxt, Myt = self._tendon_forces(et, quast_state_by_material=(quast_state_by_material),)
            N += Nt
            Mx += Mxt
            My += Myt

        return N, Mx, My

    # ==================================================================
    #  Analytical tangent stiffness
    # ==================================================================

    def integrate_with_tangent(self, eps0, chi_x, chi_y=0.0):
        r"""
        Internal forces **and** 3×3 tangent stiffness in one pass.

        The tangent stiffness matrix relates infinitesimal changes in
        the strain-plane parameters to changes in internal forces:

        .. math::

            \mathbf{K} =
            \frac{\partial (N,\,M_x,\,M_y)}
                 {\partial (\varepsilon_0,\,\chi_x,\,\chi_y)}
            = \sum_i E_{t,i} \, A_i \;
              \boldsymbol{\varphi}_i \, \boldsymbol{\varphi}_i^T

        where the shape-function vector for fiber *i* is

        .. math::

            \boldsymbol{\varphi}_i =
            \bigl[1,\; (y_i - y_{\text{ref}}),\;
                  -(x_i - x_{\text{ref}})\bigr]^T

        and :math:`E_{t,i} = d\sigma_i / d\varepsilon_i` is the
        tangent modulus at the current strain.

        Parameters
        ----------
        eps0 : float
        chi_x : float
        chi_y : float, optional

        Returns
        -------
        N, Mx, My : float
            Internal forces [N, N·mm].
        K : numpy.ndarray
            3×3 tangent stiffness matrix.
        """
        eb, er = self.strain_field(eps0, chi_x, chi_y)

        quast_state_by_material = {}

        for bulk_mat, _ in self._bulk_groups:
            material_id = id(bulk_mat)
            if material_id not in quast_state_by_material:
                quast_state_by_material[material_id] = (
                    self._compute_quast_state(
                        eps_rebar=er,
                        bulk_material=bulk_mat,))
        #print("VMB Full Section:", "state=", quast_state_by_material)

        # ---- Bulk: stress and tangent ----
        if not self._is_multi_material:

            bulk_mat = self.sec.bulk_material

            eb_arg = (
                eb
                + self._bulk_offset()
            )

            sb_target = bulk_mat.stress_array(
                eb_arg
            )

            Et_target = bulk_mat.tangent_array(
                eb_arg
            )

            state = quast_state_by_material[
                id(bulk_mat)
            ]

            # print(
            #     "VMB DEBUG |",
            #     f"idx={state['governing_index']} |",
            #     f"eps_c_ref={state['eps_c_ref']:.8e} |",
            #     f"vmb={state['vmb']:.8f} |",
            #     f"eps_max={np.max(eb_arg):.8e} |",
            #     f"sigma_before={np.max(sb_target):.8f}",
            # )

            sb, Et_b = self._apply_quast_vmb(
                material=bulk_mat,
                strain=eb_arg,
                stress=sb_target,
                tangent=Et_target,
                vmb=state["vmb"],
            )
            # print(
            #     "VMB RESULT |",
            #     f"sigma_after={np.max(sb):.8f}"
            # )
        else:
            sb = np.zeros_like(eb)
            Et_b = np.zeros_like(eb)
            for mat, idx in self._bulk_groups:
                eb_arg = (
                    eb[idx]
                    + self._bulk_offset(idx))
                sb_target_group = mat.stress_array(
                    eb_arg)
                Et_target_group = mat.tangent_array(
                    eb_arg)
                state = quast_state_by_material[
                    id(mat)]
                (
                    sb_group,
                    Et_group,
                ) = self._apply_quast_vmb(
                    material=mat,
                    strain=eb_arg,
                    stress=sb_target_group,
                    tangent=Et_target_group,
                    vmb=state["vmb"],)

                sb[idx] = sb_group
                Et_b[idx] = Et_group

        A = self.sec.A_fibers
        ly = self._ly_bulk
        lx = self._lx_bulk

        fA = sb * A
        N = float(np.sum(fA))
        Mx = float(np.sum(fA * ly))
        My = -float(np.sum(fA * lx))

        # Tangent stiffness from bulk fibers
        #   phi = [1, ly, -lx]
        #   K[i,j] = sum(Et * A * phi_i * phi_j)
        EtA = Et_b * A
        K = np.empty((3, 3))
        K[0, 0] = float(np.sum(EtA))
        K[0, 1] = float(np.sum(EtA * ly))
        K[0, 2] = -float(np.sum(EtA * lx))
        K[1, 0] = K[0, 1]
        K[1, 1] = float(np.sum(EtA * ly * ly))
        K[1, 2] = -float(np.sum(EtA * ly * lx))
        K[2, 0] = K[0, 2]
        K[2, 1] = K[1, 2]
        K[2, 2] = float(np.sum(EtA * lx * lx))
        K_vmb = np.zeros((3, 3),dtype=float)

        # ==========================================================
        # Full Quast VMB coupling tangent from bulk concrete fibers
        # ==========================================================

        for bulk_mat, idx in self._bulk_groups:
            state = quast_state_by_material[
                id(bulk_mat)]
            governing_index = state[
                "governing_index"]
            dvmb_deps_ref = state[
                "dvmb_deps_ref"]
            if (
                governing_index is None
                or dvmb_deps_ref == 0.0
            ):
                continue
            idx = np.asarray(
                idx,
                dtype=int,
            )
            strain_group = (
                eb[idx]
                + self._bulk_offset(idx)
            )
            tension = strain_group > 0.0
            if not np.any(tension):
                continue
            sigma_target_group = (
                bulk_mat.stress_array(
                    strain_group))
            idx_tension = idx[tension]
            sigma_tension = (
                sigma_target_group[tension])
            area_tension = (
                self.sec.A_fibers[
                    idx_tension])
            ly_tension = (
                self._ly_bulk[
                    idx_tension])
            lx_tension = (
                self._lx_bulk[
                    idx_tension])

            # Resultant direction generated by all target tensile stresses
            coupling_resultant = np.array([
                np.sum(
                    area_tension
                    * sigma_tension
                ),
                np.sum(
                    area_tension
                    * sigma_tension
                    * ly_tension
                ),
                -np.sum(
                    area_tension
                    * sigma_tension
                    * lx_tension
                ),
            ], dtype=float)

            # ==========================================================
            # Subtract displaced target concrete at embedded rebars
            # ==========================================================

            embedded_rebars = np.asarray(
                self.sec.embedded_rebars,
                dtype=bool,
            )

            for (
                rebar_mat,
                rebar_bulk_mat,
                rebar_idx,
            ) in self._rebar_groups:
                if rebar_bulk_mat is not bulk_mat:
                    continue
                rebar_idx = np.asarray(
                    rebar_idx,
                    dtype=int,
                )
                active_idx = rebar_idx[
                    embedded_rebars[rebar_idx]
                ]
                if active_idx.size == 0:
                    continue
                eps_bulk_rebar = (
                    er[active_idx]
                    + self._rebar_offset(active_idx)
                )
                tension_rebar = (
                    eps_bulk_rebar > 0.0
                )
                if not np.any(tension_rebar):
                    continue
                active_tension_idx = active_idx[
                    tension_rebar
                ]
                sigma_target_rebar = (
                    bulk_mat.stress_array(
                        eps_bulk_rebar[tension_rebar]
                    )
                )
                area_rebar = self.sec.A_rebars[
                    active_tension_idx
                ]
                ly_rebar = self._ly_rebar[
                    active_tension_idx
                ]
                lx_rebar = self._lx_rebar[
                    active_tension_idx
                ]
                coupling_resultant -= np.array([
                    np.sum(
                        area_rebar
                        * sigma_target_rebar
                    ),
                    np.sum(
                        area_rebar
                        * sigma_target_rebar
                        * ly_rebar
                    ),
                    -np.sum(
                        area_rebar
                        * sigma_target_rebar
                        * lx_rebar
                    ),
                ], dtype=float)

            # ==========================================================
            # Subtract displaced target concrete at embedded tendons
            # ==========================================================

            if self._has_tendons:

                embedded_tendons = np.asarray(
                    getattr(
                        self.sec,
                        "embedded_tendons",
                        np.ones(
                            len(self._ly_tendon),
                            dtype=bool,
                        ),
                    ),
                    dtype=bool,
                )

                et_sec = self._tendon_section_strain(
                    eps0,
                    chi_x,
                    chi_y,
                )

                for (
                    tendon_mat,
                    tendon_bulk_mat,
                    tendon_idx,
                    eps_init,
                ) in self._tendon_groups:

                    if tendon_bulk_mat is not bulk_mat:
                        continue

                    tendon_idx = np.asarray(
                        tendon_idx,
                        dtype=int,
                    )

                    active_idx = tendon_idx[
                        embedded_tendons[tendon_idx]
                    ]

                    if active_idx.size == 0:
                        continue

                    eps_bulk_tendon = (
                        et_sec[active_idx]
                        + self._tendon_offset(active_idx)
                    )

                    tension_tendon = (
                        eps_bulk_tendon > 0.0
                    )

                    if not np.any(tension_tendon):
                        continue

                    active_tension_idx = active_idx[
                        tension_tendon
                    ]

                    sigma_target_tendon_bulk = (
                        bulk_mat.stress_array(
                            eps_bulk_tendon[
                                tension_tendon
                            ]
                        )
                    )

                    area_tendon = self.sec.A_tendons[
                        active_tension_idx
                    ]

                    ly_tendon = self._ly_tendon[
                        active_tension_idx
                    ]

                    lx_tendon = self._lx_tendon[
                        active_tension_idx
                    ]

                    coupling_resultant -= np.array([
                        np.sum(
                            area_tendon
                            * sigma_target_tendon_bulk
                        ),
                        np.sum(
                            area_tendon
                            * sigma_target_tendon_bulk
                            * ly_tendon
                        ),
                        -np.sum(
                            area_tendon
                            * sigma_target_tendon_bulk
                            * lx_tendon
                        ),
                    ], dtype=float)

            # Strain-plane vector at the governing rebar position
            phi_ref = np.array([
                1.0,
                self._ly_rebar[
                    governing_index
                ],
                -self._lx_rebar[
                    governing_index
                ],
            ], dtype=float)

            K_vmb += (
                dvmb_deps_ref
                * np.outer(
                    coupling_resultant,
                    phi_ref,
                )
            )

        # ---- Rebar contribution ----
        # See ``integrate``: each (rebar mat, zone mat) group subtracts
        # the displaced bulk stress and tangent using the zone's law.
        embedded = self.sec.embedded_rebars

        for mat, bulk_mat, idx in self._rebar_groups:
            er_g = er[idx]
            s_rebar = mat.stress_array(er_g)
            Et_rebar = mat.tangent_array(er_g)
            bulk_eps_at_rebars = (
                er_g
                + self._rebar_offset(idx))
            sb_target_at_rebars = (
                bulk_mat.stress_array(
                    bulk_eps_at_rebars))
            Et_target_at_rebars = (
                bulk_mat.tangent_array(
                    bulk_eps_at_rebars))
            state = quast_state_by_material[
                id(bulk_mat)]

            (
                sb_at_rebars,
                Et_bulk_r,
            ) = self._apply_quast_vmb(
                material=bulk_mat,
                strain=bulk_eps_at_rebars,
                stress=sb_target_at_rebars,
                tangent=Et_target_at_rebars,
                vmb=state["vmb"],
            )
            a = self.sec.A_rebars[idx]
            emb = embedded[idx]
            ly_r = self._ly_rebar[idx]
            lx_r = self._lx_rebar[idx]

            s_net = s_rebar.copy()
            s_net[emb] -= sb_at_rebars[emb]

            Et_net = Et_rebar.copy()
            Et_net[emb] -= Et_bulk_r[emb]

            fa = s_net * a
            N += float(np.sum(fa))
            Mx += float(np.sum(fa * ly_r))
            My -= float(np.sum(fa * lx_r))

            EtA_r = Et_net * a
            K[0, 0] += float(np.sum(EtA_r))
            K[0, 1] += float(np.sum(EtA_r * ly_r))
            K[0, 2] -= float(np.sum(EtA_r * lx_r))
            K[1, 1] += float(np.sum(EtA_r * ly_r * ly_r))
            K[1, 2] -= float(np.sum(EtA_r * ly_r * lx_r))
            K[2, 2] += float(np.sum(EtA_r * lx_r * lx_r))

        # ---- Tendon contribution (prestress) ----
        # Decoupled-argument force and tangent; see _tendon_forces.
        if self._has_tendons:
            et = self._tendon_section_strain(eps0, chi_x, chi_y)
            Nt, Mxt, Myt, Kt = self._tendon_forces(
                et, want_tangent=True, quast_state_by_material=quast_state_by_material,)
            N += Nt
            Mx += Mxt
            My += Myt
            K += Kt

        # ==========================================================
        # Final tangent-matrix handling
        # ==========================================================
        #
        # Without Quast VMB the traditional fiber tangent is symmetric
        # and the original GenSec behaviour is retained exactly.
        #
        # With Quast VMB active, the additional coupling term
        #
        #     dVMB / d eps_ref
        #
        # couples all concrete tension fibers to the governing reference
        # reinforcement position. The resulting consistent tangent is
        # generally non-symmetric and must therefore NOT be symmetrized.
        # ==========================================================

        K[1, 0] = K[0, 1]
        K[2, 0] = K[0, 2]
        K[2, 1] = K[1, 2]
        K += K_vmb

        return N, Mx, My, K

    # ==================================================================
    #  Batch integration — many configurations in one NumPy call
    # ==================================================================

    def integrate_batch(
        self,
        eps0,
        chi_x,
        chi_y,
    ):
        r"""
        Evaluate internal forces for many strain configurations.

        Quast VMB is evaluated separately for every configuration
        and every concrete material zone.

        Parameters
        ----------
        eps0 : numpy.ndarray
            Shape (n,).

        chi_x : numpy.ndarray
            Shape (n,).

        chi_y : numpy.ndarray
            Shape (n,).

        Returns
        -------
        N, Mx, My : numpy.ndarray
            Internal forces for every configuration.
        """

        eps0 = np.asarray(
            eps0,
            dtype=np.float64,
        )

        chi_x = np.asarray(
            chi_x,
            dtype=np.float64,
        )

        chi_y = np.asarray(
            chi_y,
            dtype=np.float64,
        )

        if not (
            eps0.ndim == 1
            and chi_x.ndim == 1
            and chi_y.ndim == 1
        ):
            raise ValueError(
                "eps0, chi_x and chi_y must be one-dimensional."
            )

        if not (
            eps0.shape
            == chi_x.shape
            == chi_y.shape
        ):
            raise ValueError(
                "eps0, chi_x and chi_y must have equal shapes."
            )

        n_configs = eps0.size
        n_rebars = len(
            self.sec.y_rebars
        )

        # ======================================================
        # Strain fields
        # ======================================================

        eb = (
            eps0[:, None]
            + chi_x[:, None]
            * self._ly_bulk[None, :]
            - chi_y[:, None]
            * self._lx_bulk[None, :]
        )

        if n_rebars:

            er = (
                eps0[:, None]
                + chi_x[:, None]
                * self._ly_rebar[None, :]
                - chi_y[:, None]
                * self._lx_rebar[None, :]
            )

        else:

            er = np.empty(
                (n_configs, 0),
                dtype=np.float64,
            )

        # ======================================================
        # Quast state for every concrete material and state
        # ======================================================

        quast_state_by_material = {}

        for bulk_mat, _ in self._bulk_groups:

            material_id = id(
                bulk_mat
            )

            if material_id not in quast_state_by_material:

                quast_state_by_material[
                    material_id
                ] = self._compute_quast_state_batch(
                    eps_rebar=er,
                    bulk_material=bulk_mat,
                )

        # ======================================================
        # Bulk stresses
        # ======================================================

        if not self._is_multi_material:

            bulk_mat = self.sec.bulk_material

            eb_arg = (
                eb
                + self._bulk_offset()
            )

            sb = bulk_mat.stress_array(
                eb_arg
            )

            state = quast_state_by_material[
                id(bulk_mat)
            ]

            if getattr(
                bulk_mat,
                "use_vmb",
                False,
            ):

                tension = (
                    eb_arg > 0.0
                )

                sb = np.where(
                    tension,
                    sb * state["vmb"][:, None],
                    sb,
                )

        else:

            sb = np.zeros_like(
                eb,
                dtype=np.float64,
            )

            for mat, idx in self._bulk_groups:

                idx = np.asarray(
                    idx,
                    dtype=int,
                )

                eb_arg = (
                    eb[:, idx]
                    + self._bulk_offset(idx)
                )

                sb_group = mat.stress_array(
                    eb_arg
                )

                state = quast_state_by_material[
                    id(mat)
                ]

                if getattr(
                    mat,
                    "use_vmb",
                    False,
                ):

                    tension = (
                        eb_arg > 0.0
                    )

                    sb_group = np.where(
                        tension,
                        sb_group
                        * state["vmb"][:, None],
                        sb_group,
                    )

                sb[:, idx] = sb_group

        # ======================================================
        # Bulk resultants
        # ======================================================

        fA = (
            sb
            * self.sec.A_fibers[None, :]
        )

        N = fA.sum(
            axis=1
        )

        Mx = np.einsum(
            "ij,j->i",
            fA,
            self._ly_bulk,
        )

        My = -np.einsum(
            "ij,j->i",
            fA,
            self._lx_bulk,
        )

        # ======================================================
        # Rebar contribution
        # ======================================================

        if n_rebars:

            embedded = np.asarray(
                self.sec.embedded_rebars,
                dtype=bool,
            )

            for (
                mat,
                bulk_mat,
                idx,
            ) in self._rebar_groups:

                idx = np.asarray(
                    idx,
                    dtype=int,
                )

                er_g = er[:, idx]

                s_rebar = mat.stress_array(
                    er_g
                )

                bulk_eps_at_rebars = (
                    er_g
                    + self._rebar_offset(idx)
                )

                sb_at_rebars = (
                    bulk_mat.stress_array(
                        bulk_eps_at_rebars
                    )
                )

                state = quast_state_by_material[
                    id(bulk_mat)
                ]

                if getattr(
                    bulk_mat,
                    "use_vmb",
                    False,
                ):

                    tensile_bulk_at_rebars = (
                        bulk_eps_at_rebars > 0.0
                    )

                    sb_at_rebars = np.where(
                        tensile_bulk_at_rebars,
                        sb_at_rebars
                        * state["vmb"][:, None],
                        sb_at_rebars,
                    )

                a = self.sec.A_rebars[
                    idx
                ]

                emb = embedded[
                    idx
                ]

                ly_r = self._ly_rebar[
                    idx
                ]

                lx_r = self._lx_rebar[
                    idx
                ]

                s_net = s_rebar.copy()

                s_net[:, emb] -= (
                    sb_at_rebars[:, emb]
                )

                fa = (
                    s_net
                    * a[None, :]
                )

                N += fa.sum(
                    axis=1
                )

                Mx += (
                    fa
                    * ly_r[None, :]
                ).sum(
                    axis=1
                )

                My -= (
                    fa
                    * lx_r[None, :]
                ).sum(
                    axis=1
                )

        # ======================================================
        # Tendon contribution
        # ======================================================

        if self._has_tendons:

            et = (
                eps0[:, None]
                + chi_x[:, None]
                * self._ly_tendon[None, :]
                - chi_y[:, None]
                * self._lx_tendon[None, :]
            )

            Nt, Mxt, Myt = (
                self._tendon_forces(
                    et,
                    quast_state_by_material=(
                        quast_state_by_material
                    ),
                )
            )

            N += Nt
            Mx += Mxt
            My += Myt

        return N, Mx, My


    # ==================================================================
    #  Numerical Jacobian (kept for validation / fallback)
    # ==================================================================

    def jacobian(self, eps0, chi_x, chi_y=0.0, deps=1e-8):
        r"""
        Numerical 3×3 Jacobian via forward finite differences.

        .. math::

            \mathbf{J} = \frac{\partial(N, M_x, M_y)}
                              {\partial(\varepsilon_0, \chi_x, \chi_y)}

        Parameters
        ----------
        eps0, chi_x, chi_y : float
        deps : float, optional
            Perturbation. Default 1e-8.

        Returns
        -------
        numpy.ndarray
            Shape (3, 3).
        """
        f0 = np.array(self.integrate(eps0, chi_x, chi_y))
        J = np.empty((3, 3))
        for j, (de, dx, dy) in enumerate([
            (deps, 0, 0), (0, deps, 0), (0, 0, deps)
        ]):
            f1 = np.array(self.integrate(
                eps0 + de, chi_x + dx, chi_y + dy))
            J[:, j] = (f1 - f0) / deps
        return J

    def _is_uniaxial(self, axis=None):
        r"""
        Detect if the section is effectively uniaxial.

        A section is uniaxial when one of its in-plane extents is
        degenerate, so the corresponding curvature has no
        integration support and the matching Jacobian column is
        singular.

        Two degenerate cases are recognised:

        - **vertical-degenerate**: every fiber shares the same
          :math:`x` coordinate.  :math:`\chi_y` has no effect, so
          bending is meaningful only about the :math:`x` axis.
        - **horizontal-degenerate**: every fiber shares the same
          :math:`y` coordinate.  :math:`\chi_x` has no effect, so
          bending is meaningful only about the :math:`y` axis.

        Parameters
        ----------
        axis : ``'x'``, ``'y'`` or ``None``, optional
            If specified, the test is restricted to the matching
            degeneracy: ``'x'`` returns ``True`` only for the
            vertical-degenerate case (chi_y singular), ``'y'`` only
            for the horizontal-degenerate case.  Without ``axis``,
            the test returns ``True`` for either degeneracy —
            preserving the legacy single-argument semantics.

        Returns
        -------
        bool
        """
        # Range of lever arms along x (sensitivity to chi_y).
        if len(self._lx_rebar) > 0:
            lx_range = (np.max(np.abs(self._lx_bulk))
                        + np.max(np.abs(self._lx_rebar)))
        else:
            lx_range = np.max(np.abs(self._lx_bulk))

        # Range of lever arms along y (sensitivity to chi_x).
        if len(self._ly_rebar) > 0:
            ly_range = (np.max(np.abs(self._ly_bulk))
                        + np.max(np.abs(self._ly_rebar)))
        else:
            ly_range = np.max(np.abs(self._ly_bulk))

        vertical_deg = lx_range < 1e-6     # bending about x
        horizontal_deg = ly_range < 1e-6   # bending about y

        if axis == 'x':
            return vertical_deg
        if axis == 'y':
            return horizontal_deg
        return vertical_deg or horizontal_deg

    # ==================================================================
    #  Main solver entry point
    # ==================================================================

    def solve_equilibrium(self, N_target, Mx_target, My_target=0.0,
                          eps0_init=0.0, chi_x_init=1e-6,
                          chi_y_init=0.0,
                          tol=1e-3, max_iter=50):
        r"""
        Inverse problem: find strain plane for target (N, Mx, My).

        Newton-Raphson with **analytical tangent stiffness** and
        backtracking line search.  Automatically reduces to a 2×2
        system when the section is uniaxial, and further to a
        **1×1 bisection** when both target moments are negligible
        (pure axial load).

        The pure-axial branch exploits the monotonicity of
        :math:`N(\varepsilon_0)` at :math:`\chi_x = \chi_y = 0`:

        .. math::

            \frac{\partial N}{\partial \varepsilon_0}\bigg|_{\chi=0}
            = \sum_i \frac{\partial \sigma_i}
              {\partial \varepsilon}\,A_i \;\ge\; 0

        which guarantees unique bracketing and convergence of the
        bisection search.

        Parameters
        ----------
        N_target : float
            Target axial force [N].
        Mx_target : float
            Target moment about x-axis [N*mm].
        My_target : float, optional
            Target moment about y-axis [N*mm]. Default 0.
        eps0_init, chi_x_init, chi_y_init : float, optional
            Initial guesses.
        tol : float, optional
            Force tolerance [N]; moment tolerance is ``tol * 1000``.
        max_iter : int, optional

        Returns
        -------
        dict
            Keys: ``eps0``, ``chi_x``, ``chi_y``, ``N``, ``Mx``,
            ``My``, ``converged``, ``iterations``.
        """
        M_tol = tol * 1000  # moment tolerance [N·mm]

        # ----------------------------------------------------------
        #  Fast path: pure axial load (both moments negligible)
        # ----------------------------------------------------------
        if abs(Mx_target) < M_tol and abs(My_target) < M_tol:
            sol = self._solve_pure_axial(N_target, tol, max_iter)
            if (sol["converged"]
                    and abs(sol["Mx"]) < M_tol
                    and abs(sol["My"]) < M_tol):
                return sol
            # Fall through: a converged pure-axial (chi=0) state can
            # still violate moment equilibrium when the section carries
            # eccentric *internal* self-stress (e.g. an eccentric bonded
            # tendon at prestress transfer). chi=0 cannot balance that
            # moment, so the full uniaxial/biaxial solve is required.

        # ----------------------------------------------------------
        #  Uniaxial or near-uniaxial
        # ----------------------------------------------------------
        # Vertically-degenerate section: only chi_x is meaningful,
        # solve about the x-axis.
        if self._is_uniaxial(axis='x'):
            return self._solve_uniaxial(
                N_target, Mx_target, eps0_init, chi_x_init,
                tol, max_iter)

        # Horizontally-degenerate section: only chi_y is meaningful,
        # solve about the y-axis.
        if self._is_uniaxial(axis='y'):
            return self._solve_uniaxial_y(
                N_target, My_target, eps0_init, chi_y_init,
                tol, max_iter)

        # Near-uniaxial fast paths.  When one of the target moments
        # is negligible, try the corresponding 2-unknown solve in
        # the dominant plane and accept it only if the spurious
        # moment stays inside tolerance.  Both branches are present
        # for symmetry — previously only the My-near-zero branch
        # existed, penalising demands dominated by My.
        if abs(My_target) < M_tol:
            sol = self._solve_uniaxial(
                N_target, Mx_target, eps0_init, chi_x_init,
                tol, max_iter)
            if sol["converged"] and abs(sol["My"]) < M_tol:
                return sol

        if abs(Mx_target) < M_tol:
            sol = self._solve_uniaxial_y(
                N_target, My_target, eps0_init, chi_y_init,
                tol, max_iter)
            if sol["converged"] and abs(sol["Mx"]) < M_tol:
                return sol

        return self._solve_biaxial(
            N_target, Mx_target, My_target,
            eps0_init, chi_x_init, chi_y_init,
            tol, max_iter)

    # ------------------------------------------------------------------
    #  Pure axial solver  (1-unknown bisection)
    # ------------------------------------------------------------------

    def _solve_pure_axial(self, N_target, tol, max_iter):
        r"""
        Find :math:`\varepsilon_0` for pure axial load
        (:math:`\chi_x = \chi_y = 0`).

        The function :math:`N(\varepsilon_0)` at zero curvature is
        **not globally monotone** because the concrete constitutive
        law returns zero stress for strains beyond
        :math:`\varepsilon_{cu2}`, creating a non-monotone drop at
        the compressive end.  However, :math:`N` *is* monotonically
        non-decreasing on the sub-interval
        :math:`[\varepsilon_{c2},\, \varepsilon_{su}]` where all
        constitutive laws are active, and the vast majority of
        practical targets lie in this range.

        Strategy:

        1.  Scan :math:`N(\varepsilon_0)` on a fine grid spanning
            the full material strain range — using
            :meth:`integrate_batch` for speed.
        2.  Find consecutive grid points where :math:`N` crosses
            :math:`N_{\text{target}}`.  Among all crossings, pick
            the one in the monotone working region (closest to the
            centroid strain).
        3.  Bisect within that local bracket.

        Parameters
        ----------
        N_target : float
            Target axial force [N].
        tol : float
            Convergence tolerance on N [N].
        max_iter : int
            Maximum bisection iterations.

        Returns
        -------
        dict
        """
        sec = self.sec

        # --- Build the full strain scan range ---
        eps_lo = sec.bulk_material.eps_min
        eps_hi = 0.0
        for r in sec.rebars:
            eps_lo = min(eps_lo, r.material.eps_min)
            eps_hi = max(eps_hi, r.material.eps_max)
        eps_lo *= 1.01
        eps_hi *= 1.01

        # --- Batch scan ---
        n_scan = 120
        eps_vals = np.linspace(eps_lo, eps_hi, n_scan)
        chi_zero = np.zeros(n_scan)
        N_vals, _, _ = self.integrate_batch(eps_vals, chi_zero, chi_zero)

        # --- Find all crossings of N_target ---
        crossings = []
        for k in range(n_scan - 1):
            if ((N_vals[k] - N_target) * (N_vals[k + 1] - N_target) <= 0
                    and abs(N_vals[k + 1] - N_vals[k]) > 0.1):
                crossings.append(k)

        if not crossings:
            return self._fail_result()

        # Pick the best crossing: prefer the one closest to eps0=0
        # (i.e. in the working monotone region, not in the
        # non-physical tail beyond eps_cu2).
        best_k = min(crossings,
                     key=lambda k: abs(eps_vals[k] + eps_vals[k + 1]))

        a = eps_vals[best_k]
        b = eps_vals[best_k + 1]
        Na = N_vals[best_k]

        # --- Bisection within the local bracket ---
        for i in range(max_iter):
            mid = 0.5 * (a + b)
            N_mid, Mx_mid, My_mid = self.integrate(mid, 0.0, 0.0)
            if abs(N_mid - N_target) < tol:
                return {"eps0": mid, "chi_x": 0.0, "chi_y": 0.0,
                        "N": N_mid, "Mx": Mx_mid, "My": My_mid,
                        "converged": True, "iterations": i + 1}
            if (N_mid - N_target) * (Na - N_target) <= 0:
                b = mid
            else:
                a = mid
                Na = N_mid

        mid = 0.5 * (a + b)
        N_f, Mx_f, My_f = self.integrate(mid, 0.0, 0.0)
        converged = abs(N_f - N_target) < tol
        return {"eps0": mid, "chi_x": 0.0, "chi_y": 0.0,
                "N": N_f, "Mx": Mx_f, "My": My_f,
                "converged": converged, "iterations": max_iter}

    @staticmethod
    def _fail_result():
        """Return a non-converged result placeholder."""
        return {"eps0": 0.0, "chi_x": 0.0, "chi_y": 0.0,
                "N": 0.0, "Mx": 0.0, "My": 0.0,
                "converged": False, "iterations": 0}

    # ------------------------------------------------------------------
    #  Internal: elastic initial guess
    # ------------------------------------------------------------------

    def _elastic_initial_guess(self, N_target, Mx_target, My_target=0.0):
        r"""
        Estimate initial (eps0, chi_x, chi_y) from linear-elastic theory.

        Uses the **analytical** tangent stiffness at the origin,
        which is the elastic stiffness matrix.

        Returns
        -------
        eps0, chi_x, chi_y : float
        
        """
        ### TODO: we should use ideal gross section properties here, 
        ### to get a more accurate initial guess. 
        try:
            _, _, _, K0 = self.integrate_with_tangent(0.0, 0.0, 0.0)
            target = np.array([N_target, Mx_target, My_target])
            x0 = np.linalg.solve(K0, target)
            x0[0] = np.clip(x0[0], -0.003, 0.003)
            x0[1] = np.clip(x0[1], -1e-4, 1e-4)
            x0[2] = np.clip(x0[2], -1e-4, 1e-4)
            return float(x0[0]), float(x0[1]), float(x0[2])
        except np.linalg.LinAlgError:
            sec = self.sec
            A_ideal_gross = getattr(sec, 'ideal_gross_area',
                                    sec.B * sec.H)
            # Independent moments of inertia for each bending axis.
            I_x_approx = (A_ideal_gross * sec.H ** 2 / 12
                          if sec.H > 0 else 0.0)
            I_y_approx = (A_ideal_gross * sec.B ** 2 / 12
                          if sec.B > 0 else 0.0)
            ### TODO: substitute 30000 with real bulk base modulus of the section
            E_eff = 30000.0
            eps0_est = N_target / (A_ideal_gross * E_eff)
            chi_x_est = (Mx_target / (E_eff * I_x_approx)
                         if I_x_approx > 0 else 1e-6)
            chi_y_est = (My_target / (E_eff * I_y_approx)
                         if I_y_approx > 0 else 0.0)
            return (float(eps0_est),
                    float(chi_x_est),
                    float(chi_y_est))

    # ------------------------------------------------------------------
    #  Uniaxial solver (analytical Jacobian)
    # ------------------------------------------------------------------

    def _solve_uniaxial(self, N_target, Mx_target,
                        eps0_init, chi_x_init, tol, max_iter):
        r"""
        2-unknown solver (:math:`\chi_y` fixed at 0).

        Strategy:

        1. Try elastic initial guess → Newton-Raphson.
        2. If that fails, warm-start from a pure-axial solve at
           :math:`N = N_{\text{target}}` with a small perturbation
           in :math:`\chi_x` to break the singularity.
        3. Multi-start grid of :math:`\chi_x` values.

        Parameters
        ----------
        N_target, Mx_target : float
        eps0_init, chi_x_init : float
        tol : float
        max_iter : int

        Returns
        -------
        dict
        """
        # --- Attempt 1: user-supplied or elastic guess ---
        if eps0_init == 0.0 and chi_x_init == 1e-6:
            e0, c0, _ = self._elastic_initial_guess(
                N_target, Mx_target, 0.0)
        else:
            e0, c0 = eps0_init, chi_x_init

        sol = self._nr_uniaxial(N_target, Mx_target, e0, c0,
                                tol, max_iter)
        if sol["converged"]:
            return sol

        # --- Attempt 2: warm-start from pure-axial bisection ---
        sol_axial = self._solve_pure_axial(N_target, tol, max_iter)
        if sol_axial["converged"]:
            e0_ax = sol_axial["eps0"]
            # Estimate chi_x from the sign of Mx_target
            chi_sign = 1.0 if Mx_target >= 0 else -1.0
            for chi_mag in [1e-6, 5e-6, 1e-5, 3e-5, 5e-5]:
                sol2 = self._nr_uniaxial(
                    N_target, Mx_target,
                    e0_ax, chi_sign * chi_mag,
                    tol, max_iter)
                if sol2["converged"]:
                    return sol2

        # --- Attempt 3: multi-start grid ---
        emb = self.sec.bulk_material.eps_min
        for chi_sign in [1.0, -1.0]:
            for chi_mag in [1e-5, 5e-6, 2e-5, 3e-5, 5e-5]:
                chi_try = chi_sign * chi_mag
                try:
                    N0, _, _ = self.integrate(0.0, chi_try, 0.0)
                    N1, _, _ = self.integrate(1e-6, chi_try, 0.0)
                    dNde = (N1 - N0) / 1e-6
                    if abs(dNde) > 1:
                        eps0_try = (N_target - N0) / dNde
                        eps0_try = np.clip(eps0_try, emb, -emb)
                    else:
                        eps0_try = emb / 2
                except Exception:
                    eps0_try = emb / 2

                sol = self._nr_uniaxial(N_target, Mx_target,
                                        eps0_try, chi_try,
                                        tol, max_iter // 2)
                if sol["converged"]:
                    return sol

        return self._nr_uniaxial(N_target, Mx_target, e0, c0,
                                 tol, max_iter)

    def _nr_uniaxial(self, N_target, Mx_target, eps0, chi_x,
                     tol, max_iter):
        r"""
        Core Newton-Raphson for uniaxial case.

        Uses the **analytical** 2×2 sub-block of the tangent
        stiffness matrix instead of finite-difference perturbations.
        """
        for i in range(max_iter):
            N, Mx, My, K = self.integrate_with_tangent(eps0, chi_x, 0.0)
            r = np.array([N - N_target, Mx - Mx_target])

            if abs(r[0]) < tol and abs(r[1]) < tol * 1000:
                return {
                    "eps0": eps0, "chi_x": chi_x, "chi_y": 0.0,
                    "N": N, "Mx": Mx, "My": My,
                    "converged": True, "iterations": i + 1,
                }

            # Extract 2×2 sub-block [dN/de, dN/dc; dMx/de, dMx/dc]
            J = K[:2, :2]
            det = abs(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
            if det < 1e-20:
                eps0 += 1e-5
                chi_x += 1e-7
                continue

            try:
                d = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                eps0 += 1e-5
                chi_x += 1e-7
                continue

            # Backtracking line search
            alpha = 1.0
            r_norm = np.linalg.norm(r)
            for _ in range(15):
                e_new = eps0 + alpha * d[0]
                c_new = chi_x + alpha * d[1]
                Nn, Mxn, _ = self.integrate(e_new, c_new, 0.0)
                rn = np.array([Nn - N_target, Mxn - Mx_target])
                if np.linalg.norm(rn) < r_norm:
                    break
                alpha *= 0.5
            eps0 += alpha * d[0]
            chi_x += alpha * d[1]

        N, Mx, My = self.integrate(eps0, chi_x, 0.0)
        return {
            "eps0": eps0, "chi_x": chi_x, "chi_y": 0.0,
            "N": N, "Mx": Mx, "My": My,
            "converged": False, "iterations": max_iter,
        }

    # ------------------------------------------------------------------
    #  Uniaxial solver about the y-axis  (mirror of _solve_uniaxial)
    # ------------------------------------------------------------------

    def _solve_uniaxial_y(self, N_target, My_target,
                          eps0_init, chi_y_init, tol, max_iter):
        r"""
        2-unknown solver with :math:`\chi_x` fixed at 0.

        Mirror image of :meth:`_solve_uniaxial`, applicable when
        the bending is dominant about the :math:`y` axis (target
        :math:`M_x \approx 0`) or when the section is degenerate
        horizontally (:meth:`_is_uniaxial(axis='y')`).

        Strategy is identical to the x-axis variant: elastic guess
        → Newton, axial warm-start with sign-aware curvature, and
        a multi-start grid as last resort.

        Parameters
        ----------
        N_target, My_target : float
        eps0_init, chi_y_init : float
        tol : float
        max_iter : int

        Returns
        -------
        dict
            ``eps0``, ``chi_x`` (always 0), ``chi_y``, ``N``, ``Mx``,
            ``My``, ``converged``, ``iterations``.
        """
        # --- Attempt 1: user-supplied or elastic guess ---
        if eps0_init == 0.0 and chi_y_init == 0.0:
            e0, _, c0 = self._elastic_initial_guess(
                N_target, 0.0, My_target)
            if c0 == 0.0:
                # Elastic-fallback path may return chi_y = 0; nudge.
                c0 = 1e-6 if My_target >= 0 else -1e-6
        else:
            e0, c0 = eps0_init, chi_y_init

        sol = self._nr_uniaxial_y(N_target, My_target, e0, c0,
                                  tol, max_iter)
        if sol["converged"]:
            return sol

        # --- Attempt 2: warm-start from pure-axial bisection ---
        sol_axial = self._solve_pure_axial(N_target, tol, max_iter)
        if sol_axial["converged"]:
            e0_ax = sol_axial["eps0"]
            chi_sign = 1.0 if My_target >= 0 else -1.0
            for chi_mag in [1e-6, 5e-6, 1e-5, 3e-5, 5e-5]:
                sol2 = self._nr_uniaxial_y(
                    N_target, My_target,
                    e0_ax, chi_sign * chi_mag,
                    tol, max_iter)
                if sol2["converged"]:
                    return sol2

        # --- Attempt 3: multi-start grid ---
        emb = self.sec.bulk_material.eps_min
        for chi_sign in [1.0, -1.0]:
            for chi_mag in [1e-5, 5e-6, 2e-5, 3e-5, 5e-5]:
                chi_try = chi_sign * chi_mag
                try:
                    N0, _, _ = self.integrate(0.0, 0.0, chi_try)
                    N1, _, _ = self.integrate(1e-6, 0.0, chi_try)
                    dNde = (N1 - N0) / 1e-6
                    if abs(dNde) > 1:
                        eps0_try = (N_target - N0) / dNde
                        eps0_try = np.clip(eps0_try, emb, -emb)
                    else:
                        eps0_try = emb / 2
                except Exception:
                    eps0_try = emb / 2

                sol = self._nr_uniaxial_y(N_target, My_target,
                                          eps0_try, chi_try,
                                          tol, max_iter // 2)
                if sol["converged"]:
                    return sol

        return self._nr_uniaxial_y(N_target, My_target, e0, c0,
                                   tol, max_iter)

    def _nr_uniaxial_y(self, N_target, My_target, eps0, chi_y,
                       tol, max_iter):
        r"""
        Core Newton-Raphson for uniaxial bending about the y-axis.

        Uses the **analytical** 2×2 sub-block of the tangent
        stiffness matrix corresponding to
        :math:`(\varepsilon_0, \chi_y)`, i.e. rows/cols 0 and 2 of
        the full 3×3 tangent.  Mirror image of :meth:`_nr_uniaxial`.
        """
        for i in range(max_iter):
            N, Mx, My, K = self.integrate_with_tangent(eps0, 0.0, chi_y)
            r = np.array([N - N_target, My - My_target])

            if abs(r[0]) < tol and abs(r[1]) < tol * 1000:
                return {
                    "eps0": eps0, "chi_x": 0.0, "chi_y": chi_y,
                    "N": N, "Mx": Mx, "My": My,
                    "converged": True, "iterations": i + 1,
                }

            # Extract 2×2 sub-block [dN/de, dN/dcy; dMy/de, dMy/dcy]
            J = np.array([[K[0, 0], K[0, 2]],
                          [K[2, 0], K[2, 2]]])
            det = abs(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
            if det < 1e-20:
                eps0 += 1e-5
                chi_y += 1e-7
                continue

            try:
                d = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                eps0 += 1e-5
                chi_y += 1e-7
                continue

            # Backtracking line search
            alpha = 1.0
            r_norm = np.linalg.norm(r)
            for _ in range(15):
                e_new = eps0 + alpha * d[0]
                c_new = chi_y + alpha * d[1]
                Nn, _, Myn = self.integrate(e_new, 0.0, c_new)
                rn = np.array([Nn - N_target, Myn - My_target])
                if np.linalg.norm(rn) < r_norm:
                    break
                alpha *= 0.5
            eps0 += alpha * d[0]
            chi_y += alpha * d[1]

        N, Mx, My = self.integrate(eps0, 0.0, chi_y)
        return {
            "eps0": eps0, "chi_x": 0.0, "chi_y": chi_y,
            "N": N, "Mx": Mx, "My": My,
            "converged": False, "iterations": max_iter,
        }

    # ------------------------------------------------------------------
    #  Biaxial solver (analytical Jacobian)
    # ------------------------------------------------------------------

    def _solve_biaxial(self, N_target, Mx_target, My_target,
                       eps0_init, chi_x_init, chi_y_init,
                       tol, max_iter):
        """3-unknown solver with multi-start strategy."""
        if (eps0_init == 0.0 and chi_x_init == 1e-6
                and chi_y_init == 0.0):
            e0, cx0, cy0 = self._elastic_initial_guess(
                N_target, Mx_target, My_target)
        else:
            e0, cx0, cy0 = eps0_init, chi_x_init, chi_y_init

        sol = self._nr_biaxial(N_target, Mx_target, My_target,
                               e0, cx0, cy0, tol, max_iter)
        if sol["converged"]:
            return sol

        # Warm-start from uniaxial
        sol_uni = self._solve_uniaxial(
            N_target, Mx_target, 0.0, 1e-6, tol, max_iter // 2)
        if sol_uni["converged"]:
            sol2 = self._nr_biaxial(
                N_target, Mx_target, My_target,
                sol_uni["eps0"], sol_uni["chi_x"], 1e-7,
                tol, max_iter)
            if sol2["converged"]:
                return sol2

        # Grid of chi_y
        for cy_try in [1e-6, -1e-6, 5e-6, -5e-6, 1e-5, -1e-5]:
            sol3 = self._nr_biaxial(
                N_target, Mx_target, My_target,
                e0, cx0, cy_try, tol, max_iter // 2)
            if sol3["converged"]:
                return sol3

        return sol

    def _nr_biaxial(self, N_target, Mx_target, My_target,
                    eps0, chi_x, chi_y, tol, max_iter):
        r"""
        Core Newton-Raphson for biaxial case.

        Uses the **analytical** 3×3 tangent stiffness matrix.
        """
        x = np.array([eps0, chi_x, chi_y])
        target = np.array([N_target, Mx_target, My_target])
        tol_vec = np.array([tol, tol * 1000, tol * 1000])

        for i in range(max_iter):
            N, Mx, My, K = self.integrate_with_tangent(
                x[0], x[1], x[2])
            f = np.array([N, Mx, My])
            r = f - target

            if np.all(np.abs(r) < tol_vec):
                return {
                    "eps0": x[0], "chi_x": x[1], "chi_y": x[2],
                    "N": f[0], "Mx": f[1], "My": f[2],
                    "converged": True, "iterations": i + 1,
                }

            try:
                d = np.linalg.solve(K, -r)
            except np.linalg.LinAlgError:
                x += np.array([1e-5, 1e-7, 1e-7])
                continue

            # Backtracking line search
            alpha = 1.0
            r_norm = np.linalg.norm(r)
            for _ in range(15):
                x_new = x + alpha * d
                f_new = np.array(self.integrate(
                    x_new[0], x_new[1], x_new[2]))
                r_new = f_new - target
                if np.linalg.norm(r_new) < r_norm:
                    break
                alpha *= 0.5
            x = x + alpha * d

        f = np.array(self.integrate(x[0], x[1], x[2]))
        return {
            "eps0": x[0], "chi_x": x[1], "chi_y": x[2],
            "N": f[0], "Mx": f[1], "My": f[2],
            "converged": False, "iterations": max_iter,
        }

    # ------------------------------------------------------------------
    #  Post-processing
    # ------------------------------------------------------------------

    # def get_fiber_results(self, eps0, chi_x, chi_y=0.0):
    #     r"""
    #     Full strain/stress state at every fiber.

    #     For multi-material sections, each bulk fiber is evaluated
    #     with its own constitutive law.

    #     Parameters
    #     ----------
    #     eps0 : float
    #     chi_x : float
    #     chi_y : float, optional

    #     Returns
    #     -------
    #     dict
    #         ``'bulk'``: sub-dict with ``x``, ``y``, ``eps``,
    #         ``sigma``, ``dA``.
    #         ``'rebars'``: sub-dict with ``x``, ``y``, ``eps``,
    #         ``sigma`` (ideal_gross), ``sigma_net`` (net after bulk
    #         subtraction), ``A``, ``embedded``.
    #     """
    #     eb, er = self.strain_field(eps0, chi_x, chi_y)

    #     # Bulk stresses
    #     if not self._is_multi_material:
    #         sb = self.sec.bulk_material.stress_array(
    #             eb + self._bulk_offset())
    #     else:
    #         sb = np.zeros_like(eb)
    #         for mat, idx in self._bulk_groups:
    #             sb[idx] = mat.stress_array(
    #                 eb[idx] + self._bulk_offset(idx))

    #     # Rebar stresses (zone-aware grouping).
    #     sr_ideal_gross = np.zeros_like(er)
    #     sb_at_rebars = np.zeros_like(er)
    #     for mat, bulk_mat, idx in self._rebar_groups:
    #         er_g = er[idx]
    #         sr_ideal_gross[idx] = mat.stress_array(er_g)
    #         sb_at_rebars[idx] = bulk_mat.stress_array(
    #             er_g + self._rebar_offset(idx))

    #     # Net rebar stress (subtract zone-correct bulk at rebar location)
    #     sr_net = sr_ideal_gross.copy()
    #     emb = self.sec.embedded_rebars
    #     sr_net[emb] -= sb_at_rebars[emb]

    #     # ---- Tendon stresses (SLS): decoupled-argument evaluation ----
    #     # Each tendon's own law is evaluated at the offset total strain
    #     # (section strain + locked-in prestrain); the displaced bulk is
    #     # evaluated at the section strain.  Reported quantities:
    #     #   eps_sec  — section strain at the tendon
    #     #   eps_tot  — total strain seen by the tendon law
    #     #   sigma    — tendon stress sigma_p(eps_tot)
    #     #   sigma_net— net embedded stress (after bulk subtraction)
    #     result = {
    #         "bulk": {
    #             "x": self.sec.x_fibers.copy(),
    #             "y": self.sec.y_fibers.copy(),
    #             "eps": eb.copy(),
    #             "sigma": sb.copy(),
    #             "dA": self.sec.A_fibers.copy(),
    #         },
    #         "rebars": {
    #             "x": self.sec.x_rebars.copy(),
    #             "y": self.sec.y_rebars.copy(),
    #             "eps": er.copy(),
    #             "sigma": sr_ideal_gross.copy(),
    #             "sigma_net": sr_net.copy(),
    #             "A": self.sec.A_rebars.copy(),
    #             "embedded": self.sec.embedded_rebars.copy(),
    #         },
    #     }

    #     if self._has_tendons:
    #         et_sec = self._tendon_section_strain(eps0, chi_x, chi_y)
    #         et_tot = np.zeros_like(et_sec)
    #         sp = np.zeros_like(et_sec)
    #         sb_at_tendons = np.zeros_like(et_sec)
    #         for mat, bulk_mat, idx, eps_init in self._tendon_groups:
    #             e_sec_g = et_sec[idx]
    #             e_tot_g = e_sec_g + eps_init
    #             et_tot[idx] = e_tot_g
    #             sp[idx] = mat.stress_array(e_tot_g)
    #             sb_at_tendons[idx] = bulk_mat.stress_array(
    #                 e_sec_g + self._tendon_offset(idx))
    #         sp_net = sp.copy()
    #         emb_t = self.sec.embedded_tendons
    #         sp_net[emb_t] -= sb_at_tendons[emb_t]
    #         result["tendons"] = {
    #             "x": self.sec.x_tendons.copy(),
    #             "y": self.sec.y_tendons.copy(),
    #             "eps_sec": et_sec.copy(),
    #             "eps_init": self._eps_init_tendon.copy(),
    #             "eps_tot": et_tot.copy(),
    #             "sigma": sp.copy(),
    #             "sigma_net": sp_net.copy(),
    #             "A": self.sec.A_tendons.copy(),
    #             "embedded": self.sec.embedded_tendons.copy(),
    #         }

    #     return result
    def get_fiber_results(
        self,
        eps0,
        chi_x,
        chi_y=0.0,
    ):
        """
        Full strain/stress state at every fiber.

        Quast VMB is applied consistently with integrate().
        """

        eb, er = self.strain_field(
            eps0,
            chi_x,
            chi_y
        )

        # ======================================================
        # Quast VMB state
        # ======================================================

        quast_state_by_material = {}

        for bulk_mat, _ in self._bulk_groups:

            material_id = id(
                bulk_mat
            )

            if material_id not in quast_state_by_material:

                quast_state_by_material[
                    material_id
                ] = self._compute_quast_state(
                    eps_rebar=er,
                    bulk_material=bulk_mat,
                )

        # ======================================================
        # Bulk stresses
        # ======================================================

        if not self._is_multi_material:

            bulk_mat = self.sec.bulk_material

            eb_arg = (
                eb
                + self._bulk_offset()
            )

            sb_target = bulk_mat.stress_array(
                eb_arg
            )

            state = quast_state_by_material[
                id(bulk_mat)
            ]

            sb = self._apply_quast_vmb(
                material=bulk_mat,
                strain=eb_arg,
                stress=sb_target,
                vmb=state["vmb"],
            )

        else:

            sb = np.zeros_like(
                eb
            )

            for mat, idx in self._bulk_groups:

                eb_arg = (
                    eb[idx]
                    + self._bulk_offset(idx)
                )

                sb_target = mat.stress_array(
                    eb_arg
                )

                state = quast_state_by_material[
                    id(mat)
                ]

                sb[idx] = self._apply_quast_vmb(
                    material=mat,
                    strain=eb_arg,
                    stress=sb_target,
                    vmb=state["vmb"],
                )

        # ======================================================
        # Rebar stresses
        # ======================================================

        sr_ideal_gross = np.zeros_like(
            er
        )

        sb_at_rebars = np.zeros_like(
            er
        )

        for (
            mat,
            bulk_mat,
            idx,
        ) in self._rebar_groups:

            er_g = er[idx]

            # Stahlspannung
            sr_ideal_gross[idx] = (
                mat.stress_array(
                    er_g
                )
            )

            # Betondehnung an Bewehrungsposition
            bulk_eps_at_rebars = (
                er_g
                + self._rebar_offset(idx)
            )

            sb_target_at_rebars = (
                bulk_mat.stress_array(
                    bulk_eps_at_rebars
                )
            )

            state = (
                quast_state_by_material[
                    id(bulk_mat)
                ]
            )

            # VMB auch auf verdrängten Beton anwenden
            sb_at_rebars[idx] = (
                self._apply_quast_vmb(
                    material=bulk_mat,
                    strain=bulk_eps_at_rebars,
                    stress=sb_target_at_rebars,
                    vmb=state["vmb"],
                )
            )

        # ======================================================
        # Net rebar stress
        # ======================================================

        sr_net = (
            sr_ideal_gross.copy()
        )

        emb = self.sec.embedded_rebars

        sr_net[emb] -= (
            sb_at_rebars[emb]
        )

        # ======================================================
        # Result
        # ======================================================

        result = {
            "bulk": {
                "x": self.sec.x_fibers.copy(),
                "y": self.sec.y_fibers.copy(),
                "eps": eb.copy(),
                "sigma": sb.copy(),
                "dA": self.sec.A_fibers.copy(),
            },

            "rebars": {
                "x": self.sec.x_rebars.copy(),
                "y": self.sec.y_rebars.copy(),
                "eps": er.copy(),
                "sigma": sr_ideal_gross.copy(),
                "sigma_net": sr_net.copy(),
                "A": self.sec.A_rebars.copy(),
                "embedded": (
                    self.sec.embedded_rebars.copy()
                ),
            },

            # Für Kontrolle und spätere Ausgabe
            "quast_vmb": {
                id(mat): {
                    "vmb": float(
                        state["vmb"]
                    ),
                    "eps_c_ref": float(
                        state["eps_c_ref"]
                    ),
                    "governing_index": (
                        state["governing_index"]
                    ),
                    "dvmb_deps_ref": float(
                        state["dvmb_deps_ref"]
                    ),
                }
                for mat, _ in self._bulk_groups
                for state in [
                    quast_state_by_material[
                        id(mat)
                    ]
                ]
            },
        }

        # ======================================================
        # Tendons
        # ======================================================

        if self._has_tendons:

            et_sec = (
                self._tendon_section_strain(
                    eps0,
                    chi_x,
                    chi_y
                )
            )

            et_tot = np.zeros_like(
                et_sec
            )

            sp = np.zeros_like(
                et_sec
            )

            sb_at_tendons = np.zeros_like(
                et_sec
            )

            for (
                mat,
                bulk_mat,
                idx,
                eps_init,
            ) in self._tendon_groups:

                # ======================================================
                # Section and total tendon strains
                # ======================================================

                e_sec_g = et_sec[idx]

                e_tot_g = (
                    e_sec_g
                    + eps_init
                )

                et_tot[idx] = e_tot_g

                # ======================================================
                # Tendon stress
                # ======================================================
                e_tot_eval = np.minimum(
                    e_tot_g, eps_ud_limit[None, :]
                )
                sp_group = mat.stress_array(
                    e_tot_eval
                )

                # Individual permissible tensile strain
                eps_ud_limit = self._tendon_eps_ud_limits(
                    material=mat,
                    eps_init=eps_init,
                    use_german_eps_ud=getattr(
                        mat,
                        "use_german_eps_ud",
                        False,
                    ),
                )

                failed_tension = (
                    e_tot_g
                    > eps_ud_limit
                )

                sp_group = sp_group.copy()

                sp_group[
                    failed_tension
                ] = 0.0

                sp[idx] = sp_group

                # ======================================================
                # Concrete strain at tendon position
                # ======================================================

                bulk_eps_at_tendons = (
                    e_sec_g
                    + self._tendon_offset(idx)
                )

                # ======================================================
                # Unreduced concrete stress at tendon position
                # ======================================================

                sb_target_at_tendons = (
                    bulk_mat.stress_array(
                        bulk_eps_at_tendons
                    )
                )

                # ======================================================
                # Quast state of corresponding concrete material
                # ======================================================

                state = (
                    quast_state_by_material[
                        id(bulk_mat)
                    ]
                )

                # ======================================================
                # Apply same VMB as in the concrete fibers
                # ======================================================

                sb_at_tendons[idx] = (
                    self._apply_quast_vmb(
                        material=bulk_mat,
                        strain=bulk_eps_at_tendons,
                        stress=sb_target_at_tendons,
                        vmb=state["vmb"],
                    )
                )

            sp_net = sp.copy()

            emb_t = (
                self.sec.embedded_tendons
            )

            sp_net[emb_t] -= (
                sb_at_tendons[emb_t]
            )

            result["tendons"] = {
                "x": self.sec.x_tendons.copy(),
                "y": self.sec.y_tendons.copy(),

                "eps_sec": et_sec.copy(),
                "eps_init": (
                    self._eps_init_tendon.copy()
                ),
                "eps_tot": et_tot.copy(),

                "sigma": sp.copy(),
                "sigma_net": sp_net.copy(),

                "A": self.sec.A_tendons.copy(),

                "embedded": (
                    self.sec.embedded_tendons.copy()
                ),
            }

        return result

    def strains_within_limits(self, eps0, chi_x, chi_y=0.0):
        r"""
        Check whether all fiber strains are within material limits.

        A strain field is *admissible* when every fiber's strain
        lies within the range :math:`[\varepsilon_{\min},\,
        \varepsilon_{\max}]` defined by the fiber's constitutive
        law.

        This is the criterion that distinguishes interior points
        of the resistance domain from exterior points: the domain
        boundary corresponds to configurations where at least one
        fiber is *at* its limit.

        Parameters
        ----------
        eps0 : float
            Strain at the reference point.
        chi_x : float
            Curvature about the x-axis [1/mm].
        chi_y : float, optional
            Curvature about the y-axis [1/mm].

        Returns
        -------
        bool
            ``True`` if all fibers satisfy
            :math:`\varepsilon_{\min} \le \varepsilon_i
            \le \varepsilon_{\max}`.
        """
        eb, er = self.strain_field(eps0, chi_x, chi_y)

        for mat, idx in self._bulk_groups:
            e_group = eb[idx] + self._bulk_offset(idx)
            if np.any(e_group < mat.eps_min) or \
               np.any(e_group > mat.eps_max):
                return False

        for mat, _bulk_mat, idx in self._rebar_groups:
            e_group = er[idx]
            if np.any(e_group < mat.eps_min) or \
               np.any(e_group > mat.eps_max):
                return False

        # Tendons: limits apply to the TOTAL strain (section strain +
        # locked-in prestrain), since that is the argument of the
        # tendon's constitutive law.
        if self._has_tendons:
            et_sec = self._tendon_section_strain(eps0, chi_x, chi_y)
            for mat, _bulk_mat, idx, eps_init in self._tendon_groups:
                e_tot = et_sec[idx] + eps_init
                eps_ud_limit = self._tendon_eps_ud_limits(
                    material=mat,
                    eps_init=eps_init,
                    use_german_eps_ud=getattr(
                        mat,
                        "use_german_eps_ud",
                        False,
                    ),
                )

                if (
                    np.any(e_tot < mat.eps_min)
                    or np.any(e_tot > eps_ud_limit)
                ):
                    return False
                if np.any(e_tot < mat.eps_min) or \
                   np.any(e_tot > eps_ud_limit):
                    return False

        return True
