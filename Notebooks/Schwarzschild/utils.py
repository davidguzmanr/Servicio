import random
import warnings
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d


class DualNumber:
    """
    Numbers of the form, :math:`a + b\\epsilon`, where
    :math:`\\epsilon^2 = 0` and :math:`\\epsilon \\ne 0`.
    Their addition and multiplication properties make them
    suitable for Automatic Differentiation (AD).
    EinsteinPy uses AD for solving Geodesics in arbitrary spacetimes.
    This module is based on [1]_.

    References
    ----------
    .. [1] Christian, Pierre and Chan, Chi-Kwan;
        "FANTASY: User-Friendly Symplectic Geodesic Integrator
        for Arbitrary Metrics with Automatic Differentiation";
        `arXiv:2010.02237 <https://arxiv.org/abs/2010.02237>`__

    """

    def __init__(self, val, deriv):
        """
        Constructor

        Parameters
        ----------
        val : float
            Value
        deriv : float
            Directional Derivative

        """
        self.val = float(val)
        self.deriv = float(deriv)

    def __str__(self):
        return f"DualNumber({self.val}, {self.deriv})"

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.val + other.val, self.deriv + other.deriv)

        return DualNumber(self.val + other, self.deriv)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.val - other.val, self.deriv - other.deriv)

        return DualNumber(self.val - other, self.deriv)

    def __rsub__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(other.val - self.val, other.deriv - self.deriv)

        return DualNumber(other, 0) - self

    def __mul__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(
                self.val * other.val, self.deriv * other.val + self.val * other.deriv
            )

        return DualNumber(self.val * other, self.deriv * other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, DualNumber):
            if self.val == 0 and other.val == 0:
                return DualNumber(self.deriv / other.deriv, 0.0)

            return DualNumber(
                self.val / other.val,
                (self.deriv * other.val - self.val * other.deriv) / (other.val ** 2),
            )

        return DualNumber(self.val / other, self.deriv / other)

    def __rtruediv__(self, other):
        if isinstance(other, DualNumber):
            if self.val == 0 and other.val == 0:
                return DualNumber(other.deriv / self.deriv, 0.0)

            return DualNumber(
                other.val / self.val,
                (other.deriv * self.val - other.val * self.deriv) / (self.val ** 2),
            )

        return DualNumber(other, 0).__truediv__(self)

    def __eq__(self, other):
        return (self.val == other.val) and (self.deriv == other.deriv)

    def __ne__(self, other):
        return not (self == other)

    def __neg__(self):
        return DualNumber(-self.val, -self.deriv)

    def __pow__(self, power):
        return DualNumber(
            self.val ** power, self.deriv * power * self.val ** (power - 1)
        )

    def sin(self):
        return DualNumber(np.sin(self.val), self.deriv * np.cos(self.val))

    def cos(self):
        return DualNumber(np.cos(self.val), -self.deriv * np.sin(self.val))

    def tan(self):
        return np.sin(self) / np.cos(self)

    def log(self):
        return DualNumber(np.log(self.val), self.deriv / self.val)

    def exp(self):
        return DualNumber(np.exp(self.val), self.deriv * np.exp(self.val))


def _deriv(func, x):
    """
    Calculates first (partial) derivative of ``func`` at ``x``

    Parameters
    ----------
    func : callable
        Function to differentiate
    x : float
        Point, at which, the derivative will be evaluated

    Returns
    _______
    float
        First partial derivative of ``func`` at ``x``

    """
    funcdual = func(DualNumber(x, 1.0))

    if isinstance(funcdual, DualNumber):
        return funcdual.deriv

    return 0.0


def _diff_g(g, g_prms, coords, indices, wrt):
    """
    Computes derivative of metric elements

    Parameters
    ----------
    g : callable
        Metric (Contravariant) Function
    g_prms : array_like
        Tuple of parameters to pass to the metric
        E.g., ``(a,)`` for Kerr
    coords : array_like
        4-Position
    indices : array_like
        2-tuple, containing indices, indexing a metric
        element, whose derivative will be calculated
    wrt : int
        Coordinate, with respect to which, the derivative
        will be calculated
        Takes values from ``[0, 1, 2, 3]``

    Returns
    -------
    float
        Value of derivative of metric element at ``coords``

    Raises
    ------
    ValueError
        If ``wrt`` is not in [1, 2, 3, 4]
        or ``len(indices) != 2``

    """
    if wrt not in [0, 1, 2, 3]:
        raise ValueError(f"wrt takes values from [0, 1, 2, 3]. Supplied value: {wrt}")

    if len(indices) != 2:
        raise ValueError("indices must be a 2-tuple containing indices for the metric.")

    dual_coords = [
        DualNumber(coords[0], 0.0),
        DualNumber(coords[1], 0.0),
        DualNumber(coords[2], 0.0),
        DualNumber(coords[3], 0.0),
    ]

    # Coordinate, against which, derivative will be propagated
    dual_coords[wrt].deriv = 1.0

    return _deriv(lambda q: g(dual_coords, *g_prms)[indices], coords[wrt])


def _jacobian_g(g, g_prms, coords, wrt):
    """
    Part of Jacobian of Metric

    Parameters
    ----------
    g : callable
        Metric (Contravariant) Function
    g_prms : array_like
        Tuple of parameters to pass to the metric
        E.g., ``(a,)`` for Kerr
    coords : array_like
        4-Position
    wrt : int
        Coordinate, with respect to which, the derivative
        will be calculated
        Takes values from ``[0, 1, 2, 3]``

    Returns
    -------
    numpy.ndarray
        Value of derivative of metric elements,
        w.r.t a particular coordinate, at ``coords``

    """
    J = np.zeros((4, 4))

    for i in range(4):
        for j in range(4):
            if i <= j:
                J[i, j] = _diff_g(g, g_prms, coords, (i, j), wrt)

    J = J + J.T - np.diag(np.diag(J))

    return J

def _P(g, g_prms, q, p, time_like=True):
    """
    Utility function to compute 4-Momentum of the test particle

    Parameters
    ----------
    g : callable
        Metric Function
    g_prms : array_like
        Tuple of parameters to pass to the metric
        E.g., ``(a,)`` for Kerr
    q : array_like
        Initial 4-Position
    p : array_like
        Initial 3-Momentum
    time_like: bool, optional
        Determines type of Geodesic
        ``True`` for Time-like geodesics
        ``False`` for Null-like geodesics
        Defaults to ``True``

    Returns
    -------
    P: numpy.ndarray
        4-Momentum

    """
    guu = g(q, *g_prms)
    P = np.array([0.0, *p])

    A = guu[0, 0]
    B = 2 * guu[0, 3] * P[3]
    C = (
        guu[1, 1] * P[1] * P[1]
        + guu[2, 2] * P[2] * P[2]
        + guu[3, 3] * P[3] * P[3]
        + int(time_like)
    )

    P[0] = (-B + np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)

    return P


def sigma(r, theta, a):
    """
    Returns the value of :math:`r^2 + a^2 * \\cos^2(\\theta)`
    Uses natural units, with :math:`c = G = M = k_e = 1`

    Parameters
    ----------
    r : float
        r-component of 4-Position
    theta : float
        theta-component of 4-Position
    a : float
        Spin Parameter
        :math:`0 \\le a \\le 1`

    Returns
    -------
    float
        The value of :math:`r^2 + a^2 * \\cos^2(\\theta)`

    """
    sigma_ = (r ** 2) + ((a * np.cos(theta)) ** 2)

    return sigma_


def delta(r, a, Q=0):
    """
    Returns the value of :math:`r^2 - r_s r + a^2 + r_Q^2`
    Uses natural units, with :math:`c = G = M = k_e = 1`

    Parameters
    ----------
    r : float
        r-component of 4-Position
    a : float
        Spin Parameter
        :math:`0 \\le a \\le 1`
    Q : float
        Charge on gravitating body
        Defaults to ``0``

    Returns
    -------
    float
        The value of :math:`r^2 - r_s r + a^2 + r_Q^2`

    """
    delta_ = (r ** 2) - (2 * r) + (a ** 2) + Q ** 2

    return delta_


def _sch(x_vec, *params):
    """
    Contravariant Schwarzschild Metric in Spherical Polar coordinates
    Uses natural units, with :math:`c = G = M = k_e = 1`

    Parameters
    ----------
    x_vec : array_like
        4-Position

    Other Parameters
    ----------------
    params : array_like
        Tuple of parameters to pass to the metric

    Returns
    -------
    numpy.ndarray
        Contravariant Schwarzschild Metric Tensor

    """
    r, th = x_vec[1], x_vec[2]

    g = np.zeros(shape=(4, 4), dtype=DualNumber)

    tmp = 1.0 - (2 / r)
    g[0, 0] = -1 / tmp
    g[1, 1] = tmp
    g[2, 2] = 1 / (r ** 2)
    g[3, 3] = 1 / ((r * np.sin(th)) ** 2)

    return g


def _kerr(x_vec, *params):
    """
    Contravariant Kerr Metric in Boyer-Lindquist coordinates
    Uses natural units, with :math:`c = G = M = k_e = 1`

    Parameters
    ----------
    x_vec : array_like
        4-Position

    Other Parameters
    ----------------
    params : array_like
        Tuple of parameters to pass to the metric
        Should contain Spin Parameter, ``a``

    Returns
    -------
    numpy.ndarray
        Contravariant Kerr Metric Tensor

    """
    a = params[0]

    r, th = x_vec[1], x_vec[2]
    sg, dl = sigma(r, th, a), (r ** 2) - (2 * r) + (a ** 2)

    g = np.zeros(shape=(4, 4), dtype=DualNumber)

    g[0, 0] = -(r ** 2 + a ** 2 + (2 * r * (a * np.sin(th)) ** 2) / sg) / dl
    g[1, 1] = dl / sg
    g[2, 2] = 1 / sg
    g[3, 3] = (1 / (dl * np.sin(th) ** 2)) * (1 - 2 * r / sg)
    g[0, 3] = g[3, 0] = -(2 * r * a) / (sg * dl)

    return g


def _kerrnewman(x_vec, *params):
    """
    Contravariant Kerr-Newman Metric in Boyer-Lindquist coordinates
    Uses natural units, with :math:`c = G = M = k_e = 1`

    Parameters
    ----------
    x_vec : array_like
        4-Position

    Other Parameters
    ----------------
    params : array_like
        Tuple of parameters to pass to the metric
        Should contain Spin, ``a``, and Charge, ``Q``

    Returns
    -------
    numpy.ndarray
        Contravariant Kerr-Newman Metric Tensor

    """
    a, Q = params[0], params[1]

    r, th = x_vec[1], x_vec[2]
    sg, dl = sigma(r, th, a), (r ** 2) - (2 * r) + (a ** 2) + Q ** 2
    a2 = a ** 2
    r2 = r ** 2
    sint2 = np.sin(th) ** 2
    csct2 = 1 / sint2
    csct4 = 1 / sint2 ** 2

    g = np.zeros(shape=(4, 4), dtype=DualNumber)

    denom = dl * (a2 + 2 * r2 + a2 * np.cos(2 * th)) ** 2
    g[0, 0] = -(4 * sg * ((a2 + r2) ** 2 - a2 * dl * sint2) / denom)
    g[1, 1] = dl / sg
    g[2, 2] = 1 / sg
    g[3, 3] = (sg * csct4 * (-a2 + dl * csct2)) / (dl * (a2 - (a2 + r2) * csct2) ** 2)
    g[0, 3] = g[3, 0] = -(4 * a * (a2 - dl + r2) * sg / denom)

    return g


def _PartHamFlow(g, g_prms, q, p, wrt):
    """
    Partial Hamiltonian Flow computed from the Metric

    Parameters
    ----------
    g : callable
        Metric Function
    g_prms : array_like
        Tuple of parameters to pass to the metric
        E.g., ``(a,)`` for Kerr
    q : array_like
        Initial 4-Position
    p : array_like
        Initial 3-Momentum
    wrt : int
        Coordinate, with respect to which, the derivative
        will be calculated
        Takes values from ``[0, 1, 2, 3]``

    Returns
    -------
    float
        Partial Hamiltonian Flow

    References
    ----------
    .. [1] Christian, Pierre and Chan, Chi-Kwan;
        "FANTASY: User-Friendly Symplectic Geodesic Integrator
        for Arbitrary Metrics with Automatic Differentiation";
        `arXiv:2010.02237 <https://arxiv.org/abs/2010.02237>`__

    """
    return _jacobian_g(g, g_prms, q, wrt) @ p @ p


def _flow_A(g, g_prms, q1, p1, q2, p2, delta=0.5):
    """
    Overall flow of Hamiltonian, :math:`H_A`

    Parameters
    ----------
    g : callable
        Metric Function
    g_prms : array_like
        Tuple of parameters to pass to the metric
        E.g., ``(a,)`` for Kerr
    q1 : array_like
        First copy of 4-Position
    p1 : array_like
        First copy of 3-Momentum
    q2 : array_like
        Second copy of 4-Position
    p2 : array_like
        Second copy of 3-Momentum
    delta : float
        Initial integration step-size
        Defaults to ``0.5``

    Returns
    -------
    float
        Hamiltonian Flow for :math:`H_A`

    References
    ----------
    .. [1] Christian, Pierre and Chan, Chi-Kwan;
        "FANTASY: User-Friendly Symplectic Geodesic Integrator
        for Arbitrary Metrics with Automatic Differentiation";
        `arXiv:2010.02237 <https://arxiv.org/abs/2010.02237>`__

    """
    dH1 = [0.5 * (_PartHamFlow(g, g_prms, q1, p2, i)) for i in range(4)]
    dp1 = np.array(dH1)
    p1_next = p1 - delta * dp1

    dH2 = g(q1, *g_prms) @ p2
    dq2 = np.array(dH2)
    q2_next = q2 + delta * dq2

    return q2_next, p1_next


def _flow_B(g, g_prms, q1, p1, q2, p2, delta=0.5):
    """
    Overall flow of Hamiltonian, :math:`H_B`

    Parameters
    ----------
    g : callable
        Metric Function
    g_prms : array_like
        Tuple of parameters to pass to the metric
        E.g., ``(a,)`` for Kerr
    q1 : array_like
        First copy of 4-Position
    p1 : array_like
        First copy of 3-Momentum
    q2 : array_like
        Second copy of 4-Position
    p2 : array_like
        Second copy of 3-Momentum
    delta : float
        Initial integration step-size
        Defaults to ``0.5``

    Returns
    -------
    float
        Hamiltonian Flow for :math:`H_B`

    References
    ----------
    .. [1] Christian, Pierre and Chan, Chi-Kwan;
        "FANTASY: User-Friendly Symplectic Geodesic Integrator
        for Arbitrary Metrics with Automatic Differentiation";
        `arXiv:2010.02237 <https://arxiv.org/abs/2010.02237>`__

    """
    dH2 = [
        0.5
        * (
            _PartHamFlow(
                g,
                g_prms,
                q2,
                p1,
                i,
            )
        )
        for i in range(4)
    ]
    dp2 = np.array(dH2)
    p2_next = p2 - delta * dp2

    dH1 = g(q2, *g_prms) @ p1
    dq1 = np.array(dH1)
    q1_next = q1 + delta * dq1

    return q1_next, p2_next


def _flow_mixed(q1, p1, q2, p2, delta=0.5, omega=1.0):
    """
    Mixed flow of Hamiltonian, :math:`\\tilde{H}`

    Parameters
    ----------
    q1 : array_like
        First copy of 4-Position
    p1 : array_like
        First copy of 3-Momentum
    q2 : array_like
        Second copy of 4-Position
    p2 : array_like
        Second copy of 3-Momentum
    delta : float
        Initial integration step-size
        Defaults to ``0.5``
    omega : float
        Coupling for Hamiltonian Flows
        Defaults to ``1.0``

    Returns
    -------
    float
        Hamiltonian Flow for :math:`\\tilde{H}`

    References
    ----------
    .. [1] Christian, Pierre and Chan, Chi-Kwan;
        "FANTASY: User-Friendly Symplectic Geodesic Integrator
        for Arbitrary Metrics with Automatic Differentiation";
        `arXiv:2010.02237 <https://arxiv.org/abs/2010.02237>`__

    """
    q_sum = q1 + q2
    q_dif = q1 - q2
    p_sum = p1 + p2
    p_dif = p1 - p2
    cos = np.cos(2.0 * omega * delta)
    sin = np.sin(2.0 * omega * delta)

    q1_next = 0.5 * (q_sum + (q_dif) * cos + (p_dif) * sin)
    p1_next = 0.5 * (p_sum + (p_dif) * cos - (q_dif) * sin)
    q2_next = 0.5 * (q_sum - (q_dif) * cos - (p_dif) * sin)
    p2_next = 0.5 * (p_sum - (p_dif) * cos + (q_dif) * sin)

    return q1_next, p1_next, q2_next, p2_next


def _Z(order):
    """
    Returns the constants for Yoshida Triple Jump.
    Used to compose higher order (even) integrators.

    References
    ----------
    .. [1] Yoshida, Haruo,
        "Construction of higher order symplectic integrators";
         Physics Letters A, vol. 150, no. 5-7, pp. 262-268, 1990.
        `DOI: <https://doi.org/10.1016/0375-9601(90)90092-3>`__

    """
    n = (order - 2) / 2
    x = 2 ** (1 / (2 * n + 1))
    Z0 = -x / (2 - x)
    Z1 = 1 / (2 - x)

    return Z0, Z1


class GeodesicIntegrator:
    """
    Geodesic Integrator, based on [1]_.
    This module uses Forward Mode Automatic Differentiation
    to calculate metric derivatives to machine precision
    leading to stable simulations.

    References
    ----------
    .. [1] Christian, Pierre and Chan, Chi-Kwan;
        "FANTASY: User-Friendly Symplectic Geodesic Integrator
        for Arbitrary Metrics with Automatic Differentiation";
        `arXiv:2010.02237 <https://arxiv.org/abs/2010.02237>`__

    """

    # TODO: Update arXiv attributions to ApJ (See #572)
    def __init__(
        self,
        metric,
        metric_params,
        q0,
        p0,
        time_like=True,
        steps=100,
        delta=0.5,
        rtol=1e-2,
        atol=1e-2,
        order=2,
        omega=1.0,
        suppress_warnings=False,
    ):
        """
        Constructor

        Parameters
        ----------
        metric : callable
            Metric Function. Currently, these metrics are supported:
            1. Schwarzschild
            2. Kerr
            3. KerrNewman
        metric_params : array_like
            Tuple of parameters to pass to the metric
            E.g., ``(a,)`` for Kerr
        q0 : array_like
            Initial 4-Position
        p0 : array_like
            Initial 4-Momentum
        time_like : bool, optional
            Determines type of Geodesic
            ``True`` for Time-like geodesics
            ``False`` for Null-like geodesics
            Defaults to ``True``
        steps : int
            Number of integration steps
            Defaults to ``50``
        delta : float
            Initial integration step-size
            Defaults to ``0.5``
        rtol : float
            Relative Tolerance
            Defaults to ``1e-2``
        atol : float
            Absolute Tolerance
            Defaults to ``1e-2``
        order : int
            Integration Order
            Defaults to ``2``
        omega : float
            Coupling between Hamiltonian Flows
            Smaller values imply smaller integration error, but too
            small values can make the equation of motion non-integrable.
            For non-capture trajectories, ``omega = 1.0`` is recommended.
            For trajectories, that either lead to a capture or a grazing
            geodesic, a decreased value of ``0.01`` or less is recommended.
            Defaults to ``1.0``
        suppress_warnings : bool
            Whether to suppress warnings during simulation
            Warnings are shown for every step, where numerical errors
            exceed specified tolerance (controlled by ``rtol`` and ``atol``)
            Defaults to ``False``

        Raises
        ------
        NotImplementedError
            If ``order`` is not in [2, 4, 6, 8]

        """
        ORDERS = {
            2: self._ord_2,
            4: self._ord_4,
            6: self._ord_6,
            8: self._ord_8,
        }
        self.metric = metric
        self.metric_params = metric_params
        self.q0 = q0
        self.p0 = p0
        self.time_like = time_like
        self.steps = steps
        self.delta = delta
        self.omega = omega
        if order not in ORDERS:
            raise NotImplementedError(
                f"Order {order} integrator has not been implemented."
            )
        self.order = order
        self.integrator = ORDERS[order]
        self.rtol = rtol
        self.atol = atol
        self.suppress_warnings = suppress_warnings

        self.step_num = 0
        self.res_list = [q0, p0, q0, p0]
        self.results = list()

    def __str__(self):
        return f"""{self.__class__.__name__}(\n                metric : {self.metric}\n                metric_params : {self.metric_params}\n                q0 : {self.q0},\n                p0 : {self.p0},\n                time_like : {self.time_like},\n                steps : {self.steps},\n                delta : {self.delta},\n                omega : {self.omega},\n                order : {self.order},\n                rtol : {self.rtol},\n                atol : {self.atol}\n                suppress_warnings : {self.suppress_warnings}
            )"""

    def __repr__(self):
        return self.__str__()

    def _ord_2(self, q1, p1, q2, p2, delta):
        """
        Order 2 Integration Scheme

        References
        ----------
        .. [1] Christian, Pierre and Chan, Chi-Kwan;
            "FANTASY : User-Friendly Symplectic Geodesic Integrator
            for Arbitrary Metrics with Automatic Differentiation";
            `arXiv:2010.02237 <https://arxiv.org/abs/2010.02237>`__

        """
        dl, omg = delta, self.omega
        g = self.metric
        g_prms = self.metric_params

        HA1 = np.array(
            [
                q1,
                _flow_A(g, g_prms, q1, p1, q2, p2, 0.5 * dl)[1],
                _flow_A(g, g_prms, q1, p1, q2, p2, 0.5 * dl)[0],
                p2,
            ]
        )
        HB1 = np.array(
            [
                _flow_B(g, g_prms, HA1[0], HA1[1], HA1[2], HA1[3], 0.5 * dl)[0],
                HA1[1],
                HA1[2],
                _flow_B(g, g_prms, HA1[0], HA1[1], HA1[2], HA1[3], 0.5 * dl)[1],
            ]
        )
        HC = _flow_mixed(HB1[0], HB1[1], HB1[2], HB1[3], dl, omg)
        HB2 = np.array(
            [
                _flow_B(g, g_prms, HC[0], HC[1], HC[2], HC[3], 0.5 * dl)[0],
                HC[1],
                HC[2],
                _flow_B(g, g_prms, HC[0], HC[1], HC[2], HC[3], 0.5 * dl)[1],
            ]
        )
        HA2 = np.array(
            [
                HB2[0],
                _flow_A(g, g_prms, HB2[0], HB2[1], HB2[2], HB2[3], 0.5 * dl)[1],
                _flow_A(g, g_prms, HB2[0], HB2[1], HB2[2], HB2[3], 0.5 * dl)[0],
                HB2[3],
            ]
        )

        return HA2

    def _ord_4(self, q1, p1, q2, p2, delta):
        """
        Order 4 Integration Scheme

        References
        ----------
        .. [1] Yoshida, Haruo,
            "Construction of higher order symplectic integrators";
             Physics Letters A, vol. 150, no. 5-7, pp. 262-268, 1990.
            `DOI: <https://doi.org/10.1016/0375-9601(90)90092-3>`__

        """
        dl = delta

        Z0, Z1 = _Z(self.order)
        step1 = self._ord_2(q1, p1, q2, p2, dl * Z1)
        step2 = self._ord_2(step1[0], step1[1], step1[2], step1[3], dl * Z0)
        step3 = self._ord_2(step2[0], step2[1], step2[2], step2[3], dl * Z1)

        return step3

    def _ord_6(self, q1, p1, q2, p2, delta):
        """
        Order 6 Integration Scheme

        References
        ----------
        .. [1] Yoshida, Haruo,
            "Construction of higher order symplectic integrators";
             Physics Letters A, vol. 150, no. 5-7, pp. 262-268, 1990.
            `DOI: <https://doi.org/10.1016/0375-9601(90)90092-3>`__

        """
        dl = delta

        Z0, Z1 = _Z(self.order)
        step1 = self._ord_4(q1, p1, q2, p2, dl * Z1)
        step2 = self._ord_4(step1[0], step1[1], step1[2], step1[3], dl * Z0)
        step3 = self._ord_4(step2[0], step2[1], step2[2], step2[3], dl * Z1)

        return step3

    def _ord_8(self, q1, p1, q2, p2, delta):
        """
        Order 8 Integration Scheme

        References
        ----------
        .. [1] Yoshida, Haruo,
            "Construction of higher order symplectic integrators";
             Physics Letters A, vol. 150, no. 5-7, pp. 262-268, 1990.
            `DOI: <https://doi.org/10.1016/0375-9601(90)90092-3>`__

        """
        dl = delta

        Z0, Z1 = _Z(self.order)
        step1 = self._ord_6(q1, p1, q2, p2, dl * Z1)
        step2 = self._ord_6(step1[0], step1[1], step1[2], step1[3], dl * Z0)
        step3 = self._ord_6(step2[0], step2[1], step2[2], step2[3], dl * Z1)

        return step3

    def step(self):
        """
        Advances integration by one step

        """
        rl = self.res_list

        arr = self.integrator(rl[0], rl[1], rl[2], rl[3], self.delta)

        self.res_list = arr
        self.step_num += 1

        # Stability check
        if not self.suppress_warnings:
            g = self.metric
            g_prms = self.metric_params

            q1 = arr[0]
            p1 = arr[1]
            # Ignoring
            # q_2 = arr[2]
            # p_2 = arr[3]

            const = -int(self.time_like)
            # g.p.p ~ -1 or 0 (const)
            if not np.allclose(
                g(q1, *g_prms) @ p1 @ p1, const, rtol=self.rtol, atol=self.atol
            ):
                warnings.warn(
                    f"Numerical error has exceeded specified tolerance at step = {self.step_num}.",
                    RuntimeWarning,
                )

        self.results.append(self.res_list)

class Geodesic:
    """
    Base Class for defining Geodesics
    Working in Geometrized Units (M-Units),
    with :math:`c = G = M = k_e = 1`

    """

    def __init__(
        self,
        metric,
        metric_params,
        position,
        momentum,
        time_like=True,
        return_cartesian=True,
        **kwargs,
    ):
        """
        Constructor

        Parameters
        ----------
        metric : str
            Name of the metric. Currently, these metrics are supported:
            1. Schwarzschild
            2. Kerr
            3. KerrNewman
        metric_params : array_like
            Tuple of parameters to pass to the metric
            E.g., ``(a,)`` for Kerr
        position : array_like
            3-Position
            4-Position is initialized by taking ``t = 0.0``
        momentum : array_like
            3-Momentum
            4-Momentum is calculated automatically,
            considering the value of ``time_like``
        time_like : bool, optional
            Determines type of Geodesic
            ``True`` for Time-like geodesics
            ``False`` for Null-like geodesics
            Defaults to ``True``
        return_cartesian : bool, optional
            Whether to return calculated positions in Cartesian Coordinates
            This only affects the coordinates. Momenta are dimensionless
            quantities, and are returned in Spherical Polar Coordinates.
            Defaults to ``True``
        kwargs : dict
            Keyword parameters for the Geodesic Integrator
            See 'Other Parameters' below.

        Other Parameters
        ----------------
        steps : int
            Number of integration steps
            Defaults to ``50``
        delta : float
            Initial integration step-size
            Defaults to ``0.5``
        rtol : float
            Relative Tolerance
            Defaults to ``1e-2``
        atol : float
            Absolute Tolerance
            Defaults to ``1e-2``
        order : int
            Integration Order
            Defaults to ``2``
        omega : float
            Coupling between Hamiltonian Flows
            Smaller values imply smaller integration error, but too
            small values can make the equation of motion non-integrable.
            For non-capture trajectories, ``omega = 1.0`` is recommended.
            For trajectories, that either lead to a capture or a grazing
            geodesic, a decreased value of ``0.01`` or less is recommended.
            Defaults to ``1.0``
        suppress_warnings : bool
            Whether to suppress warnings during simulation
            Warnings are shown for every step, where numerical errors
            exceed specified tolerance (controlled by ``rtol`` and ``atol``)
            Defaults to ``False``

        """
        # Contravariant Metrics, defined so far
        _METRICS = {
            "Schwarzschild": _sch,
            "Kerr": _kerr,
            "KerrNewman": _kerrnewman,
        }

        if metric not in _METRICS:
            raise NotImplementedError(
                f"'{metric}' is unsupported. Currently, these metrics are supported:\
                \n1. Schwarzschild\n2. Kerr\n3. KerrNewman"
            )

        self.metric_name = metric
        self.metric = _METRICS[metric]
        self.metric_params = metric_params
        if metric == "Schwarzschild":
            self.metric_params = (0.0,)
        self.position = np.array([0.0, *position])
        self.momentum = _P(
            self.metric, metric_params, self.position, momentum, time_like
        )
        self.time_like = time_like

        self.kind = "Time-like" if time_like else "Null-like"
        self.coords = "Cartesian" if return_cartesian else "Spherical Polar"

        self._trajectory = self.calculate_trajectory(**kwargs)

    def __repr__(self):
        return f"""Geodesic Object:(\n            Type : ({self.kind}),\n            Metric : ({self.metric_name}),\n            Metric Parameters : ({self.metric_params}),\n            Initial 4-Position : ({self.position}),\n            Initial 4-Momentum : ({self.momentum}),\n            Trajectory = (\n                {self.trajectory}\n            ),\n            Output Position Coordinate System = ({self.coords})\n        ))"""

    def __str__(self):
        return self.__repr__()

    @property
    def trajectory(self):
        """
        Returns the trajectory of the test particle

        """
        return self._trajectory

    def calculate_trajectory(self, **kwargs):
        """
        Calculate trajectory in spacetime

        Parameters
        ----------
        kwargs : dict
            Keyword parameters for the Geodesic Integrator
            See 'Other Parameters' below.

        Returns
        -------
        ~numpy.ndarray
            N-element numpy array, containing step count
        ~numpy.ndarray
            Shape-(N, 8) numpy array, containing
            (4-Position, 4-Momentum) for each step

        Other Parameters
        ----------------
        steps : int
            Number of integration steps
            Defaults to ``50``
        delta : float
            Initial integration step-size
            Defaults to ``0.5``
        rtol : float
            Relative Tolerance
            Defaults to ``1e-2``
        atol : float
            Absolute Tolerance
            Defaults to ``1e-2``
        order : int
            Integration Order
            Defaults to ``2``
        omega : float
            Coupling between Hamiltonian Flows
            Smaller values imply smaller integration error, but too
            small values can make the equation of motion non-integrable.
            For non-capture trajectories, ``omega = 1.0`` is recommended.
            For trajectories, that either lead to a capture or a grazing
            geodesic, a decreased value of ``0.01`` or less is recommended.
            Defaults to ``1.0``
        suppress_warnings : bool
            Whether to suppress warnings during simulation
            Warnings are shown for every step, where numerical errors
            exceed specified tolerance (controlled by ``rtol`` and ``atol``)
            Defaults to ``False``

        """
        g, g_prms = self.metric, self.metric_params
        q0, p0 = self.position, self.momentum
        tl = self.time_like
        N = kwargs.get("steps", 50)
        dl = kwargs.get("delta", 0.5)
        rtol = kwargs.get("rtol", 1e-2)
        atol = kwargs.get("atol", 1e-2)
        order = kwargs.get("order", 2)
        omega = kwargs.get("omega", 1.0)
        sw = kwargs.get("suppress_warnings", False)
        steps = np.arange(N)

        geodint = GeodesicIntegrator(
            metric=g,
            metric_params=g_prms,
            q0=q0,
            p0=p0,
            time_like=tl,
            steps=N,
            delta=dl,
            rtol=rtol,
            atol=atol,
            order=order,
            omega=omega,
            suppress_warnings=sw,
        )

        for i in steps:
            geodint.step()

        vecs = np.array(geodint.results, dtype=float)

        q1 = vecs[:, 0]
        p1 = vecs[:, 1]
        results = np.hstack((q1, p1))
        # Ignoring
        # q2 = vecs[:, 2]
        # p2 = vecs[:, 3]

        if self.coords == "Cartesian":
            # Converting to Cartesian from Spherical Polar Coordinates
            # Note that momenta cannot be converted this way,
            # due to ambiguities in the signs of v_r and v_th (velocities)
            t, r, th, ph = q1.T
            pt, pr, pth, pph = p1.T
            x = r * np.sin(th) * np.cos(ph)
            y = r * np.sin(th) * np.sin(ph)
            z = r * np.cos(th)

            cart_results = np.vstack((t, x, y, z, pt, pr, pth, pph)).T

            return steps, cart_results

        return steps, results


class Nulllike(Geodesic):
    """
    Class for defining Null-like Geodesics

    """

    def __init__(
        self, metric, metric_params, position, momentum, return_cartesian=True, **kwargs
    ):
        """
        Constructor

        Parameters
        ----------
        metric : str
            Name of the metric. Currently, these metrics are supported:
            1. Schwarzschild
            2. Kerr
            3. KerrNewman
        metric_params : array_like
            Tuple of parameters to pass to the metric
            E.g., ``(a,)`` for Kerr
        position : array_like
            3-Position
            4-Position is initialized by taking ``t = 0.0``
        momentum : array_like
            3-Momentum
            4-Momentum is calculated automatically,
            considering the value of ``time_like``
        return_cartesian : bool, optional
            Whether to return calculated positions in Cartesian Coordinates
            This only affects the coordinates. The momenta dimensionless
            quantities, and are returned in Spherical Polar Coordinates.
            Defaults to ``True``
        kwargs : dict
            Keyword parameters for the Geodesic Integrator
            See 'Other Parameters' below.

        Other Parameters
        ----------------
        steps : int
            Number of integration steps
            Defaults to ``50``
        delta : float
            Initial integration step-size
            Defaults to ``0.5``
        rtol : float
            Relative Tolerance
            Defaults to ``1e-2``
        atol : float
            Absolute Tolerance
            Defaults to ``1e-2``
        order : int
            Integration Order
            Defaults to ``2``
        omega : float
            Coupling between Hamiltonian Flows
            Smaller values imply smaller integration error, but too
            small values can make the equation of motion non-integrable.
            For non-capture trajectories, ``omega = 1.0`` is recommended.
            For trajectories, that either lead to a capture or a grazing
            geodesic, a decreased value of ``0.01`` or less is recommended.
            Defaults to ``1.0``
        suppress_warnings : bool
            Whether to suppress warnings during simulation
            Warnings are shown for every step, where numerical errors
            exceed specified tolerance (controlled by ``rtol`` and ``atol``)
            Defaults to ``False``

        """
        super().__init__(
            metric=metric,
            metric_params=metric_params,
            position=position,
            momentum=momentum,
            time_like=False,
            return_cartesian=return_cartesian,
            **kwargs,
        )


class Timelike(Geodesic):
    """
    Class for defining Time-like Geodesics

    """

    def __init__(
        self, metric, metric_params, position, momentum, return_cartesian=True, **kwargs
    ):
        """
        Constructor

        Parameters
        ----------
        metric : str
            Name of the metric. Currently, these metrics are supported:
            1. Schwarzschild
            2. Kerr
            3. KerrNewman
        metric_params : array_like
            Tuple of parameters to pass to the metric
            E.g., ``(a,)`` for Kerr
        position : array_like
            3-Position
            4-Position is initialized by taking ``t = 0.0``
        momentum : array_like
            3-Momentum
            4-Momentum is calculated automatically,
            considering the value of ``time_like``
        return_cartesian : bool, optional
            Whether to return calculated positions in Cartesian Coordinates
            This only affects the coordinates. The momenta dimensionless
            quantities, and are returned in Spherical Polar Coordinates.
            Defaults to ``True``
        kwargs : dict
            Keyword parameters for the Geodesic Integrator
            See 'Other Parameters' below.

        Other Parameters
        ----------------
        steps : int
            Number of integration steps
            Defaults to ``50``
        delta : float
            Initial integration step-size
            Defaults to ``0.5``
        rtol : float
            Relative Tolerance
            Defaults to ``1e-2``
        atol : float
            Absolute Tolerance
            Defaults to ``1e-2``
        order : int
            Integration Order
            Defaults to ``2``
        omega : float
            Coupling between Hamiltonian Flows
            Smaller values imply smaller integration error, but too
            small values can make the equation of motion non-integrable.
            For non-capture trajectories, ``omega = 1.0`` is recommended.
            For trajectories, that either lead to a capture or a grazing
            geodesic, a decreased value of ``0.01`` or less is recommended.
            Defaults to ``1.0``
        suppress_warnings : bool
            Whether to suppress warnings during simulation
            Warnings are shown for every step, where numerical errors
            exceed specified tolerance (controlled by ``rtol`` and ``atol``)
            Defaults to ``False``

        """
        super().__init__(
            metric=metric,
            metric_params=metric_params,
            position=position,
            momentum=momentum,
            time_like=True,
            return_cartesian=return_cartesian,
            **kwargs,
        )


class StaticGeodesicPlotter:
    def __init__(self, ax=None, bh_colors=("#000", "#FFC"), draw_ergosphere=True):
        """
        Constructor

        Parameters
        ----------
        ax: ~matplotlib.axes.Axes
            Matplotlib Axes object
            To be deprecated in Version 0.5.0
            Since Version 0.4.0, `StaticGeodesicPlotter`
            automatically creates a new Axes Object.
            Defaults to ``None``
        bh_colors : tuple, optional
            2-Tuple, containing hexcodes (Strings) for the colors,
            used for the Black Hole Event Horizon (Outer) and Ergosphere (Outer)
            Defaults to ``("#000", "#FFC")``
        draw_ergosphere : bool, optional
            Whether to draw the ergosphere
            Defaults to `True`

        """
        self.ax = ax
        self.bh_colors = bh_colors
        self.draw_ergosphere = draw_ergosphere

        if ax is not None:
            warnings.warn(
                """
                Argument `ax` will be removed in Version 0.5.0.
                Since Version 0.4.0, `StaticGeodesicPlotter` automatically
                creates a new Axes Object.
                """,
                PendingDeprecationWarning,
            )

    def _draw_bh(self, a, figsize=(6, 6)):
        """
        Plots the Black Hole in 3D

        Parameters
        ----------
        a : float
            Dimensionless Spin Parameter of the Black Hole
            ``0 <= a <= 1``
        figsize : tuple, optional
            2-Tuple of Figure Size in inches
            Defaults to ``(6, 6)``

        """
        self.fig, self.ax = plt.subplots(figsize=figsize)
        fontsize = max(figsize) + 3
        self.fig.set_size_inches(figsize)
        self.ax = plt.axes(projection="3d")
        self.ax.set_xlabel("$X\\:(GM/c^2)$", fontsize=fontsize)
        self.ax.set_ylabel("$Y\\:(GM/c^2)$", fontsize=fontsize)
        self.ax.set_zlabel("$Z\\:(GM/c^2)$", fontsize=fontsize)

        theta, phi = np.linspace(0, 2 * np.pi, 50), np.linspace(0, np.pi, 50)
        THETA, PHI = np.meshgrid(theta, phi)

        # Outer Event Horizon
        rh_outer = 1 + np.sqrt(1 - a ** 2)

        XH = rh_outer * np.sin(PHI) * np.cos(THETA)
        YH = rh_outer * np.sin(PHI) * np.sin(THETA)
        ZH = rh_outer * np.cos(PHI)

        surface1 = self.ax.plot_surface(
            XH,
            YH,
            ZH,
            rstride=1,
            cstride=1,
            color=self.bh_colors[0],
            antialiased=False,
            alpha=0.2,
            label="BH Event Horizon (Outer)",
        )

        surface1._facecolors2d = surface1._facecolor3d
        surface1._edgecolors2d = surface1._edgecolor3d

        # Outer Ergosphere
        if self.draw_ergosphere:
            rE_outer = 1 + np.sqrt(1 - (a * np.cos(THETA) ** 2))

            XE = rE_outer * np.sin(PHI) * np.sin(THETA)
            YE = rE_outer * np.sin(PHI) * np.cos(THETA)
            ZE = rE_outer * np.cos(PHI)

            surface2 = self.ax.plot_surface(
                XE,
                YE,
                ZE,
                rstride=1,
                cstride=1,
                color=self.bh_colors[1],
                antialiased=False,
                alpha=0.1,
                label="BH Ergosphere (Outer)",
            )

            surface2._facecolors2d = surface2._facecolor3d
            surface2._edgecolors2d = surface2._edgecolor3d

    def _draw_bh_2D(self, a, figsize=(6, 6)):
        """
        Plots the Black Hole in 2D

        Parameters
        ----------
        a : float
            Dimensionless Spin Parameter of the Black Hole
            ``0 <= a <= 1``
        figsize : tuple, optional
            2-Tuple of Figure Size in inches
            Defaults to ``(6, 6)``

        """
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.fig.set_size_inches(figsize)

        theta = np.linspace(0, 2 * np.pi, 50)

        # Outer Event Horizon
        rh_outer = 1 + np.sqrt(1 - a ** 2)

        XH = rh_outer * np.sin(theta)
        YH = rh_outer * np.cos(theta)

        self.ax.fill(
            XH, YH, self.bh_colors[0], alpha=0.2, label="BH Event Horizon (Outer)"
        )

        # Outer Ergosphere
        if self.draw_ergosphere:
            rE_outer = 1 + np.sqrt(1 - (a * np.cos(theta) ** 2))

            XE = rE_outer * np.sin(theta)
            YE = rE_outer * np.cos(theta)

            self.ax.fill(
                XE, YE, self.bh_colors[1], alpha=0.1, label="BH Ergosphere (Outer)"
            )

    def plot(
        self,
        geodesic,
        figsize=(6, 6),
        color="#{:06x}".format(random.randint(0, 0xFFFFFF)),
    ):
        """
        Plots the Geodesic

        Parameters
        ----------
        geodesic : einsteinpy.geodesic.*
            Geodesic Object
        figsize : tuple, optional
            2-Tuple of Figure Size in inches
            Defaults to ``(6, 6)``
        color : str, optional
            Hexcode (String) for the color of the
            dashed lines, that represent the Geodesic
            Picks a random color by default

        """
        a = geodesic.metric_params[0]
        self._draw_bh(a, figsize)

        traj = geodesic.trajectory[1]
        x = traj[:, 1]
        y = traj[:, 2]
        z = traj[:, 3]

        self.ax.plot(x, y, z, "--", color=color, label=geodesic.kind + " Geodesic")

    def plot2D(
        self,
        geodesic,
        coordinates=(1, 2),
        figsize=(6, 6),
        color="#{:06x}".format(random.randint(0, 0xFFFFFF)),
    ):
        """
        Plots the Geodesic in 2D

        Parameters
        ----------
        geodesic : einsteinpy.geodesic.*
            Geodesic Object
        coordinates : tuple, optional
            2-Tuple, containing labels for coordinates to plot
            Labels for ``X1, X2, X3`` are ``(1, 2, 3)``
            Defaults to ``(1, 2)`` (X, Y)
        figsize : tuple, optional
            2-Tuple of Figure Size in inches
            Defaults to ``(6, 6)``
        color : str, optional
            Hexcode (String) for the color of the
            dashed lines, that represent the Geodesic
            Picks a random color by default

        Raises
        ------
        IndexError
            If indices in ``coordinates`` do not take values from ``(1, 2, 3)``

        """
        a = geodesic.metric_params[0]
        self._draw_bh_2D(a, figsize)

        traj = geodesic.trajectory[1]
        A = coordinates[0]
        B = coordinates[1]

        if A not in (1, 2, 3) or B not in (1, 2, 3):
            raise IndexError(
                """
                Please ensure, that indices in `coordinates` take two of these values: `(1, 2, 3)`.
                Indices for `X1, X2, X3` are `(1, 2, 3)`.
                """
            )

        fontsize = max(figsize) + 3
        self.ax.set_xlabel(f"$X{coordinates[0]}\\:(GM/c^2)$", fontsize=fontsize)
        self.ax.set_ylabel(f"$X{coordinates[1]}\\:(GM/c^2)$", fontsize=fontsize)

        self.ax.plot(
            traj[:, A], traj[:, B], "--", color=color, label=geodesic.kind + " Geodesic"
        )

    def parametric_plot(
        self, geodesic, figsize=(8, 6), colors=("#00FFFF", "#FF00FF", "#FFFF00")
    ):
        """
        Plots the coordinates of the Geodesic, against Affine Parameter

        Parameters
        ----------
        geodesic : einsteinpy.geodesic.*
            Geodesic Object
        figsize : tuple, optional
            2-Tuple of Figure Size in inches
            Defaults to ``(8, 6)``
        colors : tuple, optional
            3-Tuple, containing hexcodes (Strings) for the color
            of the lines, for each of the 3 coordinates
            Defaults to ``("#00FFFF", "#FF00FF", "#00FFFF")``

        """
        self.fig, self.ax = plt.subplots(figsize=figsize)
        fontsize = max(figsize) + 3
        self.fig.set_size_inches(figsize)
        self.ax = plt.axes()
        self.ax.set_xlabel(r"Affine Paramter, $\lambda$", fontsize=fontsize)
        self.ax.set_ylabel("Coordinates", fontsize=fontsize)

        coords = geodesic.coords
        traj = geodesic.trajectory
        lambdas = traj[0]
        X1 = traj[1][:, 1]
        X2 = traj[1][:, 2]
        X3 = traj[1][:, 3]

        self.ax.plot(lambdas, X1, color=colors[0], label=f"X1 ({coords})")
        self.ax.plot(lambdas, X2, color=colors[1], label=f"X2 ({coords})")
        self.ax.plot(lambdas, X3, color=colors[2], label=f"X3 ({coords})")

    def animate(
        self, geodesic, interval=10, color="#{:06x}".format(random.randint(0, 0xFFFFFF))
    ):
        """
        Parameters
        ----------
        geodesic : einsteinpy.geodesic.*
            Geodesic Object
        interval : int, optional
            Time (in milliseconds) between frames
            Defaults to ``10``
        color : str, optional
            Hexcode (String) for the color of the
            dashed lines, that represent the Geodesic
            Picks a random color by default

        """
        a = geodesic.metric_params[0]
        self._draw_bh(a)

        traj = geodesic.trajectory
        x = traj[1][:, 1]
        y = traj[1][:, 2]
        z = traj[1][:, 3]
        N = x.shape[0]

        x_max, x_min = max(x), min(x)
        y_max, y_min = max(y), min(y)
        z_max, z_min = max(z), min(z)
        margin_x = (x_max - x_min) * 0.2
        margin_y = (y_max - y_min) * 0.2
        margin_z = (z_max - z_min) * 0.2

        self.ax.set_xlim3d([x_min - margin_x, x_max + margin_x])
        self.ax.set_ylim3d([y_min - margin_y, y_max + margin_y])
        self.ax.set_zlim3d([z_min - margin_z, z_max + margin_z])

        data = traj[1][:, 1:4].T
        (line,) = self.ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])

        def _update(num, data, line):
            line.set_data(data[:2, :num])
            line.set_3d_properties(data[2, :num])

            return (line,)

        self.ani = FuncAnimation(
            self.fig, _update, N, fargs=(data, line), interval=interval, blit=True
        )

    def show(self, azim=-60, elev=30):
        """
        Adjusts the 3D view of the plot and \
        shows the plot during runtime. For Parametric Plots,
        only the plot is displayed.

        Parameters
        ----------
        azim : float, optional
            Azimuthal viewing angle
            Defaults to ``-60`` Degrees

        elev : float, optional
            Elevation viewing angle
            Defaults to ``30`` Degrees

        """
        figsize = self.fig.get_size_inches()
        fontsize = max(figsize) + 1.5
        if self.ax.name == "3d":
            self.ax.view_init(azim=azim, elev=elev)
        plt.legend(prop={"size": fontsize})

        plt.show()

    def clear(self):
        """
        Clears plot during runtime

        """
        self.fig.clf()

    def save(self, name="Geodesic.png"):
        """
        Saves plot locally
        Should be called before ``show()``, as
        ``show()`` erases current figure's contents.

        Parameters
        ----------
        name : str, optional
            Name of the file, with extension
            Defaults to ``Geodesic.png``

        """
        if self.ax.name != "3d" and name == "Geodesic.png":
            name = "Parametric.png"

        plt.savefig(name)