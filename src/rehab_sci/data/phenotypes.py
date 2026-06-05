"""Multivariate growth mixture model (GMM) for observed-trajectory phenotyping (G3).

Where ``data/archetypes.py`` clusters model-*predicted* recovery curves with k-means,
this module discovers recovery phenotypes from the *observed* early-recovery trajectories
— jointly over SCIM-total and total motor score — by fitting a mixture of linear mixed
models (a growth mixture model, GMM)::

    y_{i,m}(t) = beta_{k,m} . phi(t)  +  b_{i,m} . psi(t)  +  eps
                 \\____ class mean ___/   \\__ random effect _/

  * ``phi(t)`` = polynomial basis (degree ``D``) for the class-k fixed mean of measure m;
  * ``psi(t)`` = ``[1, t]`` random intercept + slope, per measure, drawn ~ N(0, G);
  * ``eps``    ~ N(0, sigma2_m) measurement noise.

Random effects are integrated out analytically, so each individual's observed cells are
marginally Gaussian with a structured covariance ``V_i = Z_i G Z_i' + R_i``.  ``G`` and
``sigma2`` are class-invariant (the robust, identifiable GMM specification that avoids the
variance-collapse non-identifiability), so ``V_i`` does not depend on the class ``k`` and is
computed once per individual per EM iteration; only the mean ``Phi_i beta_k`` is class-
specific.  ``G`` is block-diagonal across the two measures (their random effects are
independent given the class) — a structural constraint re-imposed each M-step.

Missingness is native: an individual contributes only its observed (measure, timepoint)
cells, so phenotypes are not biased by imputation (the key methodological advantage over a
k-means-on-imputed-curves approach).  Fit by EM with random restarts; the number of classes
``K`` and the polynomial degree ``D`` are chosen by BIC.

The fitted parameters live in :class:`GMMParams`; :func:`fit` selects nothing (one K, one D),
:func:`select` sweeps K x D by BIC.  :func:`predict_proba` assigns new (possibly partially
observed) individuals to phenotypes — this is what the part-2 patient surface will call.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------
# Trajectory window + time axis
# ----------------------------------------------------------------------------------------

# Measures phenotyped jointly (both ~0-100, kept in raw units so the class curves are
# directly interpretable; per-measure residual variance absorbs any scale difference).
MEASURES: tuple[str, ...] = ("SCIM_total", "TotalMotor")

# Early-recovery window (nominal timepoint slots) and their day offsets.  The terminal
# ``discharge`` slot is intentionally excluded — discharge timing varies with LOS, so it is
# treated as a *conditioned outcome*, not part of the trajectory shape.
WINDOW: tuple[str, ...] = ("0day", "72h", "2w", "4w", "6w", "2m", "3m", "4m", "5m", "6m")
WINDOW_DAYS: dict[str, int] = {
    "0day": 0, "72h": 3, "2w": 14, "4w": 28, "6w": 42,
    "2m": 60, "3m": 90, "4m": 120, "5m": 150, "6m": 180,
}
TIME_SCALE: float = 180.0  # scale days -> [0, 1] so the polynomial basis is well-conditioned

N_RANDOM: int = 2  # random effects per measure: intercept + linear slope
_VAR_FLOOR: float = 1e-4


def scaled_time(timepoint: str) -> float:
    """Scaled time in [0, 1] for a window timepoint slot."""
    return WINDOW_DAYS[timepoint] / TIME_SCALE


# ----------------------------------------------------------------------------------------
# Per-individual design matrices
# ----------------------------------------------------------------------------------------

def _poly_basis(t: np.ndarray, degree: int) -> np.ndarray:
    """Vandermonde basis ``[1, t, t^2, ..., t^degree]`` (shape ``(n, degree+1)``)."""
    return np.vander(t, N=degree + 1, increasing=True)


@dataclass
class GMMData:
    """Pre-built per-individual design matrices for the mixture of linear mixed models.

    Each list entry corresponds to one individual; ``meas`` holds the 0/1 measure index of
    every observed cell so the M-step can pool residual variance per measure.
    """

    Phi: list[np.ndarray]   # (n_i, M*(degree+1)) fixed-effect design
    Z: list[np.ndarray]     # (n_i, M*N_RANDOM)   random-effect design
    y: list[np.ndarray]     # (n_i,)              stacked observed values
    meas: list[np.ndarray]  # (n_i,)              measure index (0..M-1) per cell
    keys: list[int]         # KeyRecordNumber per individual (assignment join key)
    degree: int
    n_measures: int

    @property
    def N(self) -> int:
        return len(self.y)

    @property
    def p_fixed(self) -> int:
        return self.n_measures * (self.degree + 1)

    @property
    def p_random(self) -> int:
        return self.n_measures * N_RANDOM


def build_individual_design(
    times: np.ndarray, meas: np.ndarray, degree: int, n_measures: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(Phi, Z)`` for one individual from per-cell scaled times + measure indices."""
    n = times.shape[0]
    db = degree + 1
    phi = _poly_basis(times, degree)           # (n, db)
    psi = _poly_basis(times, N_RANDOM - 1)      # (n, N_RANDOM) = [1, t]
    Phi = np.zeros((n, n_measures * db))
    Z = np.zeros((n, n_measures * N_RANDOM))
    for m in range(n_measures):
        rows = meas == m
        if not rows.any():
            continue
        Phi[np.ix_(rows, range(m * db, (m + 1) * db))] = phi[rows]
        Z[np.ix_(rows, range(m * N_RANDOM, (m + 1) * N_RANDOM))] = psi[rows]
    return Phi, Z


def build_gmm_data(
    long_df: pd.DataFrame,
    cohort_keys: list[int],
    degree: int,
    measures: tuple[str, ...] = MEASURES,
) -> GMMData:
    """Assemble :class:`GMMData` from the longitudinal frame for the given cohort.

    For each episode in ``cohort_keys`` collect every observed (measure, window-timepoint)
    cell.  Missing cells are simply absent — the model handles them natively.
    """
    n_measures = len(measures)
    sub = long_df[long_df["TIME_Name"].isin(WINDOW)][["KeyRecordNumber", "TIME_Name", *measures]]
    sub = sub.copy()
    sub["_t"] = sub["TIME_Name"].map(scaled_time)
    by_key = {k: g for k, g in sub.groupby("KeyRecordNumber")}

    Phi_l, Z_l, y_l, meas_l, keys_l = [], [], [], [], []
    for key in cohort_keys:
        g = by_key.get(key)
        if g is None:
            continue
        times, meas_idx, vals = [], [], []
        for m, col in enumerate(measures):
            obs = g[["_t", col]].dropna(subset=[col])
            times.append(obs["_t"].to_numpy())
            meas_idx.append(np.full(len(obs), m))
            vals.append(obs[col].to_numpy(dtype=float))
        times = np.concatenate(times)
        meas_idx = np.concatenate(meas_idx)
        vals = np.concatenate(vals)
        if times.shape[0] == 0:
            continue
        Phi, Z = build_individual_design(times, meas_idx, degree, n_measures)
        Phi_l.append(Phi)
        Z_l.append(Z)
        y_l.append(vals)
        meas_l.append(meas_idx)
        keys_l.append(int(key))

    return GMMData(Phi_l, Z_l, y_l, meas_l, keys_l, degree, n_measures)


# ----------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------

@dataclass
class GMMParams:
    """Fitted growth-mixture-model parameters (class-invariant G & sigma2)."""

    K: int
    degree: int
    n_measures: int
    pi: np.ndarray       # (K,)                      mixing proportions
    beta: np.ndarray     # (K, M*(degree+1))         class fixed-effect coefficients
    G: np.ndarray        # (M*N_RANDOM, M*N_RANDOM)  shared random-effect covariance (block-diag)
    sigma2: np.ndarray   # (M,)                      per-measure residual variance

    def n_free_params(self) -> int:
        rb = N_RANDOM * (N_RANDOM + 1) // 2          # free entries of one measure's G block
        return (
            (self.K - 1)                              # mixing
            + self.K * self.n_measures * (self.degree + 1)  # fixed effects
            + self.n_measures * rb                    # shared block-diagonal G
            + self.n_measures                         # shared residual variances
        )


def _block_diag_project(G: np.ndarray, n_measures: int) -> np.ndarray:
    """Zero the cross-measure blocks so random effects are independent across measures."""
    out = np.zeros_like(G)
    r = N_RANDOM
    for m in range(n_measures):
        sl = slice(m * r, (m + 1) * r)
        out[sl, sl] = G[sl, sl]
    return out


# ----------------------------------------------------------------------------------------
# EM
# ----------------------------------------------------------------------------------------

def _cov_and_inv(Z: np.ndarray, meas: np.ndarray, G: np.ndarray, sigma2: np.ndarray):
    """Marginal covariance ``V = Z G Z' + diag(sigma2[meas])`` with its inverse + logdet."""
    V = Z @ G @ Z.T
    V[np.diag_indices_from(V)] += sigma2[meas]
    # symmetric PD -> Cholesky; jitter fallback guards rare numerical non-PD.
    try:
        L = np.linalg.cholesky(V)
    except np.linalg.LinAlgError:
        V[np.diag_indices_from(V)] += 1e-6
        L = np.linalg.cholesky(V)
    logdet = 2.0 * np.log(np.diag(L)).sum()
    Linv = np.linalg.inv(L)
    Vinv = Linv.T @ Linv
    return Vinv, logdet


def _e_step(data: GMMData, p: GMMParams):
    """Posterior class responsibilities + observed-data log-likelihood.

    Returns ``(resp[N,K], loglik, Vinv_list)``; ``Vinv_list`` is reused by the M-step
    (V is class-invariant, so it is built once per individual here).
    """
    N, K = data.N, p.K
    log_pi = np.log(p.pi)
    resp = np.empty((N, K))
    Vinv_list = []
    loglik = 0.0
    for i in range(N):
        Phi, Z, y, meas = data.Phi[i], data.Z[i], data.y[i], data.meas[i]
        Vinv, logdet = _cov_and_inv(Z, meas, p.G, p.sigma2)
        Vinv_list.append(Vinv)
        n = y.shape[0]
        base = -0.5 * (n * np.log(2.0 * np.pi) + logdet)
        logN = np.empty(K)
        for k in range(K):
            r = y - Phi @ p.beta[k]
            logN[k] = base - 0.5 * (r @ Vinv @ r)
        a = log_pi + logN
        amax = a.max()
        lse = amax + np.log(np.exp(a - amax).sum())
        resp[i] = np.exp(a - lse)
        loglik += lse
    return resp, loglik, Vinv_list


def _m_step(data: GMMData, resp: np.ndarray, p: GMMParams, Vinv_list: list[np.ndarray]) -> GMMParams:
    """ECM update: pi, then GLS beta, then random-effect covariance G + residual sigma2."""
    N, K = data.N, p.K
    M, r_dim = data.n_measures, data.p_random

    pi = resp.sum(0) / N
    pi = np.clip(pi, 1e-6, None)
    pi /= pi.sum()

    # --- beta_k via responsibility-weighted GLS (uses old V) ---
    p_fix = data.p_fixed
    A = np.zeros((K, p_fix, p_fix))
    b = np.zeros((K, p_fix))
    PtVinv = []
    for i in range(N):
        Phi, Vinv, y = data.Phi[i], Vinv_list[i], data.y[i]
        ptv = Phi.T @ Vinv               # (p_fix, n)
        PtVinv.append(ptv)
        H = ptv @ Phi                    # (p_fix, p_fix)
        g = ptv @ y                      # (p_fix,)
        for k in range(K):
            A[k] += resp[i, k] * H
            b[k] += resp[i, k] * g
    beta = np.empty((K, p_fix))
    for k in range(K):
        beta[k] = np.linalg.solve(A[k] + 1e-8 * np.eye(p_fix), b[k])

    # --- random-effect covariance G + residual sigma2 (EM for the LMM, weighted by resp) ---
    G_acc = np.zeros((r_dim, r_dim))
    sse = np.zeros(M)
    cnt = np.zeros(M)
    for i in range(N):
        Phi, Z, y, meas, Vinv = data.Phi[i], data.Z[i], data.y[i], data.meas[i], Vinv_list[i]
        GZtVinv = p.G @ Z.T @ Vinv                  # (r_dim, n)
        Cov_b = p.G - GZtVinv @ Z @ p.G             # (r_dim, r_dim) posterior cov (class-invariant)
        zcz = np.einsum("nj,jl,nl->n", Z, Cov_b, Z)  # Z Cov_b Z' diagonal, per cell
        for k in range(K):
            w = resp[i, k]
            if w < 1e-12:
                continue
            r = y - Phi @ beta[k]
            bhat = GZtVinv @ r                       # (r_dim,) posterior mean of random effects
            G_acc += w * (np.outer(bhat, bhat) + Cov_b)
            resid = r - Z @ bhat                     # (n,)
            esq = resid ** 2 + zcz                   # expected squared residual per cell
            for m in range(M):
                cells = meas == m
                if cells.any():
                    sse[m] += w * esq[cells].sum()
                    cnt[m] += w * cells.sum()
    G = _block_diag_project(G_acc / N, M)
    # keep G strictly PD on the diagonal blocks
    G[np.diag_indices_from(G)] = np.maximum(np.diag(G), _VAR_FLOOR)
    sigma2 = np.maximum(sse / np.maximum(cnt, 1.0), _VAR_FLOOR)

    return GMMParams(K, data.degree, M, pi, beta, G, sigma2)


def _init_params(data: GMMData, K: int, resp0: np.ndarray) -> GMMParams:
    """Seed parameters from an initial (hard or soft) responsibility matrix."""
    N, M = data.N, data.n_measures
    p_fix = data.p_fixed
    pi = np.clip(resp0.sum(0) / N, 1e-6, None)
    pi /= pi.sum()
    # OLS (V = I) per class for an initial beta + residual scale.
    A = np.zeros((K, p_fix, p_fix))
    b = np.zeros((K, p_fix))
    for i in range(N):
        Phi, y = data.Phi[i], data.y[i]
        H, g = Phi.T @ Phi, Phi.T @ y
        for k in range(K):
            A[k] += resp0[i, k] * H
            b[k] += resp0[i, k] * g
    beta = np.empty((K, p_fix))
    for k in range(K):
        beta[k] = np.linalg.solve(A[k] + 1e-6 * np.eye(p_fix), b[k])
    sse = np.zeros(M)
    cnt = np.zeros(M)
    for i in range(N):
        Phi, y, meas = data.Phi[i], data.y[i], data.meas[i]
        k = int(np.argmax(resp0[i]))
        resid = y - Phi @ beta[k]
        for m in range(M):
            cells = meas == m
            if cells.any():
                sse[m] += (resid[cells] ** 2).sum()
                cnt[m] += cells.sum()
    sigma2 = np.maximum(sse / np.maximum(cnt, 1.0), 1.0)
    G = np.eye(data.p_random) * float(sigma2.mean())
    return GMMParams(K, data.degree, M, pi, beta, G, sigma2)


def _crude_features(data: GMMData) -> np.ndarray:
    """Per-individual summary (per-measure mean + OLS slope) for k-means initialization."""
    M = data.n_measures
    feats = np.zeros((data.N, 2 * M))
    for i in range(data.N):
        Z, y, meas = data.Z[i], data.y[i], data.meas[i]
        t = Z[:, 1]  # scaled time sits in the intercept+slope block's slope column
        for m in range(M):
            cells = meas == m
            if cells.sum() == 0:
                continue
            ym, tm = y[cells], t[cells]
            feats[i, 2 * m] = ym.mean()
            if cells.sum() >= 2 and np.ptp(tm) > 0:
                feats[i, 2 * m + 1] = np.polyfit(tm, ym, 1)[0]
    # standardize columns
    mu, sd = feats.mean(0), feats.std(0)
    sd[sd == 0] = 1.0
    return (feats - mu) / sd


def fit_once(
    data: GMMData, K: int, resp0: np.ndarray, *, max_iter: int = 200, tol: float = 1e-4
):
    """Run EM to convergence from one initialization.  Returns ``(params, resp, loglik)``."""
    p = _init_params(data, K, resp0)
    prev = -np.inf
    resp = resp0
    loglik = prev
    for _ in range(max_iter):
        resp, loglik, Vinv_list = _e_step(data, p)
        if loglik - prev < tol and prev > -np.inf:
            break
        prev = loglik
        p = _m_step(data, resp, p, Vinv_list)
    resp, loglik, _ = _e_step(data, p)
    return p, resp, loglik


def fit(
    data: GMMData,
    K: int,
    *,
    n_restarts: int = 12,
    seed: int = 20260518,
    max_iter: int = 200,
    tol: float = 1e-4,
):
    """Fit ``K``-class GMM with multiple restarts; keep the highest-likelihood solution.

    Returns ``(best_params, best_resp, best_loglik)``.  Restart 0 seeds from k-means on
    crude per-individual features; later restarts perturb those labels (and, for higher
    indices, use fully random soft assignments) so EM escapes poor local optima.
    """
    from sklearn.cluster import KMeans

    feats = _crude_features(data)
    best = None
    for r in range(n_restarts):
        rng = np.random.default_rng(seed + r)
        if r < max(1, n_restarts // 2):
            km = KMeans(n_clusters=K, n_init=5, random_state=seed + r).fit(feats)
            resp0 = np.full((data.N, K), 0.05)
            resp0[np.arange(data.N), km.labels_] = 1.0
            resp0 += rng.dirichlet(np.ones(K), size=data.N) * 0.10
        else:
            resp0 = rng.dirichlet(np.ones(K) * 2.0, size=data.N)
        resp0 /= resp0.sum(1, keepdims=True)
        try:
            p, resp, ll = fit_once(data, K, resp0, max_iter=max_iter, tol=tol)
        except np.linalg.LinAlgError:
            continue
        if best is None or ll > best[2]:
            best = (p, resp, ll)
    if best is None:
        raise RuntimeError("GMM fit failed for all restarts")
    return best


def predict_proba(data: GMMData, p: GMMParams) -> np.ndarray:
    """Posterior phenotype membership for (possibly partially observed) individuals."""
    resp, _, _ = _e_step(data, p)
    return resp


# ----------------------------------------------------------------------------------------
# Selection, diagnostics, class curves, ordering
# ----------------------------------------------------------------------------------------

def bic(loglik: float, n_free: int, N: int) -> float:
    return -2.0 * loglik + n_free * np.log(N)


def diagnostics(resp: np.ndarray) -> dict:
    """GMM separation diagnostics: relative entropy + per-class APPA + min class share."""
    N, K = resp.shape
    r = np.clip(resp, 1e-12, 1.0)
    ent = -(r * np.log(r)).sum()
    rel_entropy = 1.0 - ent / (N * np.log(K)) if K > 1 else 1.0
    hard = resp.argmax(1)
    appa = []
    shares = []
    for k in range(K):
        sel = hard == k
        appa.append(float(resp[sel, k].mean()) if sel.any() else 0.0)
        shares.append(float(sel.mean()))
    return {
        "relative_entropy": float(rel_entropy),
        "appa": appa,
        "min_class_share": float(min(shares)),
        "class_shares": shares,
    }


def class_means(p: GMMParams, timepoints: tuple[str, ...] = WINDOW) -> np.ndarray:
    """Fitted class mean trajectories, shape ``(K, n_measures, len(timepoints))``."""
    t = np.array([scaled_time(tp) for tp in timepoints])
    phi = _poly_basis(t, p.degree)            # (T, degree+1)
    db = p.degree + 1
    out = np.empty((p.K, p.n_measures, len(timepoints)))
    for k in range(p.K):
        for m in range(p.n_measures):
            out[k, m] = phi @ p.beta[k, m * db:(m + 1) * db]
    return out


def class_support(
    long_df: pd.DataFrame,
    assignments: dict[int, int],
    k: int,
    measures: tuple[str, ...] = MEASURES,
    *,
    min_coverage: float = 0.20,
) -> np.ndarray:
    """Last window index per (class, measure) where >= ``min_coverage`` of the class is observed.

    Group-based trajectory means are only interpretable over each class's *observed* support;
    past it the polynomial extrapolates (a degree-2 basis can diverge — e.g. a fast-recovery
    class whose members are discharged early plunges far below zero at 6m).  For each phenotype
    and measure this returns the largest window index ``w`` such that at least ``min_coverage``
    of the class's episodes have an observed value at ``w`` — the point beyond which the class
    curve must not be drawn.  Returns an int array of shape ``(k, len(measures))``.
    """
    wkey = {w: i for i, w in enumerate(WINDOW)}
    sizes = np.zeros(k, dtype=int)
    for cls in assignments.values():
        if 0 <= cls < k:
            sizes[cls] += 1
    sub = long_df[long_df["TIME_Name"].isin(WINDOW)].copy()
    sub["_cls"] = sub["KeyRecordNumber"].map(assignments)
    sub["_wi"] = sub["TIME_Name"].map(wkey)
    sub = sub[sub["_cls"].notna()]
    support = np.zeros((k, len(measures)), dtype=int)
    for mi, col in enumerate(measures):
        obs = sub[sub[col].notna()][["KeyRecordNumber", "_cls", "_wi"]].drop_duplicates()
        counts = obs.groupby(["_cls", "_wi"]).size()
        for cls in range(k):
            frac = np.zeros(len(WINDOW))
            if sizes[cls] > 0 and cls in counts.index.get_level_values(0):
                for wi, c in counts.loc[cls].items():
                    frac[int(wi)] = c / sizes[cls]
            ok = np.where(frac >= min_coverage)[0]
            support[cls, mi] = int(ok.max()) if ok.size else 0
    return support


def order_by_discharge(p: GMMParams, resp: np.ndarray, support: np.ndarray | None = None):
    """Relabel classes by ascending SCIM recovery (class 0 = lowest recovery).

    Ranking is the SCIM class mean (clipped to [0, 100]) at the *latest timepoint where every
    class is still within observed support* — so the order never depends on out-of-support
    polynomial extrapolation (an early-discharge class whose quadratic mean dives below zero at
    6m would otherwise be mislabeled the *worst* recoverer).  Without ``support`` (an
    ``(K, n_measures)`` index array, e.g. from :func:`class_support`) it falls back to the final
    window timepoint.  Returns ``(reordered_params, reordered_resp, order)`` with ``order[new] = old``.
    """
    scim = np.clip(class_means(p)[:, 0, :], 0.0, 100.0)   # (K, T) measure 0 = SCIM
    idx = int(support[:, 0].min()) if support is not None else scim.shape[1] - 1
    order = np.argsort(scim[:, idx], kind="stable")
    p2 = GMMParams(
        p.K, p.degree, p.n_measures,
        p.pi[order].copy(), p.beta[order].copy(), p.G.copy(), p.sigma2.copy(),
    )
    return p2, resp[:, order].copy(), order


def select(
    data_by_degree: dict[int, GMMData],
    k_range: tuple[int, ...],
    degrees: tuple[int, ...],
    *,
    n_restarts: int = 12,
    seed: int = 20260518,
    min_class_share: float = 0.05,
    progress=None,
):
    """Sweep ``K x degree`` by BIC.  Returns ``(best_key, fits, table)``.

    ``fits[(K, degree)] = (params, resp, loglik)``; ``table`` is a list of per-cell metric
    dicts.  The best key minimizes BIC among cells whose smallest class retains at least
    ``min_class_share`` of the cohort (guards against degenerate tiny classes).  ``progress``,
    if given, is called with each completed cell's metric dict (for logging).
    """
    fits: dict[tuple[int, int], tuple] = {}
    table: list[dict] = []
    best_key = None
    best_bic = np.inf
    for degree in degrees:
        data = data_by_degree[degree]
        for K in k_range:
            p, resp, ll = fit(data, K, n_restarts=n_restarts, seed=seed)
            n_free = p.n_free_params()
            b = bic(ll, n_free, data.N)
            diag = diagnostics(resp)
            row = {
                "K": K, "degree": degree, "loglik": float(ll), "n_params": n_free,
                "bic": float(b), "relative_entropy": diag["relative_entropy"],
                "min_class_share": diag["min_class_share"],
            }
            table.append(row)
            fits[(K, degree)] = (p, resp, ll)
            if progress is not None:
                progress(row)
            if diag["min_class_share"] >= min_class_share and b < best_bic:
                best_bic = b
                best_key = (K, degree)
    if best_key is None:  # all cells degenerate -> fall back to global BIC minimum
        best_key = min(fits, key=lambda kd: bic(fits[kd][2], fits[kd][0].n_free_params(),
                                                 data_by_degree[kd[1]].N))
    return best_key, fits, table


def phenotype_summary(ep_eligible: pd.DataFrame, assignments: dict[int, int], k: int) -> list[dict]:
    """Per-phenotype demographics + conditioned outcomes (mirrors archetype_summary).

    Keyed on hard assignment (argmax posterior).  Returns one dict per class id with
    ``n``, ``mean_age``, ``pct_tetra``, ``ais_distribution``, ``mean_discharge_scim``,
    ``median_discharge_scim``, ``mean_los``.
    """
    df = ep_eligible.copy()
    df["_pheno"] = df["KeyRecordNumber"].map(assignments)
    summaries: list[dict] = []
    for pid in range(k):
        sub = df[df["_pheno"] == pid]
        n = len(sub)
        age = pd.to_numeric(sub["年齢"], errors="coerce")
        para = sub["対麻痺_四肢麻痺"].astype(str)
        ais = sub["AIS"].dropna().astype(str)
        scim = pd.to_numeric(sub.get("y_discharge_scim"), errors="coerce").dropna()
        los = pd.to_numeric(sub.get("LOS_days"), errors="coerce").dropna()
        ais_dist = {}
        if len(ais) > 0:
            vc = ais.value_counts(normalize=True)
            for grade in ["A", "B", "C", "D", "E"]:
                ais_dist[grade] = float(vc.get(grade, 0.0))
        summaries.append({
            "id": pid,
            "n": n,
            "mean_age": float(age.mean()) if len(age.dropna()) > 0 else None,
            "pct_tetra": float((para == "TETRA").sum() / n * 100) if n > 0 else 0.0,
            "ais_distribution": ais_dist,
            "mean_discharge_scim": float(scim.mean()) if len(scim) > 0 else None,
            "median_discharge_scim": float(scim.median()) if len(scim) > 0 else None,
            "mean_los": float(los.mean()) if len(los) > 0 else None,
        })
    return summaries
