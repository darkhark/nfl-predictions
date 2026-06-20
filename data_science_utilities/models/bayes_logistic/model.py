"""PyMC Bayesian logistic regression: two prior variants, mirroring the BART
predict pattern (pm.Data + set_data + posterior-predictive of a Deterministic p).
"""
import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

RANDOM_SEED = 32


def build_model(X, y, prior="normal", *, slab_scale=2.0, slab_df=4.0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n_features = X.shape[1]
    with pm.Model() as m:
        X_data = pm.Data("X", X)
        alpha = pm.Normal("alpha", 0.0, 1.5)
        if prior == "normal":
            beta = pm.Normal("beta", 0.0, 1.0, shape=n_features)
        elif prior == "horseshoe":
            # Regularized (Finnish) horseshoe, non-centered (Piironen & Vehtari 2017).
            tau = pm.HalfCauchy("tau", beta=1.0)
            lam = pm.HalfCauchy("lam", beta=1.0, shape=n_features)
            c2 = pm.InverseGamma("c2", alpha=slab_df / 2, beta=slab_df / 2 * slab_scale**2)
            lam_tilde = pt.sqrt(c2 * lam**2 / (c2 + tau**2 * lam**2))
            z = pm.Normal("z", 0.0, 1.0, shape=n_features)
            beta = pm.Deterministic("beta", z * lam_tilde * tau)
        else:
            raise ValueError(f"unknown prior {prior!r}; use 'normal' or 'horseshoe'")
        logit_p = alpha + pm.math.dot(X_data, beta)
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p))
        pm.Bernoulli("y", p=p, observed=y, shape=X_data.shape[0])
    return m


def sample(pm_model, *, draws=2000, tune=2000, chains=4, seed=RANDOM_SEED, target_accept=0.9):
    with pm_model:
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=chains,
            random_seed=seed, target_accept=target_accept, progressbar=False,
        )
    return idata


def predict(pm_model, idata, X_new, seed=RANDOM_SEED):
    X_new = np.asarray(X_new, dtype=float)
    with pm_model:
        pm.set_data({"X": X_new})
        ppc = pm.sample_posterior_predictive(
            idata, var_names=["p"], random_seed=seed, progressbar=False
        )
    post_p = ppc.posterior_predictive["p"]
    return (
        post_p.mean(dim=["chain", "draw"]).to_numpy(),
        post_p.std(dim=["chain", "draw"]).to_numpy(),
    )


def diagnostics(idata, var_names=("alpha", "beta")):
    summ = az.summary(idata, var_names=list(var_names))
    n_div = int(idata.sample_stats["diverging"].sum())
    max_rhat = float(summ["r_hat"].max())
    min_ess_bulk = float(summ["ess_bulk"].min())
    return {
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "n_divergences": n_div,
        "passed": bool(max_rhat < 1.01 and n_div == 0 and min_ess_bulk > 400),
    }
