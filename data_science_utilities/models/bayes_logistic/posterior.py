"""Expose the fitted posterior for diagnostics and for seeding a later model's prior.

The JSON summary carries both the posterior moments (on the standardized scale) and
the standardizer moments per feature, so a downstream model can rebuild a prior on
the standardized *or* raw scale.
"""
import json

import arviz as az


def coefficient_summary(idata, feature_names_out, preprocessor, prior):
    post = idata.posterior
    beta_mean = post["beta"].mean(dim=["chain", "draw"]).to_numpy()
    beta_sd = post["beta"].std(dim=["chain", "draw"]).to_numpy()
    alpha_mean = float(post["alpha"].mean(dim=["chain", "draw"]).to_numpy())
    alpha_sd = float(post["alpha"].std(dim=["chain", "draw"]).to_numpy())
    coefficients = []
    for i, name in enumerate(feature_names_out):
        coefficients.append({
            "feature": name,
            "mean": float(beta_mean[i]),
            "sd": float(beta_sd[i]),
            "standardizer_mean": float(preprocessor.means_[i]),
            "standardizer_std": float(preprocessor.stds_[i]),
        })
    return {
        "prior": prior,
        "scale": "standardized",
        "n_features": len(feature_names_out),
        "intercept": {"mean": alpha_mean, "sd": alpha_sd},
        "coefficients": coefficients,
    }


def save_summary(summary, path):
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def save_trace(idata, path):
    idata.to_netcdf(path)


def load_trace(path):
    return az.from_netcdf(path)
