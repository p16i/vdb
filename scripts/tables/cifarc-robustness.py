"""
Usage:
./scripts/tables/cifarc-robustness.py <experiment-directory>

Options:
 -h --help      Show this screen.
"""

import os
import glob

import pandas as pd
import numpy as np

import yaml

from docopt import docopt

CIFAR10C_CATEGORIES = """
gaussian_noise
shot_noise
impulse_noise
defocus_blur
glass_blur
motion_blur
zoom_blur
snow
frost
fog
brightness
contrast
elastic_transform
pixelate
jpeg_compression
""".strip().split("\n")

RESULT_DIR = "./datasets/analysis-results"

def read_cifar10c_analysis(path):
    with open(f"{path}/cifar10c-analysis.yml", "r") as fh:
        analysis = yaml.safe_load(fh)
    name = path.split("/")[-1]
    with open(f"{path}/summary.yml", "r") as fh:
        summary = yaml.safe_load(fh)
        
    arch = summary["model"].split("/")[0]
    
    key = "-".join(
        [arch, summary["strategy"],
        summary["class_loss"],
        summary["cov_type"],
        "M" + str(summary["M"])]
    )

    rows = []

    for cat in analysis["categories"]:
        for r in analysis["categories"][cat]:
            
            acc = r["metrics"][1]            
            assert 12 == acc["L"]
            
            rows.append(dict(
                name=name,
                category=cat,
                severity=r["severity"],
                accuracy=acc["accuracy"],
                key=key,
                model=arch,
                strategy=summary["strategy"],
                class_loss=summary["class_loss"],
                cov_type=summary["cov_type"],
                M=summary["M"],
                clean_accuracy=summary["metrics"]["test"]["accuracy_L12"],
                beta=summary["beta"],
            ))
            
    return rows


def read_cifar10c_experiment_metrics(directory):
    models = glob.glob(f"{directory}/summary.yml")
    
    models = list(map(lambda x: os.path.dirname(x), models))

    print(f"Found {len(models)} models")
    
    rows = []
    
    for m in models:
        
        rows.extend(read_cifar10c_analysis(m))
        
    df = pd.DataFrame(rows)

    df["error"] = 100*(1-df["accuracy"])
    df["error_clean"] = 100*(1-df["clean_accuracy"])

    return df
    

def compute_corruption_error(df, ref_key, strategies=["oneshot", "algo2/k:10"]):
    group_key = ["key", "category"]

    df = df[
        df.strategy.isin(strategies) &
        df.category.isin(CIFAR10C_CATEGORIES)
    ].copy()
    
    df = df.groupby(["name", "category", "key"])\
        .agg(
            sum_error=("error", "sum"),
            clean_error=("error_clean", "mean") # this error_clean are the same for each model; hence getting mean is equal to the value
        )\
        .reset_index()
  
    # calculate relative_error for each run
    df["rel_sum_error"] = df["sum_error"] - df["clean_error"]

    # sum reference metrics over all runs 
    df_ref_setting = df[df.key == ref_key].groupby(group_key)\
        .agg(ref_sum_error=("sum_error", "sum"), ref_rel_sum_error=("rel_sum_error", "sum")) \
        .reset_index()

    # sun metrics over all runs
    df_other_models = df.groupby(group_key) \
        .agg(sum_error=("sum_error", "sum"), rel_sum_error=("rel_sum_error", "sum")) \
        .reset_index()

    df_result = df_other_models.merge(df_ref_setting[["category", "ref_sum_error", "ref_rel_sum_error"]], on="category")

    df_result["ce"] = 100* df_result["sum_error"] / df_result["ref_sum_error"]
    df_result["rel_ce"] = 100* df_result["rel_sum_error"] / df_result["ref_rel_sum_error"]

    return df_result.groupby(["key"])\
        .agg(
            mCe=("ce", "mean"),
            rel_mCe=("rel_ce", "mean"), 
        )

def main(directory):

    print(f"Getting CIFAR10-C Results from {directory}")

    df = read_cifar10c_experiment_metrics(directory)


    setting_keys = ["class_loss", "strategy", "cov_type", "M"] 
    runs_per_model = df[ (df.severity == 1) & (df.category == "zoom_blur") ]\
        .groupby(setting_keys)["model"].count() \
        .values

    assert np.all(runs_per_model[0] == runs_per_model), "All settings have the same number of runs"

    print(f"We have {runs_per_model[0]} runs for each setting ({','.join(setting_keys)})")

    beta_values = df.beta.unique()
    assert len(beta_values) == 1, "there are more than one β values"


    result_dir  = f"{RESULT_DIR}/cifar10-analysis-beta{int(np.log10(beta_values[0]))}"

    os.makedirs(result_dir, exist_ok=True)
    print(f"Saving analysis results to {result_dir}")

    with pd.option_context('display.float_format', '{:0.2f}'.format):
        df_mce = compute_corruption_error(df, "resnet20-oneshot-vdb-diag-M1")

        print(df_mce)

        with open(f"{result_dir}/latex-table.txt", "w") as fh:
            fh.write(df_mce.to_latex(escape=False))
        
        df_sum_level_errors = df.groupby(["name", "key", "category", "error_clean"]).agg(
                sum_level_error=("error", "sum"),
                run_error_clean=("error_clean", "mean") # 
            ) \
            .reset_index() \

        df_details = df_sum_level_errors\
            .groupby(["key", "category"]) \
            .agg(cat_error=("sum_level_error", "mean")) \
            .reset_index() \
            .pivot(index="key", columns="category") \
            .droplevel(0, axis=1) 

        df_clean_accuracy = df[(df.severity == 1) & (df.category == "zoom_blur")] \
            .groupby("key") \
            .agg(
                error_clean_mean=("error_clean", "mean"),
                error_clean_std=("error_clean", "std")
            ).reset_index()

        df_merged = df_details.merge(df_clean_accuracy, on="key")[[
            "key", "error_clean_mean", "error_clean_std", *CIFAR10C_CATEGORIES
        ]]

        df_merged.to_csv(f"{result_dir}/per-category-errors.csv", index=False)

        df_rel_mean_error = df_sum_level_errors
        df_rel_mean_error["rel_error"] = df_sum_level_errors["sum_level_error"] - df_sum_level_errors.run_error_clean

        xx = df_rel_mean_error\
            .groupby(["key", "category"])\
            .agg(mean_rel_error=("rel_error", "mean"))\
            .reset_index()\
            .pivot(index="key", columns="category") \
            .droplevel(0, axis=1)[[*CIFAR10C_CATEGORIES]] \
            .to_csv(f"{result_dir}/per-category-relative-errors.csv")

        print(xx)



if __name__ == "__main__":
    arguments = docopt(__doc__, version="Table : CIFAR-C Robustness")

    experiment_directory = arguments["<experiment-directory>"]

    main(experiment_directory)
