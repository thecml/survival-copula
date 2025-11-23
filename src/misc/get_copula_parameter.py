import pandas as pd
from pathlib import Path
import config as cfg

def get_copula_parameter(df):
    """
    Extract unique copula parameters from semisynthetic_results_org.csv
    and save them into RESULTS_DIR/copula_parameters.csv.

    Copula is determined by:
    - Seed
    - Dataset
    - Strategy
    - BestCopulaName
    - BestCopulaTheta
    """
    # Expected columns
    required = {
        "Seed", "Dataset", "Strategy",
        "BestCopulaName", "BestCopulaTheta"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Extract unique copulas
    cop_df = df[list(required)].drop_duplicates()
    
    return cop_df

if __name__ == "__main__":
    results = pd.read_csv(Path.joinpath(cfg.RESULTS_DIR, "semisynthetic_results_org.csv"))

    cop_df = get_copula_parameter(results)
    
    # Save
    out_file = cfg.RESULTS_DIR / "copula_parameters.csv"
    cop_df.to_csv(out_file, index=False)
    
    print(f"Saved copula parameters to {out_file}")
