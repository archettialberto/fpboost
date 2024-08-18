"""Module to load datasets from the package."""

import os
from functools import cache

import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), "datasets")  # Path to the datasets folder


@cache
def get_available_datasets() -> list[str]:
    """Return the names of the datasets available in the package.

    Returns:
        List of dataset names.
    """
    available_csv_files = [file_name for file_name in os.listdir(CSV_PATH)]
    names = [file_name.replace(".csv", "") for file_name in available_csv_files]
    return sorted(names)


def load_dataframe(dataset: str) -> pd.DataFrame:
    """Return the Pandas dataframe of a given dataset. No preprocessing is applied.

    Args:
        dataset: Name of the dataset to load.

    Returns:
        DataFrame with the dataset.
    """
    assert (
        dataset.lower() in get_available_datasets()
    ), f"Dataset {dataset} not found. Available datasets: {get_available_datasets()}"
    df = pd.read_csv(os.path.join(CSV_PATH, f"{dataset.lower()}.csv"))
    return df
