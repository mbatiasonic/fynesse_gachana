from typing import Any, Union
import pandas as pd
import logging

from .config import *
from . import access

# Set up logging
logger = logging.getLogger(__name__)

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded.
How are missing values encoded, how are outliers encoded? What do columns represent,
makes rure they are correctly labeled. How is the data indexed. Crete visualisation
routines to assess the data (e.g. in bokeh). Ensure that date formats are correct
and correctly timezoned."""


def data() -> Union[pd.DataFrame, Any]:
    """
    Load the data from access and ensure missing values are correctly encoded as well as
    indices correct, column names informative, date and times correctly formatted.
    Return a structured data structure such as a data frame.

    IMPLEMENTATION GUIDE FOR STUDENTS:
    ==================================

    1. REPLACE THIS FUNCTION WITH YOUR DATA ASSESSMENT CODE:
       - Load data using the access module
       - Check for missing values and handle them appropriately
       - Validate data types and formats
       - Clean and prepare data for analysis

    2. ADD ERROR HANDLING:
       - Handle cases where access.data() returns None
       - Check for data quality issues
       - Validate data structure and content

    3. ADD BASIC LOGGING:
       - Log data quality issues found
       - Log cleaning operations performed
       - Log final data summary

    4. EXAMPLE IMPLEMENTATION:
       df = access.data()
       if df is None:
           print("Error: No data available from access module")
           return None

       print(f"Assessing data quality for {len(df)} rows...")
       # Your data assessment code here
       return df
    """
    logger.info("Starting data assessment")

    # Load data from access module
    df = access.data()

    # Check if data was loaded successfully
    if df is None:
        logger.error("No data available from access module")
        print("Error: Could not load data from access module")
        return None

    logger.info(f"Assessing data quality for {len(df)} rows, {len(df.columns)} columns")

    try:
        # STUDENT IMPLEMENTATION: Add your data assessment code here

        # Example: Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Found missing values: {missing_counts.to_dict()}")
            print(f"Missing values found: {missing_counts.sum()} total")

        # Example: Check data types
        logger.info(f"Data types: {df.dtypes.to_dict()}")

        # Example: Basic data cleaning (students should customize this)
        # Remove completely empty rows
        df_cleaned = df.dropna(how="all")
        if len(df_cleaned) < len(df):
            logger.info(f"Removed {len(df) - len(df_cleaned)} completely empty rows")

        logger.info(f"Data assessment completed. Final shape: {df_cleaned.shape}")
        return df_cleaned

    except Exception as e:
        logger.error(f"Error during data assessment: {e}")
        print(f"Error assessing data: {e}")
        return None


def query(data: Union[pd.DataFrame, Any]) -> str:
    """Request user input for some aspect of the data."""
    raise NotImplementedError


def view(data: Union[pd.DataFrame, Any]) -> None:
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError


def labelled(data: Union[pd.DataFrame, Any]) -> Union[pd.DataFrame, Any]:
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError


import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from fynesse.access import get_osm_datapoints

features = [
    ("building", None),
    ("amenity", None),
    ("amenity", "school"),
    ("amenity", "hospital"),
    ("amenity", "restaurant"),
    ("amenity", "cafe"),
    ("shop", None),
    ("tourism", None),
    ("tourism", "hotel"),
    ("tourism", "museum"),
    ("leisure", None),
    ("leisure", "park"),
    ("historic", None),
    ("amenity", "place_of_worship"),
]


tags = {k: True for k, _ in features} if features else {}


def get_osm_features(latitude, longitude, box_size_km=2, tags=None):
    """
    Access raw OSM features.
    """
    return get_osm_datapoints(latitude, longitude, box_size_km, tags)


def get_feature_vector(latitude, longitude, box_size_km=2, features=None):
    """
    Quantify geographic features into a feature vector.
    """

    pois = get_osm_datapoints(latitude, longitude, box_size_km, tags)

    # Initialize with zeros
    all_features = [f"{k}:{v}" if v else k for k, v in features]
    feature_vec = {feat: 0 for feat in all_features}

    if pois is None or pois.empty:
        return feature_vec

    pois_df = pois.reset_index()

    for key, value in features:
        col_name = f"{key}:{value}" if value else key
        if key in pois_df.columns:
            if value:
                feature_vec[col_name] = (
                    pois_df[key].astype(str).str.lower().eq(str(value).lower()).sum()
                )
            else:
                feature_vec[col_name] = pois_df[key].notna().sum()

    return feature_vec


def build_feature_dataframe(city_dicts, features, box_size_km=1):
    results = {}
    for country, cities in city_dicts:
        for city, coords in cities.items():
            vec = get_feature_vector(
                coords["latitude"],
                coords["longitude"],
                box_size_km=box_size_km,
                features=features,
            )
            vec["country"] = country
            results[city] = vec
    return pd.DataFrame(results).T


def visualize_feature_space(X, y, method="PCA"):
    if method == "PCA":
        reducer = PCA(n_components=2)
    elif method == "tSNE":
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'PCA' or 'tSNE'")

    X_reduced = reducer.fit_transform(X)
    y_codes = pd.Series(y).astype("category").cat.codes

    plt.figure(figsize=(8, 6))
    for country, color in [("Kenya", "green"), ("England", "blue")]:
        mask = (y == country)
        plt.scatter(X_proj[mask, 0], X_proj[mask, 1],
                    label=country, color=color, s=100, alpha=0.7)

    for i, city in enumerate(df.index):
        plt.text(X_proj[i,0]+0.02, X_proj[i,1], city, fontsize=4)
    # scatter = plt.scatter(
    #     X_reduced[:, 0], X_reduced[:, 1], c=y_codes, cmap="tab10", alpha=0.7
    # )
    # Use proper legend with original labels
    legend_labels = pd.Series(y).astype("category").cat.categories
    plt.legend(*scatter.legend_elements(), title="Class", labels=legend_labels)
    plt.title(f"Feature Space Visualization ({method})")
    plt.show()
