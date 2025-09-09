"""
Access module for the fynesse framework.

This module handles data access functionality including:
- Data loading from various sources (web, local files, databases)
- Legal compliance (intellectual property, privacy rights)
- Ethical considerations for data usage
- Error handling for access issues

Legal and ethical considerations are paramount in data access.
Ensure compliance with e.g. .GDPR, intellectual property laws, and ethical guidelines.

Best Practice on Implementation
===============================

1. BASIC ERROR HANDLING:
   - Use try/except blocks to catch common errors
   - Provide helpful error messages for debugging
   - Log important events for troubleshooting

2. WHERE TO ADD ERROR HANDLING:
   - File not found errors when loading data
   - Network errors when downloading from web
   - Permission errors when accessing files
   - Data format errors when parsing files

3. SIMPLE LOGGING:
   - Use print() statements for basic logging
   - Log when operations start and complete
   - Log errors with context information
   - Log data summary information

4. EXAMPLE PATTERNS:
   
   Basic error handling:
   try:
       df = pd.read_csv('data.csv')
   except FileNotFoundError:
       print("Error: Could not find data.csv file")
       return None
   
   With logging:
   print("Loading data from data.csv...")
   try:
       df = pd.read_csv('data.csv')
       print(f"Successfully loaded {len(df)} rows of data")
       return df
   except FileNotFoundError:
       print("Error: Could not find data.csv file")
       return None
"""

from typing import Any, Union
import pandas as pd
import logging

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def data() -> Union[pd.DataFrame, None]:
    """
    Read the data from the web or local file, returning structured format such as a data frame.

    IMPLEMENTATION GUIDE
    ====================

    1. REPLACE THIS FUNCTION WITH YOUR ACTUAL DATA LOADING CODE:
       - Load data from your specific sources
       - Handle common errors (file not found, network issues)
       - Validate that data loaded correctly
       - Return the data in a useful format

    2. ADD ERROR HANDLING:
       - Use try/except blocks for file operations
       - Check if data is empty or corrupted
       - Provide helpful error messages

    3. ADD BASIC LOGGING:
       - Log when you start loading data
       - Log success with data summary
       - Log errors with context

    4. EXAMPLE IMPLEMENTATION:
       try:
           print("Loading data from data.csv...")
           df = pd.read_csv('data.csv')
           print(f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
           return df
       except FileNotFoundError:
           print("Error: data.csv file not found")
           return None
       except Exception as e:
           print(f"Error loading data: {e}")
           return None

    Returns:
        DataFrame or other structured data format
    """
    logger.info("Starting data access operation")

    try:
        # IMPLEMENTATION: Replace this with your actual data loading code
        # Example: Load data from a CSV file
        logger.info("Loading data from data.csv")
        df = pd.read_csv("data.csv")

        # Basic validation
        if df.empty:
            logger.warning("Loaded data is empty")
            return None

        logger.info(
            f"Successfully loaded data: {len(df)} rows, {len(df.columns)} columns"
        )
        return df

    except FileNotFoundError:
        logger.error("Data file not found: data.csv")
        print("Error: Could not find data.csv file. Please check the file path.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        print(f"Error loading data: {e}")
        return None


import osmnx as ox
import matplotlib.pyplot as plt

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


def get_osm_datapoints(latitude, longitude, box_size_km=2, poi_tags=None):

    box_width = box_size_km / 111
    box_height = box_size_km / 111
    north = latitude + box_height
    south = latitude - box_height
    west = longitude - box_width
    east = longitude + box_width

    try:
        bbox = (west, south, east, north)
        pois = ox.features_from_bbox(bbox, tags=tags)
        return pois.head()

    except Exception as e:
        print(f"[Warning] OSM query failed: {e}")
        return None


def plot_city_map(place_name, latitude=None, longitude=None, coords_df=None, box_size_km=2, poi_tags=None):

    if coords_df is not None: 
        lat_min, lat_max = coords_df["Latitude"].min(), coords_df["Latitude"].max()
        lon_min, lon_max = coords_df["Longitude"].min(), coords_df["Longitude"].max()

        lat_margin = box_size_km / 111
        lon_margin = box_size_km / 111
        north = lat_max + lat_margin/10
        south = lat_min - lat_margin/10
        west = lon_min - lon_margin/10
        east = lon_max + lon_margin/10

    elif latitude is not None and longitude is not None: 
        box_width = box_size_km / 111
        box_height = box_size_km / 111
        north = latitude + box_height/2
        south = latitude - box_height/2
        west = longitude - box_width/2
        east = longitude + box_width/2

    else:
        raise ValueError("No Cordinates available")

    bbox = (west, south, east, north)

    graph = ox.graph_from_bbox(bbox)
    area = ox.geocode_to_gdf(place_name)
    nodes, edges = ox.graph_to_gdfs(graph)
    buildings = ox.features_from_bbox(bbox, tags={"building": True})
    pois = ox.features_from_bbox(bbox, poi_tags)

    try:
        fig, ax = plt.subplots(figsize=(6,6))
        area.plot(ax=ax, color="tan", alpha=0.5)
        buildings.plot(ax=ax, facecolor="gray", edgecolor="gray")
        edges.plot(ax=ax, linewidth=1, edgecolor="black", alpha=0.3)
        nodes.plot(ax=ax, color="black", markersize=1, alpha=0.3)
        pois.plot(ax=ax, color="green", markersize=5, alpha=1)

        if coords_df is not None:
            ax.scatter(
                coords_df["Longitude"],
                coords_df["Latitude"],
                c="red", s=10, marker="o", label="Cameras"
            )
        else:
            ax.scatter(
                longitude, latitude,
                c="red", s=30, marker="x", label="Point"
            )

        ax.set_xlim(west, east)
        ax.set_ylim(south, north)
        ax.set_title(f"{place_name}", fontsize=14)
        ax.legend()
        plt.show()

    except Exception as e:
        print(f"[Warning] Could not plot map: {e}")
