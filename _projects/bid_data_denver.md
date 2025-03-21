<div class="row" style="margin-top: 20px; margin-bottom: 20px; margin-left: 10px; margin-right: 10px;">
    <img src="/img/big_data/denver_city.jpg" alt="City of Denver" width="80%" height="80%" 
longdesc="https://www.istockphoto.com/photo/denver-colorado-skyscrapers-snowy-longs-peak-rocky-mountains-summer-gm537215344-95103705" /> 
</div>

#### Introduction
# Urban Greenspace and Asthma Prevalence

This project examines the link between vegitation cover (using average NDVI as a measurement of vegatation health) and human health. In this notebooke I will calucalte patch, edge, and fragmentation statistics about urban greenspace in Washington D.C. These statistics should be reflective of the connectivity and spread of urban greenspace, which are important for ecosystem function and access. I then use a inear model to identify statistically significant correlations between the distribution of greenspace and health data compiled by the US Center for Disease Control.

## Using 'Big Data' in the Form of Census Tracts

For this project, I split up the green space (NDVI) data by census tract, because this matches the human health data from the CDC. Because I need to know more about the structure of green space within each tract, I need higher resolution data, so I will use National Agricultural Imagery Program (NAIP) data, which is taken for the continental US every few years at 1m resolution. The main purpose of the NAIP data is, as the name suggests, agriculture. However, it’s also a great source for urban areas where lots is happening on a very small scale.

## References
City-Data. (n.d.). Income in Denver, Colorado. City-Data.com. Retrieved March 18, 2025, from https://www.city-data.com/income/income-Denver-Colorado.html



# Set Up Analysis

### Imports Libraries

```python

import pandas as pd
import rioxarray as rxr # Work with geospatial raster data
import geopandas as gpd
import pathlib
import os
import hvplot.pandas
import geoviews as gv
import cartopy.crs as ccrs
gv.extension('bokeh')  # Activates GeoViews with the Bokeh backend

# For the NDVI Data
import re # Extract metadata from file names
import zipfile # Work with zip files
from io import BytesIO # Stream binary (zip) files
import numpy as np # Unpack bit-wise Fmask
import requests # Request data over HTTP
import pystac_client
import rioxarray.merge as rxrmerge
import shapely
import xarray as xr
import glob
from cartopy import crs as ccrs
from scipy.ndimage import convolve
from sklearn.model_selection import KFold
from scipy.ndimage import label
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

warnings.simplefilter('ignore')
```

```python
# Prevent GDAL from quitting due to momentary disruptions
os.environ["GDAL_HTTP_MAX_RETRY"] = "5"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "1"
```

## STEP 2: Create a Site Map

Use the Center for Disease Control (CDC) Places dataset for human health data to compare with vegetation. CDC Places also provides
some modified census tracts, clipped to the city boundary, to go along with the health data. Start by downloading the matching geographic
data, and then select the City of Denver, Colorado.

```python
# Define info for CDC download, and only download cdc data once
if not os.path.exists(cdc_data_path):
    cdc_data_url = ('https://data.cdc.gov/download/x7zy-2xmx/application%2Fzip')
    cdc_data_gdf = gpd.read_file(cdc_data_url)
    denver_gdf = cdc_data_gdf[cdc_data_gdf.PlaceName=='Denver']
    denver_gdf.to_file(cdc_data_path, index=False)

# Load in cdc data
denver_gdf = gpd.read_file(cdc_data_path)

# Check the data - site plot 
denver_gdf.plot()

(
    denver_gdf
    .to_crs(ccrs.Mercator())
    .hvplot(
        line_color='red', fill_color=None,
        crs=ccrs.Mercator(), tiles='EsriImagery',
        frame_width=600
    )
)
```

<div class="row" style="margin-top: 20px; margin-bottom: 20px; margin-left: 10px; margin-right: 10px;">
    <img src="/img/big_data/cdc_census_tracts.png" alt="Denver CDC Census Tracts" width="70%" height="70%" /> 
</div>

# City of Denver Data Description

Data is drawn from the Center for Disease Control PLACES: Local Data for Better Health data set. The Places Dataset estimates chronic disease and other health-related measures at various geographic levels of the United States using a small area estimation methodology. Data are derived from Behavioral Risk Factors Surveillance System data, Census population data, and American Community Survey data. There are a total of 40 measures genereated in the 2024 release. For more information see the CDC Places website.

For this analysis, I will be using data from the city level - specifically the city of Denver. 

For information on how to access this data via the CDC portal, you can follow this tutorial. https://www.cdc.gov/places/tools/explore-places-data-portal.html

Reference: https://www.cdc.gov/places/methodology/index.html

You can obtain urls for the U.S. Census Tract shapefiles from [the TIGER
service](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html).
You’ll notice that these URLs use the state FIPS, which you can get by
looking it up
(e.g. [here](https://www.bls.gov/respondents/mwr/electronic-data-interchange/appendix-d-usps-state-abbreviations-and-fips-codes.htm),
or by installing and using the `us` package.

```python
# Define info for big data download (Colorado)
census_data_co_url=("https://www2.census.gov/geo/tiger/TIGER2024/PLACE/tl_2024_08_place.zip"
)

census_data_co_dir = os.path.join(data_dir, 'census_data_co')
os.makedirs(census_data_co_dir, exist_ok=True)
census_data_co_path = os.path.join(census_data_co_dir, 'tl_2024_08_place.shp') # Spatial Join 

# Only download the census tracts once
if not os.path.exists(census_data_co_path):
    census_data_co_gdf = gpd.read_file(census_data_co_url)
    census_data_co_gdf.to_file(census_data_co_path)
    den_census_gdf = census_data_co_gdf[census_data_co_gdf.NAME=='Denver']
    den_census_gdf.to_file(census_data_co_gdf, index=False)
    

# Load in census tract data
census_data_co_gdf = gpd.read_file(census_data_co_path)
```

## Step 3 - Access Asthma and Urban Greenspaces Data

```python
# Set up a path for the asthma data
cdc_path = os.path.join(data_dir, 'asthma.csv')

# Download asthma data (only once)
if not os.path.exists(cdc_path):
    asthma_url = (
        "https://data.cdc.gov/resource/cwsq-ngmh.csv"
        "?year=2022"
        "&stateabbr=CO"
        "&countyname=Denver"
        "&measureid=CASTHMA"
        "&$limit=1500"
    )
    asthma_df = (
        pd.read_csv(asthma_url)
        .rename(columns={
            'data_value': 'asthma',
            'low_confidence_limit': 'asthma_ci_low',
            'high_confidence_limit': 'asthma_ci_high',
            'locationname': 'tract'})
        [[
            'year', 'tract', 
            'asthma', 'asthma_ci_low', 'asthma_ci_high', 'data_value_unit',
            'totalpopulation', 'totalpop18plus'
        ]]
    )
    asthma_df.to_csv(asthma_path, index=False)

# Load in asthma data
asthma_df = pd.read_csv(asthma_path)

# Preview asthma data
asthma_df
```

### Join Health Data with Census Tract Boundaries

```python
# Join the census tract GeoDataFrame with the asthma prevalence DataFrame using the .merge() method.
# Note: You will need to change the data type of one of the merge columns to match, e.g. using .astype('int64')

# Change tract identifier datatype for mergingusing the .merge()method
denver_gdf.tract2010 = denver_gdf.tract2010.astype('int64')

# Merge census data with geometry
merged_gdf = (
    denver_gdf
    .merge(asthma_df, left_on='tract2010', right_on='tract', how='inner')
)

# Plot asthma data as chloropleth
(
    gv.tile_sources.EsriImagery
    * 
    gv.Polygons(
        merged_gdf.to_crs(ccrs.Mercator()),
        vdims=['asthma', 'tract2010'],
        crs=ccrs.Mercator()
    ).opts(color='asthma', colorbar=True, tools=['hover'])
).opts(width=600, height=600, xaxis=None, yaxis=None)
```

<div class="row" style="margin-top: 20px; margin-bottom: 20px; margin-left: 10px; margin-right: 10px;">
    <img src="/img/big_data/Asthma_Denver.png" alt="Asthma Prevelence in Denver, CO" width="80%" height="80%"/> 
</div>

# CDC ASTHMA PLACES DESCRIPTION AND CITATION

## Interpretation

There does appear to be a geographical component to the distribution of adults who have asthma. Whether this is a by product of another variable or not cannot be determined at this time. Other potential causes could be income disparity, areas with high pollution due to heavy industry or other reasons, or areas with insufficent access to affordable healthcare. For intance, in the map of income of Denver, there are similarities between those in lower income areas and those with asthma. The area in the Asthma Incidence map shows that the average income in that census tract is approximately 26K. The areas directly to the north has a significantly lower rate of asthma and a median income of 240k (City-Data.com, n.d.). 


There is one map that does closely match the distribution of asthma prevalence, and that is a map based on more diverse communities. Communities with the least around of diversity showed the lowest levels of asthma (City-Data.com, n.d.). 

<div class="row" style="margin-top: 20px; margin-bottom: 20px; margin-left: 10px; margin-right: 10px;">
    <img src="/img/big_data/" alt="Asthma Prevelence in Denver, CO" width="40%" height="40%"/> 
</div>

# Connect to the planetary computer catalog
```python
e84_catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)
e84_catalog.title
```

```python
# Use a loop and for each Census track search for the approrpiate data. 
# Convert geometry to lat/lon for STAC
merged_latlon_gdf = merged_gdf.to_crs(4326)

# Define a path to save NDVI stats
ndvi_stats_path = os.path.join(data_dir, 'denver-ndvi-stats.csv')

# Check for existing data - do not access duplicate tracts
downloaded_tracts = []
if os.path.exists(ndvi_stats_path):
    ndvi_stats_df = pd.read_csv(ndvi_stats_path)
    downloaded_tracts = ndvi_stats_df.tract.values
else:
    print('No census tracts downloaded so far')
    
# Loop through each census tract
scene_dfs = []
for i, tract_values in tqdm(merged_latlon_gdf.iterrows()):
    tract = tract_values.tract2010
    # Check if statistics are already downloaded for this tract
    if not (tract in downloaded_tracts):
        # Retry up to 5 times in case of a momentary disruption
        i = 0
        retry_limit = 5
        while i < retry_limit:
            # Try accessing the STAC
            try:
                # Search for tiles
                naip_search = e84_catalog.search(
                    collections=["naip"],
                    intersects=shapely.to_geojson(tract_values.geometry),
                    datetime="2021"
                )
                
                # Build dataframe with tracts and tile urls
                scene_dfs.append(pd.DataFrame(dict(
                    tract=tract,
                    date=[pd.to_datetime(scene.datetime).date() 
                          for scene in naip_search.items()],
                    rgbir_href=[scene.assets['image'].href for scene in naip_search.items()],
                )))
                
                break
            # Try again in case of an APIError
            except pystac_client.exceptions.APIError:
                print(
                    f'Could not connect with STAC server. '
                    f'Retrying tract {tract}...')
                time.sleep(2)
                i += 1
                continue
```

```python
# Concatenate the url dataframes
if scene_dfs:
    scene_df = pd.concat(scene_dfs).reset_index(drop=True)
else:
    scene_df = None

# Preview the URL DataFrame
scene_df
```

```python
# Skip this step if data are already downloaded 
if not scene_df is None:
    # Get an example tract
    tract = denver_gdf.loc[0].tract2010
    ex_scene_gdf = scene_df[scene_df.tract==tract]

    # Loop through all images for tract
    tile_das = []
    for _, href_s in ex_scene_gdf.iterrows():
        # Open vsi connection to data
        tile_da = rxr.open_rasterio(
            href_s.rgbir_href, masked=True).squeeze()
        
        # Crop data to the bounding box of the census tract
        boundary = (
            merged_gdf
            .set_index('tract2010')
            .loc[[tract]]
            .to_crs(tile_da.rio.crs)
            .geometry
        )
        crop_da = tile_da.rio.clip_box(
            *boundary.envelope.total_bounds,
            auto_expand=True)

        # Clip data to the boundary of the census tract
        clip_da = crop_da.rio.clip(boundary, all_touched=True)
            
        # Compute NDVI
        ndvi_da = (
            (clip_da.sel(band=4) - clip_da.sel(band=1)) 
            / (clip_da.sel(band=4) + clip_da.sel(band=1))
        )

        # Accumulate result
        tile_das.append(ndvi_da)

    # Merge data
    scene_da = rxrmerge.merge_arrays(tile_das)

    # Mask vegetation
    veg_mask = (scene_da>.3)

    # Calculate mean patch size
    labeled_patches, num_patches = label(veg_mask)
    
    # Count patch pixels, ignoring background at patch 0
    patch_sizes = np.bincount(labeled_patches.ravel())[1:]
    mean_patch_size = patch_sizes.mean()

    # Calculate edge density
    kernel = np.array([
        [1, 1, 1], 
        [1, -8, 1], 
        [1, 1, 1]])
    edges = convolve(veg_mask, kernel, mode='constant')
    edge_density = np.sum(edges != 0) / veg_mask.size
```

```python
# Skip this step if data are already downloaded 
if not scene_df is None:
    ndvi_dfs = []
    # Loop through the census tracts with URLs
    for tract, tract_date_gdf in tqdm(scene_df.groupby('tract')):
        # Open all images for tract
        tile_das = []
        for _, href_s in tract_date_gdf.iterrows():
            # Open vsi connection to data
            tile_da = rxr.open_rasterio(
                href_s.rgbir_href, masked=True).squeeze()
            
            # Clip data
            boundary = (
                merged_gdf
                .set_index('tract2010')
                .loc[[tract]]
                .to_crs(tile_da.rio.crs)
                .geometry
            )
            crop_da = tile_da.rio.clip_box(
                *boundary.envelope.total_bounds,
                auto_expand=True)
            clip_da = crop_da.rio.clip(boundary, all_touched=True)
                
            # Compute NDVI
            ndvi_da = (
                (clip_da.sel(band=4) - clip_da.sel(band=1)) 
                / (clip_da.sel(band=4) + clip_da.sel(band=1))
            )

            # Accumulate result
            tile_das.append(ndvi_da)

               # Merge data
        scene_da = rxrmerge.merge_arrays(tile_das)

        # Mask vegetation
        veg_mask = (scene_da>.3)

        # Calculate statistics and save data to file
        total_pixels = scene_da.notnull().sum()
        veg_pixels = veg_mask.sum()

        # Calculate mean patch size
        labeled_patches, num_patches = label(veg_mask)
        
        # Count patch pixels, ignoring background at patch 0
        patch_sizes = np.bincount(labeled_patches.ravel())[1:] 
        mean_patch_size = patch_sizes.mean()

        # Calculate edge density
        kernel = np.array([
            [1, 1, 1], 
            [1, -8, 1], 
            [1, 1, 1]])
        edges = convolve(veg_mask, kernel, mode='constant')
        edge_density = np.sum(edges != 0) / veg_mask.size
        
        # Add a row to the statistics file for this tract
        pd.DataFrame(dict(
            tract=[tract],
            total_pixels=[int(total_pixels)],
            frac_veg=[float(veg_pixels/total_pixels)],
            mean_patch_size=[mean_patch_size],
            edge_density=[edge_density]
        )).to_csv(
            ndvi_stats_path, 
            mode='a', 
            index=False, 
            header=(not os.path.exists(ndvi_stats_path))
        )

# Re-load results from file
ndvi_stats_df = pd.read_csv(ndvi_stats_path)
ndvi_stats_df
```


```python
# Merge census data with geometry
merged_ndvi_gdf = (
    merged_gdf
    .merge(
        ndvi_stats_df,
        left_on='tract2010', right_on='tract', how='inner')
)

# Plot choropleths with vegetation statistics
def plot_chloropleth(gdf, colorbar_opts=None, **opts):
    """Generate a choropleth with the given color column and colorbar label"""
    return gv.Polygons(
        gdf.to_crs(ccrs.Mercator()),
        crs=ccrs.Mercator()
    ).opts(
        xaxis=None, 
        yaxis=None, 
        colorbar=True,
        colorbar_opts=colorbar_opts,   
        xlabel="Longitude",
        ylabel="Latitude",
        **opts 
    )

# Generate plots with correct colorbar labels
(
    plot_chloropleth(
        merged_ndvi_gdf, 
        color='asthma', 
        cmap='viridis', 
        title="Asthma Rates by Census Tract", 
        colorbar_opts={'title': "Asthma Rate"}  # Fixed here
    ) 
    +
    plot_chloropleth(
        merged_ndvi_gdf, 
        color='edge_density', 
        cmap='Greens',
        title="Vegetation Density by Census Tract in Denver", 
        colorbar_opts={'title': "Edge Density"}  # Fixed here
    )
)
```
<div class="row" style="margin-top: 20px; margin-bottom: 20px; margin-left: 10px; margin-right: 10px;">
    <img src="/img/big_data/asthma_rate_denver.png" alt="City of Denver" width="80%" height="80%"/> 
</div>
<div class="row" style="margin-top: 20px; margin-bottom: 20px; margin-left: 10px; margin-right: 10px;">
    <img src="/img/big_data/veg_density_denver.png" alt="City of Denver" width="80%" height="80%"/> 
</div>

## Plot discription and Comparison

There are several coreleation between the two plots and other sources of data that show a slight link between vegitation density and asthma rates. However, additional research has show a more persuasive correlation between income and asthma and ethnicity and asthma. There also seems to be a link between how close individuals live highways and industrial areas and asthma rates.


