import pandas as pd
import pyproj
import os
from osgeo import gdal, osr

def geo_to_xy(image_path, or_csv_path, save_csv_path):
    # Disable celestial body check for coordinate system
    os.environ['PROJ_IGNORE_CELESTIAL_BODY'] = 'YES'
    # Define input (lat/long) and output (Mars Equidistant) coordinate systems
    input_proj = pyproj.Proj(proj='latlong', datum='WGS84')
    output_proj = pyproj.Proj(proj='eqc', lon_0=0, lat_ts=0, a=3396190.0)

    # Open geospatial image file
    ds = gdal.Open(image_path)
    if ds is None:
        print("Failed to open image file")
        exit()

    # Extract projection from image and update output projection
    projection = ds.GetProjection()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)
    output_proj = pyproj.Proj(srs.ExportToProj4())

    data = pd.read_csv(or_csv_path)
    # Convert geographic coordinates to planar coordinates
    lon = data['long'].values
    lat = data['lat'].values
    x, y = pyproj.transform(input_proj, output_proj, lon, lat)

    # Update dataframe with converted coordinates
    data['long'] = x
    data['lat'] = y
    data.to_csv(save_csv_path, index=False)

if __name__ == "__main__":
    # Example file paths
    image_path = '/home/xgq/Desktop/HF/yunshi/data/train_tiles_for_seg/HMC_11E10_dt5.tif'
    or_csv_path = '/home/xgq/Desktop/HF/yunshi/data/catalog/hrsc_open_area.csv'
    save_csv_path = '/home/xgq/Desktop/HF/yunshi/data/catalog/hrsc_open_area_xy.csv'

    # Execute coordinate conversion
    geo_to_xy(image_path, or_csv_path, save_csv_path)
