import pandas as pd
import pyproj
import os
from osgeo import gdal, osr

def geo_to_xy(image_path, or_csv_path, save_csv_path):

    # 禁用坐标系统的天体检查
    os.environ['PROJ_IGNORE_CELESTIAL_BODY'] = 'YES'

    # 定义输入坐标系统（经纬度）和输出坐标系统（EQUIDISTANT MARS）
    input_proj = pyproj.Proj(proj='latlong', datum='WGS84')
    output_proj = pyproj.Proj(proj='eqc', lon_0=0, lat_ts=0, a=3396190.0)

    #image_path = './home/xgq/Desktop/HF/生成标注的文件/mars_data/test_set/dom/HMC_13E20_co5.tif'  # 替换为你的影像文件路径
    ds = gdal.Open(image_path)
    if ds is None:
        print("无法打开影像文件")
        exit()

    projection = ds.GetProjection()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)
    output_proj = pyproj.Proj(srs.ExportToProj4())

    # 读取包含经纬度的表格
    data = pd.read_csv(or_csv_path)  # 替换为你的表格文件名

    # 转换经纬度到EQUIDISTANT MARS坐标
    lon = data['long'].values  # 替换为你的经度列的列名
    lat = data['lat'].values  # 替换为你的纬度列的列名
    x, y = pyproj.transform(input_proj, output_proj, lon, lat)

    # 将转换后的坐标添加到表格
    data['long'] = x
    data['lat'] = y

    # 保存包含平面坐标的表格
    data.to_csv(save_csv_path, index=False)  # 替换为你想要保存的文件名


if __name__ == "__main__":
    image_path = '/home/xgq/Desktop/HF/yunshi/data/train_tiles_for_seg/HMC_11E10_dt5.tif'
    or_csv_path = '/home/xgq/Desktop/HF/yunshi/data/catalog/hrsc_open_area.csv'
    save_csv_path = '/home/xgq/Desktop/HF/yunshi/data/catalog/hrsc_open_area_xy.csv'

    geo_to_xy(image_path, or_csv_path, save_csv_path)

