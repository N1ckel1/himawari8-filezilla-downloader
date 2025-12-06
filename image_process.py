# 改进版图像处理脚本 - 避免模糊问题
# 主要改进：
# 1. 使用PIL直接保存原始分辨率，避免matplotlib插值
# 2. 提供可选的高质量放大功能
# 3. 保持原始数据的清晰度

import os
import glob
import numpy as np
from xarray import open_dataset
from scipy.interpolate import interp1d
from PIL import Image
from scipy.ndimage import zoom

# 定义文件路径
file_path = ''
output_dir = ''

# 创建输出目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# 获取所有NC文件（递归搜索所有子目录）
file_list = glob.glob(os.path.join(file_path, '**', '*.nc'), recursive=True)
print(f"找到 {len(file_list)} 个文件待处理")

# # 定义研究区域范围 如果需要处理其他区域，请修改以下经纬度范围

lonmin = 114.06
lonmax = 116.06
latmin = 37.73
latmax = 39.73

# 配置选项
SAVE_ORIGINAL = True      # 保存原始分辨率 (约101×125像素)
SAVE_UPSCALED = False     # 保存高质量放大版本
UPSCALE_FACTOR = 4        # 放大倍数 (4倍 = 约404×500像素)
USE_HIGH_QUALITY = True   # 使用高质量插值 (bicubic)

# 定义数据拉伸函数
def stretch(data):
    data = (data - 0) / (1 - 0) * 255
    x = [0, 30, 60, 120, 190, 255]
    y = [0, 110, 160, 210, 240, 255]
    interp = interp1d(x, y, bounds_error=False, fill_value=255)
    return interp(data).astype(int)

# 循环处理每个文件
for i, filename in enumerate(file_list):
    try:
        print(f"\n正在处理第 {i+1}/{len(file_list)} 个文件: {os.path.basename(filename)}")
        
        # 使用xarray读取数据集
        dataset = open_dataset(filename, engine='h5netcdf')
        
        # 读取经纬度信息
        lon = dataset['longitude'].loc[lonmin:lonmax]
        lat = dataset['latitude'].loc[latmax:latmin]

        # 读取反射率数据(R:0.64um, G:0.51um, B:0.47um)
        B = dataset['albedo_01'].loc[latmax:latmin, lonmin:lonmax]
        G = dataset['albedo_02'].loc[latmax:latmin, lonmin:lonmax]
        R = dataset['albedo_03'].loc[latmax:latmin, lonmin:lonmax]

        # 数据拉伸
        B_stretched = stretch(B.to_numpy())
        G_stretched = stretch(G.to_numpy())
        R_stretched = stretch(R.to_numpy())

        # 合成RGB图像
        RGB_original = np.dstack((R_stretched, G_stretched, B_stretched)).astype(np.uint8)
        
        print(f"  原始数据尺寸: {RGB_original.shape[1]}×{RGB_original.shape[0]}")
        
        base_name = os.path.basename(filename).replace('.nc', '')
        
        # 保存原始分辨率版本
        if SAVE_ORIGINAL:
            output_filename = os.path.join(output_dir, f"{base_name}_original.png")
            img_pil = Image.fromarray(RGB_original)
            img_pil.save(output_filename, format='PNG', compress_level=6)
            file_size = os.path.getsize(output_filename) / 1024
            print(f"  ✓ 保存原始分辨率: {output_filename}")
            print(f"    尺寸: {RGB_original.shape[1]}×{RGB_original.shape[0]}, 文件大小: {file_size:.1f} KB")
        
        # 保存高质量放大版本
        if SAVE_UPSCALED:
            if USE_HIGH_QUALITY:
                # 使用PIL的高质量LANCZOS重采样
                img_pil = Image.fromarray(RGB_original)
                new_size = (RGB_original.shape[1] * UPSCALE_FACTOR, 
                           RGB_original.shape[0] * UPSCALE_FACTOR)
                img_upscaled = img_pil.resize(new_size, Image.Resampling.LANCZOS)
                output_filename = os.path.join(output_dir, f"{base_name}_upscaled_{UPSCALE_FACTOR}x.png")
                img_upscaled.save(output_filename, format='PNG', compress_level=6)
            else:
                # 使用scipy的双三次插值
                RGB_upscaled = zoom(RGB_original, (UPSCALE_FACTOR, UPSCALE_FACTOR, 1), order=3)
                RGB_upscaled = np.clip(RGB_upscaled, 0, 255).astype(np.uint8)
                img_upscaled = Image.fromarray(RGB_upscaled)
                output_filename = os.path.join(output_dir, f"{base_name}_upscaled_{UPSCALE_FACTOR}x.png")
                img_upscaled.save(output_filename, format='PNG', compress_level=6)
            
            file_size = os.path.getsize(output_filename) / 1024
            print(f"  ✓ 保存放大版本: {output_filename}")
            print(f"    尺寸: {img_upscaled.size[0]}×{img_upscaled.size[1]}, 文件大小: {file_size:.1f} KB")
        
        dataset.close()
        
    except Exception as e:
        print(f"  ✗ 处理文件 {filename} 时出错: {str(e)}")
        continue

print("\n" + "="*50)
print("所有文件处理完成！")
print(f"输出目录: {output_dir}")
