# _*_ coding:utf-8 _*_
import os
import numpy as np
import SimpleITK

def norm(input):
    output = (input - np.min(input)
              ) / (np.max(input) - np.min(input))
    return output

def mover(
        SRC_PATH="D:\\BaiduYunDownload\\finding-lungs-in-ct-data\\2d_images\\",
        SAVE_X_PATH="D:\\BaiduYunDownload\\finding-lungs-in-ct-data\\X\\",
):
    try:
        os.makedirs(SAVE_X_PATH)
    except os.error:
        pass

    index_files =  os.listdir(SRC_PATH)
    for file in index_files:
        arr = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(SRC_PATH+ file)).astype( 'float32')
        arr = norm(arr)
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(arr.astype("float32")), SAVE_X_PATH + file.replace(".tif", ".tiff"))

if __name__ == '__main__':
    mover(
       SRC_PATH="D:\\BaiduYunDownload\\finding-lungs-in-ct-data\\2d_images\\",
        SAVE_X_PATH="D:\\BaiduYunDownload\\finding-lungs-in-ct-data\\X\\",
    )
