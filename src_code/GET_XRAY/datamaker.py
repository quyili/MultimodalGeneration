# _*_ coding:utf-8 _*_
import os
import numpy as np
import SimpleITK


def mover(
        SRC_PATH="D:\\BaiduYunDownload\\chest_xray\\train\\PNEUMONIA\\",
        SAVE_X_PATH="D:\\BaiduYunDownload\\chest_xray\\train\\X\\",
        SAVE_L_PATH="D:\\BaiduYunDownload\\chest_xray\\train/labels\\",
):
    try:
        os.makedirs(SAVE_X_PATH)
        os.makedirs(SAVE_L_PATH)
    except os.error:
        pass

    index_files =  os.listdir(SRC_PATH)
    for file in index_files:
        L=np.zeros([512,512])
        if "bacteria" in file:
            L = L + 1
        elif "virus" in file:
            L = L + 2
        cmd='copy ' + SRC_PATH + file + ' ' + SAVE_X_PATH + file
        print(cmd)
        os.system(cmd)
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(L.astype("float32")), SAVE_L_PATH + file.replace(".jpeg", ".tiff"))

if __name__ == '__main__':
    mover(
        SRC_PATH="D:\\BaiduYunDownload\\chest_xray\\test\\PNEUMONIA\\",
        SAVE_X_PATH="D:\\BaiduYunDownload\\chest_xray\\test\\X\\",
        SAVE_L_PATH="D:\\BaiduYunDownload\\chest_xray\\test/labels\\",
    )
