# _*_ coding:utf-8 _*_
import os
import numpy as np
import SimpleITK


def mover(
        INDEX_PATH="D:/BaiduYunDownload/TC19/labels",
        SRC_PATH="D:/BaiduYunDownload/TC19/images/",
        SAVE_PATH="D:/BaiduYunDownload/TC19/new_images/",
):
    try:
        os.makedirs(SAVE_PATH)
    except os.error:
        pass

    index_files =  os.listdir(INDEX_PATH)
    for file in index_files:
        input_x = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(SRC_PATH + file.replace(".txt", ".png")))
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(input_x), SAVE_PATH + file.replace(".txt", ".mha"))

if __name__ == '__main__':
    mover(
        INDEX_PATH="D:/BaiduYunDownload/TC19/labels",
        SRC_PATH="D:/BaiduYunDownload/TC19/images/",
        SAVE_PATH="D:/BaiduYunDownload/TC19/new_images/",
    )
