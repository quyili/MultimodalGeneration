# _*_ coding:utf-8 _*_
import os
import numpy as np
import SimpleITK


def norm(input):
    output = (input - np.min(input)
              ) / (np.max(input) - np.min(input))
    return output


def mover(
        SRC_PATH="D:\\BaiduYunDownload\\FIRE\Images\\",
        SAVE_X_PATH="D:\\BaiduYunDownload\\SWM\\train\\X\\",
):
    try:
        os.makedirs(SAVE_X_PATH)
    except os.error:
        pass

    index_files = os.listdir(SRC_PATH)
    for file in index_files:
        arr = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(SRC_PATH + file))
        print(arr.shape)
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(arr.reshape([3, 2912, 2912])),
                             SAVE_X_PATH + file.replace(".jpg", ".tif"))


if __name__ == '__main__':
    mover(
        SRC_PATH="D:\\BaiduYunDownload\\FIRE\Images\\",
        SAVE_X_PATH="D:\\BaiduYunDownload\\SWM\\train\\X\\",
    )
