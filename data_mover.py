# _*_ coding:gbk _*_
import os
import numpy as np
import SimpleITK


def mover(
        SRC_F1="D:\\BaiduYunDownload\\TC19\\new_images_F1\\",
        SRC_M1="D:\\BaiduYunDownload\\TC19\\new_images_M1\\",
        SRC_F2="D:\\BaiduYunDownload\\TC19\\new_images_F2\\",
        SRC_M2="D:\\BaiduYunDownload\\TC19\\new_images_M2\\",
        SRC_F3="D:\\BaiduYunDownload\\TC19\\new_images_F3\\",
        SRC_M3="D:\\BaiduYunDownload\\TC19\\new_images_M3\\",
        SRC_F="D:\\BaiduYunDownload\\TC19\\new_images_F\\",
        SRC_M="D:\\BaiduYunDownload\\TC19\\new_images_M\\",
        SRC_LABEL="D:\\BaiduYunDownload\\TC19\\new_labels\\",

        SAVE_F1="D:\\BaiduYunDownload\\TC19\\selected\\new_images_F1\\",
        SAVE_M1="D:\\BaiduYunDownload\\TC19\\selected\\new_images_M1\\",
        SAVE_F2="D:\\BaiduYunDownload\\TC19\\selected\\new_images_F2\\",
        SAVE_M2="D:\\BaiduYunDownload\\TC19\\selected\\new_images_M2\\",
        SAVE_F3="D:\\BaiduYunDownload\\TC19\\selected\\new_images_F3\\",
        SAVE_M3="D:\\BaiduYunDownload\\TC19\\selected\\new_images_M3\\",
        SAVE_F="D:\\BaiduYunDownload\\TC19\\selected\\new_images_F\\",
        SAVE_M="D:\\BaiduYunDownload\\TC19\\selected\\new_images_M\\",

        SAVE_LABEL="D:\\BaiduYunDownload\\TC19\\selected\\new_labels\\",

        limit=0.25
):
    try:
        os.makedirs(SAVE_F1)
        os.makedirs(SAVE_M1)
        os.makedirs(SAVE_F2)
        os.makedirs(SAVE_M2)
        os.makedirs(SAVE_F3)
        os.makedirs(SAVE_M3)
        os.makedirs(SAVE_F)
        os.makedirs(SAVE_M)
        os.makedirs(SAVE_LABEL)
    except os.error:
        pass

    index_files = os.listdir(SRC_F2)
    # index_files=["323490_043.tiff","322897_054.tiff","322897_021.tiff",
    #              "366938_000.tiff","367880_023.tiff","367880_056.tiff","369542_113.tiff",
    #              "371756_024.tiff","387453_044.tiff"]
    for file in index_files:
        arr_mean = np.mean(SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(SRC_F2 + file))[128:384, 128:384])
        mha_flie = file
        mha_flie = mha_flie.replace(".tiff", ".mha")
        txt_file = file
        txt_file = txt_file.replace(".tiff", ".txt")

        if arr_mean > limit:
            #     print(file,'MOVE')
            # else:
            #     print(file, 'NO MOVE')
            os.system('move ' + SRC_F1 + file + ' ' + SAVE_F1 + file)
            os.system('move ' + SRC_M1 + file + ' ' + SAVE_M1 + file)
            os.system('move ' + SRC_F2 + file + ' ' + SAVE_F2 + file)
            os.system('move ' + SRC_M2 + file + ' ' + SAVE_M2 + file)
            os.system('move ' + SRC_F3 + file + ' ' + SAVE_F3 + file)
            os.system('move ' + SRC_M3 + file + ' ' + SAVE_M3 + file)

            os.system('move ' + SRC_F + mha_flie + ' ' + SAVE_F + mha_flie)
            os.system('move ' + SRC_M + mha_flie + ' ' + SAVE_M + mha_flie)

            os.system('move ' + SRC_LABEL + txt_file + ' ' + SAVE_LABEL + txt_file)


if __name__ == '__main__':
    mover(
        SRC_F1="D:\\BaiduYunDownload\\TC19\\new_images_F1\\",
        SRC_M1="D:\\BaiduYunDownload\\TC19\\new_images_M1\\",
        SRC_F2="D:\\BaiduYunDownload\\TC19\\new_images_F2\\",
        SRC_M2="D:\\BaiduYunDownload\\TC19\\new_images_M2\\",
        SRC_F3="D:\\BaiduYunDownload\\TC19\\new_images_F3\\",
        SRC_M3="D:\\BaiduYunDownload\\TC19\\new_images_M3\\",
        SRC_F="D:\\BaiduYunDownload\\TC19\\new_images_F\\",
        SRC_M="D:\\BaiduYunDownload\\TC19\\new_images_M\\",
        SRC_LABEL="D:\\BaiduYunDownload\\TC19\\new_labels\\",

        SAVE_F1="D:\\BaiduYunDownload\\TC19\\selected\\new_images_F1\\",
        SAVE_M1="D:\\BaiduYunDownload\\TC19\\selected\\new_images_M1\\",
        SAVE_F2="D:\\BaiduYunDownload\\TC19\\selected\\new_images_F2\\",
        SAVE_M2="D:\\BaiduYunDownload\\TC19\\selected\\new_images_M2\\",
        SAVE_F3="D:\\BaiduYunDownload\\TC19\\selected\\new_images_F3\\",
        SAVE_M3="D:\\BaiduYunDownload\\TC19\\selected\\new_images_M3\\",
        SAVE_F="D:\\BaiduYunDownload\\TC19\\selected\\new_images_F\\",
        SAVE_M="D:\\BaiduYunDownload\\TC19\\selected\\new_images_M\\",

        SAVE_LABEL="D:\\BaiduYunDownload\\TC19\\selected\\new_labels\\",

        limit=0.28
    )
