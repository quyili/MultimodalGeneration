# _*_ coding:utf-8 _*_
import os
import numpy as np
import SimpleITK
import cv2
import scipy.signal as signal
import os
from skimage import transform
import matplotlib.pyplot as plt

def mynorm(input):
    output = (input - np.min(input)
              ) / (np.max(input) - np.min(input))
    return output.astype("float32")


def binary(x, beta=0.1):
    return np.asarray(x > beta).astype("float32")


def binary_run(
        SRC_PATH="E:/project/MultimodalGeneration/src_code/paper-LaTeX/figures/SWM_SkrGAN_F.png",
        SAVE_PATH="E:/project/MultimodalGeneration/src_code/paper-LaTeX/figures/new_SWM_SkrGAN_F.tiff",
):
    input_x = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(SRC_PATH))
    input_x = (input_x[:, :, 0] + input_x[:, :, 1] + input_x[:, :, 2]) / 3.0
    print(input_x.shape)
    input_x = mynorm(input_x)
    input_x = binary(input_x)
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(1.0 - input_x), SAVE_PATH)

def get_mask_from_s(imgfile):
    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    gray = cv2.GaussianBlur(img, (3, 3), 0)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    c_list = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = c_list[-2], c_list[-1]
    cv2.drawContours(img, contours, -1, (255, 255, 255), thickness=-1)
    return np.asarray(img, dtype="float32")

def binary_png_run(
        SRC_PATH="E:/project/MultimodalGeneration/src_code/paper-LaTeX/figures/SWM_SkrGAN_F.png",
        SAVE_PATH="E:/project/MultimodalGeneration/src_code/paper-LaTeX/figures/new_SWM_SkrGAN_F.tiff",
):
    input_x = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(SRC_PATH))
    input_x = (input_x[:, :, 0] + input_x[:, :, 1] + input_x[:, :, 2]) / 3.0
    print(input_x.shape)
    input_x = mynorm(input_x)
    input_x = binary(input_x,beta=0.4)
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(input_x), "quyili.tiff")
    output=np.zeros([input_x.shape[0],input_x.shape[1],4])
    output[:, :, 0] = 1.0 -input_x
    output[:, :, 1] = 1.0 -input_x
    output[:, :, 2] = 1.0 -input_x
    output[:, :, 3] = 1.0 -input_x
    cv2.imwrite("quyili1.jpg", output[:,:,0:3] * 255)
    img = cv2.imread("quyili1.jpg", cv2.IMREAD_GRAYSCALE)
    gray = cv2.GaussianBlur(img, (3, 3), 0)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    c_list = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = c_list[-2], c_list[-1]
    cv2.drawContours(img, contours, -1, (255, 255, 255), thickness=-1)
    cv2.imwrite(SAVE_PATH,output*255)


def run(
        SRC_PATH="E:/project/MultimodalGeneration/src_code/paper-LaTeX/bib/SkrGAN/CT/PGGAN.jpg",
        SAVE_PATH="E:/project/MultimodalGeneration/src_code/paper-LaTeX/bib/SkrGAN/CT/PGGAN_.tiff",
):
    input_x = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(SRC_PATH))
    input_x = (input_x[:, :, 0] + input_x[:, :, 1] + input_x[:, :, 2]) / 3.0
    print(input_x.shape)
    input_x = mynorm(input_x)
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(input_x), SAVE_PATH)


def image_fesion(
        SRC_PATH1="E:/project/MultimodalGeneration/src_code/paper-LaTeX/samples/fjjct/x_1_2.tiff",
        SRC_PATH2="E:/project/MultimodalGeneration/src_code/paper-LaTeX/samples/fjjct/x_g_1_1_.tiff",
        SAVE_PATH="E:/project/MultimodalGeneration/src_code/paper-LaTeX/samples/fjjct/x_g_1_1_.tiff",
):
    input_x = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(SRC_PATH1))
    input_y = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(SRC_PATH2))
    output = input_x * 0.35 + input_y * 0.65
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(output), SAVE_PATH)


def image_fesion4(
        SRC_PATH1="E:/project/MultimodalGeneration/src_code/samples/fx_.tiff",
        SRC_PATH2="E:/project/MultimodalGeneration/src_code/samples/fy_.tiff",
        SRC_PATH3="E:/project/MultimodalGeneration/src_code/samples/fz_.tiff",
        SRC_PATH4="E:/project/MultimodalGeneration/src_code/samples/fw_.tiff",
        SAVE_PATH="E:/project/MultimodalGeneration/src_code/samples/fxyzw_.tiff",
):
    input_1 = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(SRC_PATH1))
    input_2 = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(SRC_PATH2))
    input_3 = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(SRC_PATH3))
    input_4 = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(SRC_PATH4))
    output = input_1 * 0.25 + input_2 * 0.25 + input_3 * 0.25 + input_4 * 0.25
    output = binary(output, beta=0.0)
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(output), SAVE_PATH)


def noise_fesion(
        SRC_PATH1="E:/project/MultimodalGeneration/src_code/paper-LaTeX/samples/F_1_3.tiff",
        SRC_PATH2="E:/project/MultimodalGeneration/src_code/paper-LaTeX/samples/M_1_3.tiff",
        SAVE_PATH="E:/project/MultimodalGeneration/src_code/paper-LaTeX/samples/NF_1_3.tiff",
):
    f = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(SRC_PATH1))
    mask = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(SRC_PATH2))
    # mask=mask[:,:,0]
    print(f.shape)
    print(mask.shape)

    new_f = f + np.random.uniform(0.5, 0.6, f.shape) * (1.0 - mask) * (1.0 - f)

    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(new_f.astype("float32")), SAVE_PATH)

def x_noise_fesion(
        SRC_PATH1="E:/project/MultimodalGeneration/src_code/paper-LaTeX/samples/BRATS/1584786325.jpg",
        SRC_PATH2="E:/project/MultimodalGeneration/src_code/paper-LaTeX/samples/BRATS/1584786435.jpg",
        SAVE_PATH="E:/project/MultimodalGeneration/src_code/paper-LaTeX/samples/BRATS/",
):
    f = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(SRC_PATH1))
    f = np.asarray(mynorm(f[:,:,0]) > 0.5).astype("float32")

    mask = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(SRC_PATH2))
    mask = 1.0-np.asarray(mynorm(mask[:, :, 0]) > 0.08).astype("float32")
    mask = signal.medfilt2d(mask, kernel_size=17)
    # mask=mask[:,:,0]
    # print(f.shape)
    # print(mask.shape)

    new_f = f + np.random.uniform(0.5, 0.6, f.shape) * (1.0 - mask) * (1.0 - f)

    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(mask.astype("float32")), SAVE_PATH+"mask.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(new_f.astype("float32")), SAVE_PATH+"fusion.tiff")

def m_noise_fesion(
        SRC_PATH1="E:/project/MultimodalGeneration/src_code/paper-LaTeX/samples/BRATS/f7.jpg",
        SRC_PATH2="E:/project/MultimodalGeneration/src_code/paper-LaTeX/samples/BRATS/m7.jpg",
        SAVE_PATH="E:/project/MultimodalGeneration/src_code/paper-LaTeX/samples/BRATS/",
):
    f = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(SRC_PATH1))
    f = np.asarray(mynorm(f[:,:,0]) > 0.5).astype("float32")

    mask = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(SRC_PATH2))
    mask = np.asarray(mynorm(mask[:,:,0]) > 0.5).astype("float32")
    # mask = signal.medfilt2d(mask, kernel_size=17)

    new_f = f + np.random.uniform(0.5, 0.6, f.shape) * (1.0 - mask) * (1.0 - f)

    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(f.astype("float32")), SAVE_PATH+"new_f7.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(mask.astype("float32")), SAVE_PATH+"new_m7.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(new_f.astype("float32")), SAVE_PATH+"fusion7.tiff")


def mask_fesion(
        SRC_PATH1="E:/project/MultimodalGeneration/src_code/samples/fxyzw_.tiff",
        SRC_PATH2="E:/project/MultimodalGeneration/src_code/samples/mask_l_.tiff",
        SAVE_PATH="E:/project/MultimodalGeneration/src_code/samples/nfxyzw_.tiff",
):
    f = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(SRC_PATH1))
    mask = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(SRC_PATH2))
    # mask=mask[:,:,0]
    print(f.shape)
    print(mask.shape)

    new_f = f * mask

    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(new_f.astype("float32")), SAVE_PATH)


if __name__ == '__main__':
    binary_png_run(
        SRC_PATH="quyili.jpg",
        SAVE_PATH="quyili.png",
    )