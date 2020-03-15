# _*_ coding:utf-8 _*_
import os
import numpy as np
import SimpleITK


def mynorm(input):
    output = (input - np.min(input)
              ) / (np.max(input) - np.min(input))
    return output.astype("float32")


def binary(x, beta=0.5):
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


def run(
        SRC_PATH="E:/project/MultimodalGeneration/src_code/paper-LaTeX/figures/SWM_SkrGAN_F.png",
        SAVE_PATH="E:/project/MultimodalGeneration/src_code/paper-LaTeX/figures/new_SWM_SkrGAN_F.tiff",
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
    mask_fesion()
