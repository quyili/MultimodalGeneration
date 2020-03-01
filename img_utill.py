# _*_ coding:utf-8 _*_
import os
import numpy as np
import SimpleITK

def mynorm(input):
    output = (input - np.min(input)
              ) / (np.max(input) - np.min(input))
    return output.astype("float32")           

def binary(x,beta=0.5):
    return np.asarray(x > beta).astype("float32")

def binary_run(
        SRC_PATH="E:/project/MultimodalGeneration/src_code/paper-LaTeX/figures/SWM_SkrGAN_F.png",
        SAVE_PATH="E:/project/MultimodalGeneration/src_code/paper-LaTeX/figures/new_SWM_SkrGAN_F.tiff",
):
        input_x = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(SRC_PATH))
        input_x=(input_x[:,:,0]+input_x[:,:,1]+input_x[:,:,2])/3.0
        print(input_x.shape)
        input_x=mynorm(input_x)
        input_x=binary(input_x)
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(1.0-input_x), SAVE_PATH)


def run(
        SRC_PATH="E:/project/MultimodalGeneration/src_code/paper-LaTeX/figures/SWM_SkrGAN_F.png",
        SAVE_PATH="E:/project/MultimodalGeneration/src_code/paper-LaTeX/figures/new_SWM_SkrGAN_F.tiff",
):
    input_x = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(SRC_PATH))
    input_x = (input_x[:, :, 0] + input_x[:, :, 1] + input_x[:, :, 2]) / 3.0
    print(input_x.shape)
    input_x = mynorm(input_x)
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(input_x), SAVE_PATH)


if __name__ == '__main__':
    run(
        SRC_PATH="E:/project/MultimodalGeneration/src_code/paper-LaTeX/figures/fbct_SkrGAN.png",
        SAVE_PATH="E:/project/MultimodalGeneration/src_code/paper-LaTeX/figures/fbct_SkrGAN.tiff",
    )
