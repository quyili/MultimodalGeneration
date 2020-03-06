# _*_ coding:utf-8 _*_
import os
import numpy as np
import SimpleITK
import cv2
from PIL import Image

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

    index_files =  os.listdir(SRC_PATH)
    out_size=[512,512,3]
    for file in index_files:
         L_arr = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(SRC_PATH+ file))
         print(L_arr.shape)
         if  len(L_arr.shape)==2 :
             img = cv2.merge([L_arr [:,:], L_arr [:,:], L_arr [:,:]])
         elif  L_arr.shape[2]==1 :
             img = cv2.merge([L_arr [:,:,0], L_arr [:,:,0], L_arr [:,:,0]])
         elif  L_arr.shape[2]==3:
             img = cv2.merge([L_arr [:,:,0], L_arr [:,:,1], L_arr [:,:,2]])
         if out_size== None:
             img = cv2.resize(img, (FLAGS.image_size[0],FLAGS.image_size[1]), interpolation=cv2.INTER_NEAREST)
             img = np.asarray(img)[:,:,0:FLAGS.image_size[2]].astype('float32')
         else:
             img = cv2.resize(img, (out_size[0],out_size[1]), interpolation=cv2.INTER_NEAREST)  
             img = np.asarray(img)[:,:,0:out_size[2]].astype('float32')
         image=norm(img)
         print(image.shape)
         image=(image*255).astype(np.uint8)
         Image.fromarray(image).save(SAVE_X_PATH + file.replace(".jpg", ".tif"))
         #SimpleITK.WriteImage(SimpleITK.GetImageFromArray(image.reshape([512,512,3])), SAVE_X_PATH + file.replace(".jpg", ".tif"))

if __name__ == '__main__':
    mover(
        SRC_PATH="D:\\BaiduYunDownload\\FIRE\Images\\",
        SAVE_X_PATH="D:\\BaiduYunDownload\\SWM\\train\\X_\\",
    )
