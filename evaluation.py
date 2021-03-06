# -*- coding: utf-8 -*-
import metrics.new_msssim as MSSSIM
import metrics.frechet_inception_distance as FID

out1 = MSSSIM.calculate_msssim_given_paths(["D:/BaiduYunDownload/EVALU/X2", "D:/BaiduYunDownload/EVALU/Y2"])
out2 = FID.calculate_fid_given_paths(["D:/BaiduYunDownload/EVALU/X2", "D:/BaiduYunDownload/EVALU/Y2"],
                                     "D:/project/MultimodalGeneration/metrics")

print(out1, out2)
