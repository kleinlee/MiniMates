import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pickle
import cv2
import numpy as np
import random
import os
import glob
import copy
import torch
from torch.utils.data import DataLoader
from data.dataset_faceReenactment3 import Dataset_Face2facerho,data_preparation,Tensor2img
path_ = r"../../preparation/mingxingzhufushipin"
video_list = glob.glob("{}/*".format(path_))
# print(video_list)
# exit()
out_size = 384
# video_list = video_list[150:175]
video_list = random.sample(video_list, 20)
dict_info = data_preparation(video_list)
device = torch.device("cuda:0")
test_set = Dataset_Face2facerho(dict_info, is_train=True, mode="texture", out_size=out_size)
testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)

for iteration, batch in enumerate(testing_data_loader):
    source_img = batch["source_img"].to(device)
    source_prompt = batch["source_prompt"].to(device)
    driving_img = batch["drving_img"].to(device)
    driving_prompt = batch["drving_prompt"].to(device)
    driving_img_face = batch["driving_img_face"].to(device)

    video_name = os.path.split(batch["video_name"][0])[0]
    video_name = os.path.split(video_name)[0]
    video_name = os.path.split(video_name)[1]
    print(source_img.size(), source_prompt.size(), driving_img.size(), driving_prompt.size(), driving_img_face.size())

    frame0 = Tensor2img(source_img[0], 0).copy()
    frame1 = Tensor2img(source_prompt[0], 0)
    frame2 = Tensor2img(driving_img[0], 0)
    frame3 = Tensor2img(driving_prompt[0], 0)
    frame4 = Tensor2img(driving_img_face[0], 0)

    frame = np.concatenate([frame0, frame1, frame2, frame3, frame4], axis=1)
    # cv2.imwrite("ss.png", frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    cv2.imshow("ss", frame)
    # if iteration > 840:
    #     cv2.waitKey(-1)
    cv2.waitKey(-1)
    # break
cv2.destroyAllWindows()