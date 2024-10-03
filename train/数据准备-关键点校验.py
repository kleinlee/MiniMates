import pickle
import cv2
import numpy as np
import tqdm
import glob
import os
file_name = r"F:\C\AI\CV\TalkingFace\preparation\bilibili_hq\video_bili_5"
# 文件校验
Path_output_pkl = "{}/keypoint_rotate.pkl".format(file_name)
with open(Path_output_pkl, "rb") as f:
    pts_3d = pickle.load(f)
point_size = 1
point_color = (0, 0, 255)  # BGR
thickness = 4  # 0 、4、8
frames_list = glob.glob(r"{}/image/*.png".format(file_name))
for index, frame in tqdm.tqdm(enumerate(frames_list)):
    pts_ = pts_3d[index]
    img = cv2.imread(frame)
    for coor in pts_:
        # coor = (coor +1 )/2.
        cv2.circle(img, (int(coor[0]), int(coor[1])), point_size, point_color, thickness)
    cv2.imshow("a", img)
    cv2.waitKey(30)