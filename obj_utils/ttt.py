from opengl_texture_render import render2cv
import pickle
import cv2
import time
Path_output_pkl = r"E:\Code\Gan\DH008_few_shot\preparation\adf0ca96-b60f-11ed--0\keypoint_rotate.pkl"
import glob
img_filenames = glob.glob(r"E:\Code\Gan\DH008_few_shot\preparation\adf0ca96-b60f-11ed--0\image/*.png")
with open(Path_output_pkl, "rb") as f:
    images_info = pickle.load(f)
start_time = time.time()
for frame_index in range(len(images_info)*100):
    pts_array_ = images_info[frame_index%len(images_info), :, :2]
    # render2cv(pts_array_)
    rgb = render2cv(pts_array_, 512, 512)

    print(frame_index, time.time() - start_time, (frame_index + 1) / (time.time() - start_time + 0.0001))
    cv2.imshow('scene', rgb)
    cv2.waitKey(30)