import tqdm
import numpy as np
import cv2
import glob
import os
import math
import pickle
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

def ExtractFromVideo(model_name, video_path):
    # ### 将视频提取为60FPS的png集合 和 音频wav文件
    # video_path = r"E:/data/video/video/{}.mp4".format(model_name)
    bg_path = "../preparation/{}/image".format(model_name)
    os.makedirs(bg_path, exist_ok=True)

    wave_path = "../preparation/{}/wav.wav".format(model_name)
    if not os.path.isfile(wave_path):
        os.system("ffmpeg -i {}  -ac 1 -ar 16000 {}".format(video_path, wave_path))

    frames_list = []
    cap = cv2.VideoCapture(video_path)
    vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 宽度
    vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 高度
    frame_index_ = 0
    while cap.isOpened():
        ret, frame = cap.read()  # 按帧读取视频
        # #到视频结尾时终止
        if ret is False:
            break
        frames_list.append(frame)
        frame_index_ += 1
        if frame_index_/30 > 600:
            break
    cap.release()  # 释放视频对象


    pts_3d = np.zeros([len(frames_list), 478, 3])
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:

        for index, frame in tqdm.tqdm(enumerate(frames_list)):
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                print("****** WARNING! No face detected! ******")
                pts_3d[index] = 0
                return
                # continue
            image_height, image_width = frame.shape[:2]
            for face_landmarks in results.multi_face_landmarks:
                for index_, i in enumerate(face_landmarks.landmark):
                    x_px = min(math.floor(i.x * image_width), image_width - 1)
                    y_px = min(math.floor(i.y * image_height), image_height - 1)
                    z_px = min(math.floor(i.z * image_height), image_width - 1)
                    pts_3d[index, index_] = np.array([x_px, y_px, z_px])

    # 计算整个视频中人脸的范围

    x_min, y_min, x_max, y_max = np.min(pts_3d[:, :, 0]), np.min(
        pts_3d[:, :, 1]), np.max(
        pts_3d[:, :, 0]), np.max(pts_3d[:, :, 1])
    new_w = int((x_max - x_min) * 0.55)*2
    new_h = int((y_max - y_min) * 0.6)*2
    center_x = int((x_max + x_min) / 2.)
    center_y = int(y_min + (y_max - y_min) * 0.6)
    size = max(new_h, new_w)
    x_min, y_min, x_max, y_max = int(center_x - size // 2), int(center_y - size // 2), int(
        center_x + size // 2), int(center_y + size // 2)

    # 确定裁剪区域上边top和左边left坐标
    top = y_min
    left = x_min
    # 裁剪区域与原图的重合区域
    top_coincidence = int(max(top, 0))
    bottom_coincidence = int(min(y_max, vid_height))
    left_coincidence = int(max(left, 0))
    right_coincidence = int(min(x_max, vid_width))

    out_size = 512
    scale = 512. / size
    pts_3d = (pts_3d - np.array([left, top, 0])) * scale
    Path_output_pkl = "../preparation/{}/keypoint_rotate.pkl".format(model_name)
    with open(Path_output_pkl, "wb") as f:
        pickle.dump(pts_3d, f)

    np.savetxt(r"../preparation/{}/face_rect.txt".format(model_name), np.array([center_x, center_y, size]))
    print(np.array([x_min, y_min, x_max, y_max]))

    for index, frame in tqdm.tqdm(enumerate(frames_list)):
        img_new = np.zeros([size, size, 3], dtype=np.uint8)
        img_new[top_coincidence - top:bottom_coincidence - top, left_coincidence - left:right_coincidence - left,:] = \
            frame[top_coincidence:bottom_coincidence, left_coincidence:right_coincidence, :]
        img_new = cv2.resize(img_new, (out_size, out_size))
        cv2.imwrite("../preparation/{}/image/{:0>6d}.png".format(model_name, index), img_new)
        # cv2.imshow("sss", img_new)
        # cv2.waitKey(30)

if __name__ == '__main__':
    path = r"E:\AI\DH008_few_shot\data4"
    existed_data = os.listdir(r"../preparation")
    file_list = os.listdir(path)
    for item in file_list[:]:
        model_name = os.path.basename(item)[:-4]
        item_ = os.path.join(path, item)

        if model_name in existed_data:
            continue

        print(model_name)
        ExtractFromVideo(model_name, item_)
        # break

