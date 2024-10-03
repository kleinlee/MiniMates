import numpy as np
import cv2
import random
import tqdm
import glob
import pickle
import torch
from torch.utils.data import Dataset
from obj_utils.opengl_texture_render import render2cv,set_ref_texture
from talkingface.utils import INDEX_LIPS_INNER,main_keypoints_index
from obj_utils.utils import adjust_verts

def generate_prompt(keypoints,mode = "face",size = 256 ,ref_image = None, ref_vt = None):
    if mode == "texture":
        set_ref_texture(ref_vt, ref_image)
        render_256 = render2cv(keypoints[:, :3], 2*size, size)
    else:
        render_256 = render2cv(keypoints[:, :3], size, size)
    return render_256

def get_image(A_path, crop_coords, input_type, resize= 256):
    (x_min, y_min, x_max, y_max) = crop_coords
    size = x_max - x_min

    if input_type == 'mediapipe':
        pose_pts = (A_path - np.array([x_min, y_min, 0])) * resize / size
        return pose_pts
    elif input_type == 'img':
        img_output = A_path[y_min:y_max, x_min:x_max, :]
        img_output = cv2.resize(img_output, (resize, resize))
        return img_output
    elif input_type == 'pose':
        head_pose = A_path.copy()
        head_pose[0] = head_pose[0] * resize / size
        head_pose[1:3] = (head_pose[1:3] - np.array([x_min, y_min])) * resize / size
        return head_pose
    else:
        print("get_image input_type error!")
        exit()

def get_standard_image(img, kps, crop_coords, resize= 256):
    h = img.shape[0]
    w = img.shape[1]
    (x_min, y_min, x_max, y_max) = [int(ii) for ii in crop_coords]
    size = x_max - x_min
    img_new = np.zeros([int(size), int(size), img.shape[2]], dtype=np.uint8)

    # 确定裁剪区域上边top和左边left坐标，中心点是(x2d.max() + x2d.min()/2, y2d.max() + y2d.min()/2)
    top = int(y_min)
    left = int(x_min)
    # 裁剪区域与原图的重合区域
    top_coincidence = int(max(top, 0))
    bottom_coincidence = int(min(y_max, h))
    left_coincidence = int(max(left, 0))
    right_coincidence = int(min(x_max, w))
    # print(x_min, y_min, x_max, y_max)
    # print(bottom_coincidence, top)
    # print(top_coincidence - top,bottom_coincidence - top, left_coincidence - left,right_coincidence - left)
    # print(img_new.shape)
    img_new[top_coincidence - top:bottom_coincidence - top, left_coincidence - left:right_coincidence - left, :] = img[
                                                                                                                   top_coincidence:bottom_coincidence,
                                                                                                                   left_coincidence:right_coincidence,
                                                                                                                   :]
    img_new = cv2.resize(img_new, (resize, resize))

    factor = resize/size
    kps[:, 0] = (kps[:, 0] - left) * factor
    kps[:, 1] = (kps[:, 1] - top) * factor
    kps[:, 2] = (kps[:, 2] - 0) * factor
    return img_new, kps

def crop_(pts_array_origin, img_w, img_h):
    if pts_array_origin.sum() == 0:
        return np.array([0, 0, img_w, img_h])
    x_min, y_min, x_max, y_max = np.min(pts_array_origin[:, :, 0]), np.min(
        pts_array_origin[:, :, 1]), np.max(
        pts_array_origin[:, :, 0]), np.max(pts_array_origin[:, :, 1])
    new_w = (x_max - x_min) * 2
    new_h = (y_max - y_min) * 2
    new_size = int(max(new_w / 2., new_h / 2.))
    center_x = int((x_max + x_min) / 2.)
    # center_y = int((y_max + y_min) / 2.)
    center_y = int(y_min + (y_max - y_min) * 0.35)
    x_min, y_min, x_max, y_max = int(center_x - new_size), int(center_y - new_size), int(
        center_x + new_size), int(center_y + new_size)
    return np.array([x_min, y_min, x_max, y_max])

class Dataset_Face2facerho(Dataset):
    '''
    dict_info:
        images: 图片文件路径
        prompt: 图片对应的限制文件路径
        prompt_mode: 分为None、keypoint、image（可以使segmentation或depth map、UV map）
    '''
    def __init__(self, dict_info, is_train = False, mode = "opengl", out_size = 256):
        super(Dataset_Face2facerho, self).__init__()
        self.out_size = out_size
        self.is_train = is_train

        self.images = dict_info["images"]
        self.prompt = dict_info["prompt"]
        self.head_pose = dict_info["head_pose"]
        self.driven_teeth_images = dict_info["driven_teeth_image"]
        self.driven_teeth_rect = dict_info["driven_teeth_rect"]
        self.sample_num = np.sum([len(i) for i in self.images])

        # list: 每个视频序列的视频块个数
        self.clip_count_list = []  # number of frames in each sequence
        self.crop_coords = []
        for video_index,path in enumerate(self.images):
            if len(path) > 2:
                self.clip_count_list.append(len(path))
            h,w = cv2.imread(path[0]).shape[:2]
            self.crop_coords.append(crop_(self.prompt[video_index], w, h))
        self.n_ref = 1
        self.mode = mode

        self.source_index = None

    def set_source_index(self, index):
        self.source_index = index

    def __getitem__(self, index):
        if self.is_train:
            video_index = random.randint(0, len(self.images) - 1)
            # 最多重复三次挑选，意图为改变样本head_pose分布，使得source和drving的head_pose相差尽可能大一些
            for jj in range(3):
                clips_index = random.sample(range(self.clip_count_list[video_index]), 2)  # 从当前视频选1+n_ref个图片
                source_index = clips_index[0]
                drving_index = clips_index[1]
                head_pose_dist = np.linalg.norm(self.head_pose[video_index][source_index, 3:] - self.head_pose[video_index][drving_index, 3:])
                if head_pose_dist > 10:
                    break
            # print(jj, head_pose_dist)
        else:
            video_index = 0
            if self.source_index is not None:
                source_index = self.source_index
            else:
                source_index = 0
            drving_index = index


        source_img = cv2.imread(self.images[video_index][source_index], cv2.IMREAD_UNCHANGED)
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGRA2RGBA)
        drving_img = cv2.imread(self.images[video_index][drving_index], cv2.IMREAD_UNCHANGED)
        drving_img = cv2.cvtColor(drving_img, cv2.COLOR_BGRA2RGBA)
        source_prompt = self.prompt[video_index][source_index].copy()
        drving_prompt = self.prompt[video_index][drving_index].copy()


        # source_img = cv2.resize(source_img, (self.out_size, self.out_size))
        # drving_img = cv2.resize(drving_img, (self.out_size, self.out_size))
        # crop_coords = crop_(self.prompt[video_index], img_w, img_h)
        source_face_rect = crop_(source_prompt[np.newaxis, :, :], 10000, 10000)
        source_img,source_prompt = get_standard_image(source_img, source_prompt, source_face_rect, resize=self.out_size)
        source_prompt = adjust_verts(source_prompt)

        source_teeth_rect = self.driven_teeth_rect[video_index][source_index]
        crop_size = source_face_rect[2] - source_face_rect[0]
        # source_teeth_rect = np.array([(source_teeth_rect[0] - self.crop_coords[video_index][0]) * self.out_size / crop_size,
        #                      (source_teeth_rect[1] - self.crop_coords[video_index][1]) * self.out_size / crop_size,
        #                      (source_teeth_rect[2] - self.crop_coords[video_index][0]) * self.out_size / crop_size,
        #                      (source_teeth_rect[3] - self.crop_coords[video_index][1]) * self.out_size / crop_size], dtype = int)
        source_crop_coords = source_face_rect
        source_teeth_rect = np.array(
            [max(0, (source_teeth_rect[0] - source_crop_coords[0]) * self.out_size / crop_size),
             max(0, (source_teeth_rect[1] - source_crop_coords[1]) * self.out_size / crop_size),
             min(self.out_size, (source_teeth_rect[2] - source_crop_coords[0]) * self.out_size / crop_size),
             min(self.out_size, (source_teeth_rect[3] - source_crop_coords[1]) * self.out_size / crop_size)], dtype=int)


        crop_coords = None

        crop_coords = crop_(drving_prompt[np.newaxis, :, :], 10000, 10000)
        drving_img,drving_prompt = get_standard_image(drving_img, drving_prompt, crop_coords, resize=self.out_size)
        drving_prompt = adjust_verts(drving_prompt)


        drving_teeth_rect = self.driven_teeth_rect[video_index][drving_index]
        crop_size = crop_coords[2] - crop_coords[0]
        drving_teeth_rect = np.array(
            [max(0, (drving_teeth_rect[0] - crop_coords[0]) * self.out_size / crop_size),
             max(0, (drving_teeth_rect[1] - crop_coords[1]) * self.out_size / crop_size),
             min(self.out_size, (drving_teeth_rect[2] - crop_coords[0]) * self.out_size / crop_size),
             min(self.out_size, (drving_teeth_rect[3] - crop_coords[1]) * self.out_size / crop_size)], dtype=int)

        drving_teeth_mask = np.zeros([self.out_size, self.out_size, 3], dtype = np.uint8)
        # print("drving_prompt", drving_prompt.shape)
        drving_teeth_pts = drving_prompt[main_keypoints_index][INDEX_LIPS_INNER,:2].reshape((-1,1,2)).astype(np.int32)
        cv2.fillPoly(drving_teeth_mask, [drving_teeth_pts], color=(1,1,1))
        # cv2.imshow("s", drving_teeth_mask)
        # cv2.waitKey(-1)

        drving_prompt = generate_prompt(drving_prompt, mode="texture", size=self.out_size, ref_image=source_img, ref_vt=source_prompt[:, :2]/self.out_size)
        # print(drving_prompt.shape)
        driving_img_face = drving_prompt[:self.out_size, self.out_size:]
        # driving_img_face = driving_img_face*(1-drving_teeth_mask) + drving_img*drving_teeth_mask
        drving_prompt = drving_prompt[:self.out_size, :self.out_size]
        source_prompt = generate_prompt(source_prompt, mode="face", size=self.out_size)

        if source_teeth_rect[2] - source_teeth_rect[0] > 0 and source_teeth_rect[3] - source_teeth_rect[1] > 0:
            source_teeth_img = cv2.imread(self.driven_teeth_images[video_index][source_index])
            source_teeth_img = cv2.resize(source_teeth_img[:, source_teeth_img.shape[1] // 2:, :], (
                source_teeth_rect[2] - source_teeth_rect[0], source_teeth_rect[3] - source_teeth_rect[1]))
            source_prompt[source_teeth_rect[1]:source_teeth_rect[3], source_teeth_rect[0]:source_teeth_rect[2], 1][
                np.where(source_teeth_img[:, :, 0] == 0)] = 255
            source_prompt[source_teeth_rect[1]:source_teeth_rect[3], source_teeth_rect[0]:source_teeth_rect[2], 2][
                np.where(source_teeth_img[:, :, 2] == 255)] = 255
            source_prompt[source_teeth_rect[1]:source_teeth_rect[3], source_teeth_rect[0]:source_teeth_rect[2], 3][
                np.where(source_teeth_img[:, :, 0] == 0)] = 255
            source_prompt[source_teeth_rect[1]:source_teeth_rect[3], source_teeth_rect[0]:source_teeth_rect[2], 3][
                np.where(source_teeth_img[:, :, 2] == 255)] = 255
        if drving_teeth_rect[2] - drving_teeth_rect[0] > 0 and drving_teeth_rect[3] - drving_teeth_rect[1] > 0:
            drving_teeth_img = cv2.imread(self.driven_teeth_images[video_index][drving_index])
            drving_teeth_img = cv2.resize(drving_teeth_img[:, drving_teeth_img.shape[1] // 2:, :], (
                drving_teeth_rect[2] - drving_teeth_rect[0], drving_teeth_rect[3] - drving_teeth_rect[1]))
            drving_prompt[drving_teeth_rect[1]:drving_teeth_rect[3], drving_teeth_rect[0]:drving_teeth_rect[2], 1][
                np.where(drving_teeth_img[:, :, 0] == 0)] = 255
            drving_prompt[drving_teeth_rect[1]:drving_teeth_rect[3], drving_teeth_rect[0]:drving_teeth_rect[2], 2][
                np.where(drving_teeth_img[:, :, 2] == 255)] = 255
            drving_prompt[drving_teeth_rect[1]:drving_teeth_rect[3], drving_teeth_rect[0]:drving_teeth_rect[2], 3][
                np.where(drving_teeth_img[:, :, 0] == 0)] = 255
            drving_prompt[drving_teeth_rect[1]:drving_teeth_rect[3], drving_teeth_rect[0]:drving_teeth_rect[2], 3][
                np.where(drving_teeth_img[:, :, 2] == 255)] = 255

        input_data = {
            'source_img': torch.from_numpy(source_img / 255.).float().permute(2, 0, 1),
            'drving_img': torch.from_numpy(drving_img / 255.).float().permute(2, 0, 1),
            'source_prompt': torch.from_numpy(source_prompt / 255.).float().permute(2, 0, 1),
            'drving_prompt': torch.from_numpy(drving_prompt / 255.).float().permute(2, 0, 1),
            'driving_img_face': torch.from_numpy(driving_img_face / 255.).float().permute(2, 0, 1),
            'video_name': self.images[video_index][source_index]
        }
        return input_data

    def __len__(self):
        if self.is_train:
            return len(self.images)
        else:
            return len(self.images[0])
        # return self.sample_num

def data_preparation(train_video_list):
    img_all = []
    keypoints_all = []
    head_pose_all = []
    teeth_img_all = []
    teeth_rect_all = []
    for i in tqdm.tqdm(train_video_list):
        # for i in ["xiaochangzhang/00004"]:
        model_name = i
        img_filelist = glob.glob("{}/image2/*.png".format(model_name))
        img_filelist.sort()
        if len(img_filelist) == 0:
            continue
        img_all.append(img_filelist)

        Path_output_pkl = "{}/keypoint_rotate.pkl".format(model_name)
        with open(Path_output_pkl, "rb") as f:
            images_info = pickle.load(f)
        keypoints_all.append(images_info[:, :, :])

        Path_output_pkl = "{}/head_pose.pkl".format(model_name)
        with open(Path_output_pkl, "rb") as f:
            head_pose = pickle.load(f)
        head_pose_all.append(head_pose)

        img_teeth_filelist = glob.glob("{}/teeth_seg/*.png".format(model_name))
        img_teeth_filelist.sort()
        teeth_img_all.append(img_teeth_filelist)

        teeth_rect_array = np.loadtxt("{}/teeth_seg/all.txt".format(model_name))
        teeth_rect_all.append(teeth_rect_array)

    print("train size: ", len(img_all))
    dict_info = {}
    dict_info["images"] = img_all
    dict_info["prompt"] = keypoints_all
    dict_info["head_pose"] = head_pose_all
    dict_info["driven_teeth_rect"] = teeth_rect_all
    dict_info["driven_teeth_image"] = teeth_img_all
    return dict_info

# def Tensor2img(tensor_, channel_index, mode = "gpu"):
#     if mode == "gpu":
#         frame = tensor_[channel_index:channel_index + 4, :, :].detach().squeeze(0).cpu().float().numpy()
#     else:
#         frame = tensor_[channel_index:channel_index + 4, :, :].float().numpy()
#     frame = np.transpose(frame, (1, 2, 0)) * 255.0
#     frame = frame.clip(0, 255)
#     return frame[:,:,:3].astype(np.uint8)

def Tensor2img(tensor_, channel_index, mode = "gpu", background = [0, 0, 0]):
    if mode == "gpu":
        frame = tensor_[channel_index:channel_index + 4, :, :].detach().squeeze(0).cpu().float().numpy()
    else:
        frame = tensor_[channel_index:channel_index + 4, :, :].float().numpy()
    frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[2] == 4:
        frame[:,:,0] = frame[:,:,0]*frame[:,:,3]
        frame[:,:,1] = frame[:,:,1]*frame[:,:,3]
        frame[:,:,2] = frame[:,:,2]*frame[:,:,3]
        frame = frame[:,:,:3]
    frame = frame * 255.0
    frame = frame.clip(0, 255)
    return frame[:,:,:3].astype(np.uint8)