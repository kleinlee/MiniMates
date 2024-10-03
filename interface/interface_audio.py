import os
import random
os.environ["KMP_DUPLICATE_LIB_OK"] = "true"
current_dir = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(current_dir, ".."))
import cv2
import numpy as np
import pickle
import numpy as np
import tqdm
import torch
import glob
import uuid
import sys
import os
from obj_utils.utils import generateRenderInfo,adjust_verts,device

from interface.utils import rotate,face_interface,face_process,audio_interface,eularAngle2Matrix,generate_bg_mask
from obj_utils.utils import INDEX_MP_LIPS
out_size = 384
model_size_ = 384
current_dir = os.path.dirname(os.path.abspath(__file__))
bg_mask = generate_bg_mask(out_size)
tensor_bg_mask = torch.from_numpy(bg_mask).float().permute(2, 0, 1).unsqueeze(0).to(device)

from train.data.dataset_faceReenactment3 import crop_,get_image,generate_prompt,Tensor2img


render_verts_, _ = generateRenderInfo()
face_pts_mean = render_verts_[:478, :3]
face_pts_mean = adjust_verts(face_pts_mean)
teeth_verts_ = render_verts_[478:, :3]
head_joint = np.array([out_size * 0.5, out_size * 3 / 4, -0.])
def run_audio(img_path, wavpath, output_path, template_path = None):
    img_primer_rgba, source_img, source_crop_pts, source_crop_pts_vt, source_crop_coords = face_process(img_path, out_size)

    # print(source_img.shape)
    # cv2.imshow("s", source_img)
    # cv2.waitKey(-1)

    # 找到标准人脸，求出旋转矩阵
    from talkingface.run_utils import calc_face_mat

    mat_list, pts_normalized_list, face_pts_mean_personal = calc_face_mat(source_crop_pts[np.newaxis, :, :],
                                                                          face_pts_mean)
    source_rotateM = mat_list[0]
    # 标准人脸旋转回去找到标准嘴巴部分的顶点
    face_pts_mean_personal[INDEX_MP_LIPS] = face_pts_mean[INDEX_MP_LIPS]*0.6 + face_pts_mean_personal[INDEX_MP_LIPS]*0.4


    teeth_verts = teeth_verts_.copy()

    source_prompt = generate_prompt(source_crop_pts, mode="face", size=out_size)
    tensor_source_img = torch.from_numpy(source_img[:, :, :4] / 255.).float().permute(2, 0, 1).unsqueeze(0).to(
        device)
    tensor_source_prompt = torch.from_numpy(source_prompt / 255.).float().permute(2, 0, 1).unsqueeze(0).to(
        device)

    pts_audio_driving = audio_interface(wavpath)
    frame_num = len(pts_audio_driving)
    import uuid
    task_id = str(uuid.uuid1())
    save_path = "{}.mp4".format(task_id)
    # vid_width, vid_height = model_size_ * 4, model_size_
    vid_width, vid_height = img_primer_rgba.shape[1], img_primer_rgba.shape[0]
    videoWriter = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (int(vid_width), int(vid_height)))

    if template_path is not None:
        with open(template_path, "rb") as f:
            [rotate_trans_param, blendshape_verts_bias, rect_video_face] = pickle.load(f)
        weight_list = np.loadtxt("../obj_utils/weight_list.txt")[:, np.newaxis]

        rotate_trans_param2 = rotate_trans_param.copy()
        blendshape_verts_bias2 = blendshape_verts_bias.copy()
        for i in range(frame_num//len(rotate_trans_param)):
            if i%2 == 0:
                rotate_trans_param2 = np.concatenate([rotate_trans_param2, rotate_trans_param[::-1]], axis = 0)
                blendshape_verts_bias2 = np.concatenate([blendshape_verts_bias2, blendshape_verts_bias[::-1]], axis=0)
            else:
                rotate_trans_param2 = np.concatenate([rotate_trans_param2, rotate_trans_param], axis=0)
                blendshape_verts_bias2 = np.concatenate([blendshape_verts_bias2, blendshape_verts_bias], axis=0)
        rotate_trans_param = rotate_trans_param2
        blendshape_verts_bias = blendshape_verts_bias2

    else:
        pass
    for frame_index in tqdm.tqdm(range(frame_num)):
        pts_3d = pts_audio_driving[frame_index].reshape(-1, 3) + face_pts_mean_personal

        teeth_verts_copy = teeth_verts.copy()
        teeth_verts_copy[18:, 1] = teeth_verts_copy[18:, 1] - (pts_3d[13, 1] - pts_3d[14, 1]) * 0.5
        pts_array_ = np.concatenate([pts_3d, teeth_verts_copy], axis=0)

        # driving 驱动
        keypoints = np.ones([4, len(pts_array_)])
        keypoints[:3, :] = pts_array_.T
        source_pts = source_rotateM.dot(keypoints).T
        source_pts = source_pts[:, :3]

        driving_pts = source_pts

        if template_path is not None:
            if frame_index + 1 > len(rotate_trans_param):
                break
            euler_angles = rotate_trans_param[frame_index, :3]
            blendshape_verts_driving = blendshape_verts_bias[frame_index]*(1-weight_list) * 0.5
            rotationMatrix = eularAngle2Matrix(euler_angles)
            new_source_pts = driving_pts

            new_source_pts = new_source_pts - head_joint
            new_source_pts[:478] = new_source_pts[:478] + blendshape_verts_driving
            keypoints = np.ones([3, len(new_source_pts)])
            keypoints[:3, :] = new_source_pts.T
            keypoints_rotated = rotationMatrix.dot(keypoints).T

            driving_pts = keypoints_rotated + head_joint

        drving_prompt = generate_prompt(driving_pts[:], mode="texture", size=out_size,
                                        ref_image=source_img, ref_vt=source_crop_pts_vt)
        driving_img_face = drving_prompt[:out_size, out_size:]
        drving_prompt = drving_prompt[:out_size, :out_size]

        tensor_drving_img = tensor_source_img
        tensor_drving_prompt = torch.from_numpy(drving_prompt / 255.).float().permute(2, 0, 1).unsqueeze(0).to(
            device)
        tensor_driving_img_face = torch.from_numpy(driving_img_face / 255.).float().permute(2, 0, 1).unsqueeze(
            0).to(device)

        with torch.no_grad():
            in0 = tensor_source_img
            in1 = tensor_source_prompt
            in2 = tensor_drving_img
            in3 = tensor_driving_img_face
            in4 = tensor_drving_prompt
            in5 = tensor_bg_mask
            # in6 = F.interpolate(tensor_wrap_mask, size=(model_size_, model_size_), mode='nearest')
            # print(in0.size(), in1.size(), in2.size(), in3.size(), in4.size(), in5.size())
            fake_out = face_interface(in0, in1, in2, in3, in4, in5, model_size_)
        fake_out = Tensor2img(fake_out[0], 0)

        fake_out = cv2.resize(fake_out, (
            source_crop_coords[2] - source_crop_coords[0], source_crop_coords[3] - source_crop_coords[1]))
        out2 = img_primer_rgba[:, :, :3].copy()
        x_min, y_min, x_max, y_max = source_crop_coords




        #     x_min = max(0, x_min)
        #     y_min = max(0, y_min)
        #     x_max = min(x_max, img_w)
        #     y_max = min(y_max, img_h)
        print(x_min, y_min, x_max, y_max, fake_out.shape)
        x_min_fake, y_min_fake, x_max_fake, y_max_fake = [0, 0, fake_out.shape[1], fake_out.shape[0]]

        if x_min < 0:
            x_min_fake = -x_min
            x_min = 0
        if y_min < 0:
            y_min_fake = -y_min
            y_min = 0
        if x_max > out2.shape[1]:
            x_max = out2.shape[1]
            x_max_fake = x_min_fake + (x_max - x_min)

        if y_max > out2.shape[0]:
            y_max = out2.shape[0]
            y_max_fake = y_min_fake + (y_max - y_min)
        # print(y_min_fake, y_max_fake, x_min_fake, x_max_fake, fake_out.shape)
        # cv2.imshow("s", fake_out[y_min_fake:y_max_fake, x_min_fake:x_max_fake])
        # cv2.waitKey(-1)
        # cv2.imshow("s", out2[y_min:y_max, x_min:x_max])
        # cv2.waitKey(-1)
        # print(out2.shape)
        # print(x_min_fake, y_min_fake, x_max_fake, y_max_fake, x_min, y_min, x_max, y_max, fake_out.shape)
        out2[y_min:y_max, x_min:x_max] = fake_out[y_min_fake:y_max_fake, x_min_fake:x_max_fake]

        frame = out2
        # cv2.imshow('scene', frame[..., ::-1])
        # cv2.waitKey(40)
        videoWriter.write(frame[..., ::-1])
    videoWriter.release()
    val_video = output_path
    wav_path = wavpath
    os.system(
        "ffmpeg -i {} -i {} -c:v libx264 -pix_fmt yuv420p {}".format(save_path, wav_path, val_video))
    os.remove(save_path)
    cv2.destroyAllWindows()

def main():
    # 检查命令行参数的数量
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: python interface_audio.py <img_path> <wav_path> <output_path> <template_path>")
        sys.exit(1)  # 参数数量不正确时退出程序

    img_path = sys.argv[1]
    wav_path = sys.argv[2]
    output_path = sys.argv[3]
    if len(sys.argv) == 4:
        template_path = None
    else:
        template_path = sys.argv[4]
    print(f"img path is set to: {img_path}, wav path is set to: {wav_path}, output path is set to: {output_path}")
    run_audio(img_path, wav_path, output_path, template_path)

if __name__ == "__main__":
    main()

