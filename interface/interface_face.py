import os
import torch.nn.functional as F
os.environ["KMP_DUPLICATE_LIB_OK"] = "true"
current_dir = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(current_dir, ".."))
from train.data.dataset_faceReenactment3 import crop_,get_image,generate_prompt,Tensor2img,adjust_verts
import pickle
import torch
import uuid
import sys
import tqdm
import numpy as np
import cv2
import os
import glob
import pickle
device = "cuda" if torch.cuda.is_available() else "cpu"
from interface.utils import rotate,face_interface,face_process,rotation_matrix_to_euler_angles,eularAngle2Matrix,rgb_face_process,generate_bg_mask
from talkingface.run_utils import calc_face_mat
from obj_utils.utils import generateRenderInfo,device

render_verts_, _ = generateRenderInfo()
face_pts_mean = render_verts_[:478, :3]
face_pts_mean = adjust_verts(face_pts_mean)
teeth_verts_ = render_verts_[478:, :3]



out_size = 384
model_size_ = 384
current_dir = os.path.dirname(os.path.abspath(__file__))
bg_mask = generate_bg_mask(out_size)
tensor_bg_mask = torch.from_numpy(bg_mask).float().permute(2, 0, 1).unsqueeze(0).to(device)
from talkingface.mediapipe_utils import detect_face_mesh

# video_list = glob.glob(r"F:\C\AI\CV\TalkingFace\preparation\tiktok_zhongguodianyingbaodao3/*")
# video_list.sort()
# video_list = random.sample(video_list, 10)
def run_avatar(img_path):
    img_primer_rgba, source_img, source_crop_pts, source_crop_pts_vt, _ = face_process(img_path, out_size)
    source_prompt = generate_prompt(source_crop_pts, mode="face", size=out_size)

    head_joint = np.mean(source_crop_pts, axis=0)
    head_joint = np.array([out_size*0.5, out_size*3/4, -0.])
    # head_joint[2] = head_joint[2] + 60
    mat_list, pts_normalized_list, face_pts_mean_personal = calc_face_mat(source_crop_pts[np.newaxis, :, :],
                                                                          face_pts_mean)
    source_rotateM = mat_list[0]
    task_id = str(uuid.uuid1())
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 打开默认摄像头 (设备索引为 0)
    cap = cv2.VideoCapture(0)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    while True:
        # 读取摄像头的每一帧
        ret, frame = cap.read()
        # print("frame: ", frame.shape)

        # 如果读取帧失败，则退出循环
        if not ret:
            print("无法接收帧（可能是摄像头断开）")
            break

        # driving_pts_coord 为（out_size，out_size）范围内的人脸坐标
        driving_img, driving_pts_coord = rgb_face_process(frame, out_size)

        drving_prompt = generate_prompt(driving_pts_coord, mode="texture", size=out_size,
                                        ref_image=source_img, ref_vt=source_crop_pts_vt)
        # cv2.imshow("a", source_img)
        # cv2.waitKey(-1)
        # cv2.imshow("a", drving_prompt)
        # cv2.waitKey(-1)
        driving_img_face = drving_prompt[:out_size, out_size:]
        drving_prompt = drving_prompt[:out_size, :out_size]

        tensor_source_img = torch.from_numpy(source_img[:, :, :4] / 255.).float().permute(2, 0, 1).unsqueeze(0).to(
            device)
        tensor_drving_img = torch.from_numpy(source_img[:, :, :4] / 255.).float().permute(2, 0, 1).unsqueeze(0).to(
            device)
        tensor_source_prompt = torch.from_numpy(source_prompt / 255.).float().permute(2, 0, 1).unsqueeze(0).to(
            device)
        tensor_drving_prompt = torch.from_numpy(drving_prompt / 255.).float().permute(2, 0, 1).unsqueeze(0).to(
            device)
        tensor_driving_img_face = torch.from_numpy(driving_img_face / 255.).float().permute(2, 0, 1).unsqueeze(
            0).to(device)
        with torch.no_grad():
            in0 = F.interpolate(tensor_source_img, size=(model_size_, model_size_), mode='nearest')
            in1 = F.interpolate(tensor_source_prompt, size=(model_size_, model_size_), mode='nearest')
            in2 = F.interpolate(tensor_drving_img, size=(model_size_, model_size_), mode='nearest')
            in3 = F.interpolate(tensor_driving_img_face, size=(model_size_, model_size_), mode='nearest')
            in4 = F.interpolate(tensor_drving_prompt, size=(model_size_, model_size_), mode='nearest')
            in5 = F.interpolate(tensor_bg_mask, size=(model_size_, model_size_), mode='nearest')
            # in6 = F.interpolate(tensor_wrap_mask, size=(model_size_, model_size_), mode='nearest')

            # print(in0.size(), in1.size(), in2.size(), in3.size(), in4.size(), in5.size())
            fake_out = face_interface(in0, in1, in2, in3, in4, in5, model_size_)

        in0 = Tensor2img(tensor_source_img[0], 0)
        # in1 = Tensor2img(tensor_source_prompt[0], 0)
        # in2 = Tensor2img(tensor_drving_prompt[0], 0)
        # in3 = Tensor2img(tensor_driving_img_face[0], 0)
        out = Tensor2img(fake_out[0], 0)
        frame = np.concatenate([in0, driving_img, out], axis=1)
        cv2.imshow("a", frame[..., ::-1])
        cv2.waitKey(50)

def main():
    # 检查命令行参数的数量
    if len(sys.argv) != 2:
        print("Usage: python interface/interface_face_rotation.py <img_path>")
        sys.exit(1)  # 参数数量不正确时退出程序

    img_path = sys.argv[1]
    print(f"img path is set to: {img_path}")
    run_avatar(img_path)

if __name__ == "__main__":
    main()