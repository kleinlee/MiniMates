import os
import torch.nn.functional as F
os.environ["KMP_DUPLICATE_LIB_OK"] = "true"
current_dir = os.path.dirname(os.path.abspath(__file__))
import cv2
import numpy as np
import kaldi_native_fbank as knf
from scipy.io import wavfile
import torch
import os
from talkingface.mediapipe_utils import detect_face_mesh
from train.data.dataset_faceReenactment3 import crop_,get_image,adjust_verts,get_standard_image
device = "cuda" if torch.cuda.is_available() else "cpu"
FaceModel = None
Audio2FeatureModel = None
PcaModel = None
def generate_bg_mask(out_size):
    bg_mask = cv2.imread(current_dir + "/../obj_utils/bg_mask.png")
    bg_mask[np.where(bg_mask < 128)] = 0
    bg_mask = cv2.resize(bg_mask, (out_size, out_size))
    bg_mask = (bg_mask > 0).astype(int)
    bg_mask = np.concatenate([bg_mask[:, :, 0:1], bg_mask[:, :, 0:1], bg_mask[:, :, 0:1], bg_mask[:, :, 0:1]], axis=2)
    return bg_mask

def rotate(image, angle, trans, center=None, scale=1.0):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        # center = (w // 2, h // 2)
        center = (w // 2, 1.3*h)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[:, 2] += trans
    rotated = cv2.warpAffine(image, M, (w, h))

    # return the rotated image
    return rotated
from math import cos,sin,radians
def eularAngle2Matrix(tmp):   #tmp为xyz的旋转角,角度值
    tmp = [radians(i) for i in tmp]
    matX = np.array([[1.0,          0,            0],
                     [0.0,          cos(tmp[0]), -sin(tmp[0])],
                     [0.0,          sin(tmp[0]),  cos(tmp[0])]])
    matY = np.array([[cos(tmp[1]),  0,            sin(tmp[1])],
                     [0.0, 1, 0],
                     [-sin(tmp[1]),  0,            cos(tmp[1])]])
    matZ = np.array([[cos(tmp[2]), -sin(tmp[2]),  0],
                     [sin(tmp[2]),  cos(tmp[2]),  0],
                     [0, 0, 1]])
    matRotate = np.matmul(matZ, matY)
    matRotate = np.matmul(matRotate, matX)
    return matRotate
def face_interface(tensor_source_img, tensor_source_prompt, tensor_drving_img,
                   tensor_driving_img_face, tensor_drving_prompt, tensor_bg_mask, model_size):
    global FaceModel
    if FaceModel is None:
        from models.FreeFace import FreeFace as FreeFaceModel
        FaceModel = FreeFaceModel(8, 8, model_size, is_train=False).to(device)  #
        current_dir = os.path.dirname(os.path.abspath(__file__))
        inference_model_path = os.path.join(current_dir, "../checkpoint/FreeFace/epoch_40.pth")
        state_dict = torch.load(inference_model_path, map_location=device)['state_dict']['net_g']
        FaceModel.load_state_dict(state_dict)
        FaceModel.eval()
    in0 = F.interpolate(tensor_source_img, size=(model_size, model_size), mode='nearest')
    in1 = F.interpolate(tensor_source_prompt, size=(model_size, model_size), mode='nearest')
    in2 = F.interpolate(tensor_drving_img, size=(model_size, model_size), mode='nearest')
    in3 = F.interpolate(tensor_driving_img_face, size=(model_size, model_size), mode='nearest')
    in4 = F.interpolate(tensor_drving_prompt, size=(model_size, model_size), mode='nearest')
    in5 = F.interpolate(tensor_bg_mask, size=(model_size, model_size), mode='nearest')
    fake_out = FaceModel(in0, in1, in2, in3, in4, in5)
    return fake_out

def face_process(img_path, out_size):
    img_primer_rgba = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    assert img_primer_rgba.shape[2] == 4
    source_pts = detect_face_mesh([img_primer_rgba[:, :, :3]])[0]
    source_pts[:, 2] = source_pts[:, 2] - np.max(source_pts[:, 2])
    img_primer_rgba = cv2.cvtColor(img_primer_rgba, cv2.COLOR_BGRA2RGBA)
    (h, w) = img_primer_rgba.shape[:2]
    source_crop_rect = crop_(source_pts[np.newaxis, :, :], w, h)
    # source_img = get_image(img_primer_rgba, source_crop_coords, input_type='img', resize=out_size)
    # source_crop_pts = get_image(source_pts, source_crop_coords, input_type='mediapipe', resize=out_size)
    source_img, source_crop_pts = get_standard_image(img_primer_rgba, source_pts, source_crop_rect, resize=out_size)
    source_crop_pts = adjust_verts(source_crop_pts)
    source_crop_pts_vt = source_crop_pts.copy()
    source_crop_pts_vt = source_crop_pts_vt[:,:2]/ out_size
    return img_primer_rgba, source_img, source_crop_pts, source_crop_pts_vt,source_crop_rect

def rgb_face_process(img_primer_bgr, out_size):
    source_pts = detect_face_mesh([img_primer_bgr[:, :, :3]])[0]
    source_pts[:, 2] = source_pts[:, 2] - np.max(source_pts[:, 2])
    img_primer_rgb = cv2.cvtColor(img_primer_bgr, cv2.COLOR_BGR2RGB)
    (h, w) = img_primer_rgb.shape[:2]
    source_crop_coords = crop_(source_pts[np.newaxis, :, :], w, h)
    source_img, source_crop_pts = get_standard_image(img_primer_rgb, source_pts, source_crop_coords, resize=out_size)
    source_crop_pts = adjust_verts(source_crop_pts)
    return source_img, source_crop_pts

# def rgb_face_rotation_process(img_primer_bgr, out_size):
#     source_pts = detect_face_mesh([img_primer_bgr[:, :, :3]])[0]
#     source_pts[:, 2] = source_pts[:, 2] - np.max(source_pts[:, 2])
#     img_primer_rgb = cv2.cvtColor(img_primer_bgr, cv2.COLOR_BGR2RGB)
#     (h, w) = img_primer_rgb.shape[:2]
#     source_crop_coords = crop_(source_pts[np.newaxis, :, :], w, h)
#     source_img, source_crop_pts = get_standard_image(img_primer_rgb, source_pts, source_crop_coords, resize=out_size)
#     source_crop_pts = adjust_verts(source_crop_pts)
#
#     mat_list, _, face_pts_mean_personal_primer = calc_face_mat(pts_driven, face_pts_mean)
#     return source_img, source_crop_pts

# 读取音频文件
def load_audio(file_path):
    import librosa
    import numpy as np

    # 使用 librosa 读取音频文件
    # sr=None 表示不改变原始采样率，mono=True 表示转换为单声道
    y, sr = librosa.load(file_path, sr=None, mono=True)

    # 将采样率转换为 16kHz
    y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)

    # 确保数据类型为 float32
    y_16k = y_16k.astype(np.float32)
    return y_16k

def audio_interface(audio_path):
    global Audio2FeatureModel,PcaModel
    if Audio2FeatureModel is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt_path = os.path.join(current_dir, "../checkpoint/lstm/lstm_model_epoch_590.pth")
        # ckpt_path = r"F:\C\AI\CV\DH_few_shot\DH029_wav2lip_mediapipe\checkpoint\lstm\lstm_model_epoch_590.pth"
        from talkingface.models.audio2bs_lstm import Audio2Feature
        Audio2FeatureModel = Audio2Feature()  # 调用模型Model
        Audio2FeatureModel.load_state_dict(torch.load(ckpt_path))
        Audio2FeatureModel = Audio2FeatureModel.to(device)
        Audio2FeatureModel.eval()
    # rate, wav = wavfile.read(wavpath, mmap=False)
    #
    # augmented_samples = wav
    # augmented_samples2 = augmented_samples.astype(np.float32, order='C') / 32768.0
    augmented_samples2 = load_audio(audio_path)

    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.frame_length_ms = 50
    opts.frame_opts.frame_shift_ms = 20
    opts.mel_opts.num_bins = 80
    opts.frame_opts.snip_edges = False
    opts.mel_opts.debug_mel = False
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(16000, augmented_samples2.tolist())
    seq_len = fbank.num_frames_ready // 2
    A2Lsamples = np.zeros([2 * seq_len, 80])
    for i in range(2 * seq_len):
        # print(i)
        f2 = fbank.get_frame(i)
        A2Lsamples[i] = f2

    orig_mel = A2Lsamples
    input = torch.from_numpy(orig_mel).unsqueeze(0).float().to(device)
    h0 = torch.zeros(2, 1, 192).to(device)
    c0 = torch.zeros(2, 1, 192).to(device)
    bs_array, hn, cn = Audio2FeatureModel(input, h0, c0)
    # print(bs_array.shape)
    bs_array = bs_array[0, 4:].detach().cpu().float().numpy()

    if PcaModel is None:
        ckpt_path = os.path.join(current_dir, "../checkpoint/pca_mediapipe.pkl")
        import pickle
        with open(ckpt_path, "rb") as f:
            PcaModel = pickle.load(f)

    frame_num = len(bs_array)
    output = np.zeros([frame_num, 478, 3])
    for frame_index in range(frame_num):
        # pts = np.dot(bs_array[frame_index, :6], PcaModel.components_[:6]) + PcaModel.mean_
        pts = np.dot(bs_array[frame_index, :6], PcaModel.components_[:6]) * 0.7
        print(frame_index, pts.shape)
        pts = pts.reshape(-1, 3)
        pts[:, 0] *= 0.7
        output[frame_index] = pts

        # point_size = 1
        # point_color = (0, 0, 255)  # BGR
        # thickness = 4  # 0 、4、8
        #
        # img = np.zeros([1000, 1000, 3])
        # cv2.line(img, (0, 780), (1000, 780), point_color, 1)
        # for coor in pts:
        #     # coor = (coor +1 )/2.
        #     cv2.circle(img, (int(coor[0]), int(coor[1])), point_size, point_color, thickness)
        # cv2.imshow("a", img)
        # cv2.waitKey(40)

    return output

import math
def rotation_matrix_to_euler_angles(R):
    # 检查矩阵是否为3x3
    assert R.shape == (3, 3), "Rotation matrix must be 3x3"

    U, S, Vt = np.linalg.svd(R)

    # 2. 构造正交矩阵 W
    W = U @ Vt

    # 3. 调整行列式
    det_W = np.linalg.det(W)
    if det_W < 0:
        # 交换最后一列
        W[:, -1] *= -1
    # print(det_W, S)
    #
    # print(R, U @ Vt*np.mean(S))

    R = W
    # 计算欧拉角
    sy = math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)

    singular = sy < 1e-6  # 当sy接近0时，可能会出现除零错误

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    # 将弧度转换为角度
    x = math.degrees(x)
    y = math.degrees(y)
    z = math.degrees(z)
    # print([x, y, z])
    # exit()
    return [x, y, z]
