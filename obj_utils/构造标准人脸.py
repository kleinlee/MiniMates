import pickle
import numpy as np
import cv2
from talkingface.mediapipe_utils import detect_face_mesh
INDEX_FACE_EDGE = [
    454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150,
    136, 172, 58, 132, 93, 234,
    # 206, 426                 # 脸颊两个点，用来裁剪嘴部区域
    127,162,
    21, 54,103,67,109,10,338,297,332,284, 251,
    389,356,
]

def get_image(A_path, crop_coords, input_type, resize= 512):
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

def crop_(pts_array_origin, img_w, img_h):
    x_min, y_min, x_max, y_max = np.min(pts_array_origin[:, :, 0]), np.min(
        pts_array_origin[:, :, 1]), np.max(
        pts_array_origin[:, :, 0]), np.max(pts_array_origin[:, :, 1])
    new_w = (x_max - x_min) * 2.4
    new_h = (y_max - y_min) * 2.4
    center_x = (x_max + x_min) / 2.
    center_y = y_min + (y_max - y_min) * 0.2
    x_min, y_min, x_max, y_max = int(center_x - new_w / 2), int(center_y - new_h / 2), int(
        center_x + new_w / 2), int(center_y + new_h / 2)
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(x_max, img_w)
    y_max = min(y_max, img_h)
    new_size = min((x_max - x_min) / 2., (y_max - y_min) / 2.)
    center_x = (x_max + x_min) / 2.
    center_y = (y_max + y_min) / 2.
    x_min, y_min, x_max, y_max = int(center_x - new_size), int(center_y - new_size), int(
        center_x + new_size), int(center_y + new_size)
    return np.array([x_min, y_min, x_max, y_max])
out_size = 1000
# img_path = r"F:\C\AI\CV\TalkingFace\preparation\tiktok_zhongguodianyingbaodao3\2022-01-25-16-15-33-2\image2/0015.png"
# source_image = cv2.imread(img_path)[:, :, ::-1]
# face_pts_mean = detect_face_mesh([source_image[:, :, :3]])[0]
#
# (h, w) = source_image.shape[:2]
face_pts_mean = np.loadtxt("face_pts_mean.txt")
face_pts_mean = face_pts_mean + np.array([1000,1000,0])
w, h = (out_size, out_size)
source_crop_coords = crop_(face_pts_mean[np.newaxis, :, :], 10000, 10000)
face_pts_mean = get_image(face_pts_mean, source_crop_coords, input_type='mediapipe', resize=out_size)
face_pts_mean[:,2] -= np.mean(face_pts_mean[:, 2]) + (source_crop_coords[2] - source_crop_coords[0])/10
print(source_crop_coords, face_pts_mean)
w,h = (out_size, out_size)
def addUnicorn(verts):
    vert_unicorn = np.zeros([3, 3])
    vert_unicorn[0] = verts[103]
    vert_unicorn[1] = verts[332]
    vert_unicorn[2] = (verts[234] + verts[454])/2. + (verts[10] - verts[152])*0.8
    # 332 103 478
    # print(verts.shape, vert_unicorn.shape)
    verts2 = np.concatenate([verts, vert_unicorn], axis = 0)
    return verts2
vertices = addUnicorn(face_pts_mean)

with open("mediapipe_UV.pkl", "rb") as f:
    vt_, face = pickle.load(f)
face.extend([468,469,470])
face.extend([468,470,471])
face.extend([468,471,472])
face.extend([468,472,469])
face.extend([473,474,475])
face.extend([473,475,476])
face.extend([473,476,477])
face.extend([473,477,474])
face.extend([478, 479, 480])
INDEX_LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
INDEX_RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
ss = INDEX_LEFT_EYE + INDEX_RIGHT_EYE
face_ = []
for i in range(len(face)//3):
    tmp = face[3*i:3*i + 3]
    if not (tmp[0] in ss and tmp[1] in ss and tmp[2] in ss):
        face_.extend(tmp)
face = face_
vertices = vertices.flatten().tolist()
# vertices = vertices.flatten().tolist()
face_verts_num = 478 + 3

with open("face3D2.obj", "w") as f:
    for i in range(len(vertices)//3):
        f.write("v {:.3f} {:.3f} {:.3f}\n".format(vertices[3*i], vertices[3*i+1],vertices[3*i+2]))
    for i in range(468):
        f.write("vt {:.3f} {:.3f}\n".format(vt_[2*i], vt_[2*i+1]))
    for i in range(468, face_verts_num):
        f.write("vt 0 0\n")
    for i in range(len(face)//3):
        f.write("f {0}/{0} {1}/{1} {2}/{2}\n".format(face[3*i]+1, face[3*i+1]+1,face[3*i+2]+1))