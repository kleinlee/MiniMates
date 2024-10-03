import uuid
import numpy as np
import cv2
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
def readObjFile(filepath):
    with_vn = False
    with_vt = False
    v_ = []
    vt = []
    vn = []
    face = []
    with open(filepath) as f:
        # with open(r"face3D.obj") as f:
        content = f.readlines()
    for i in content:
        if i[:2] == "v ":
            v0,v1,v2 = i[2:-1].split(" ")
            v_.append(float(v0))
            v_.append(float(v1))
            v_.append(float(v2))
        if i[:3] == "vt ":
            with_vt = True
            vt0,vt1 = i[3:-1].split(" ")
            vt.append(float(vt0))
            vt.append(float(vt1))
        if i[:3] == "vn ":
            with_vn = True
            vn0,vn1,vn2 = i[3:-1].split(" ")
            vn.append(float(vn0))
            vn.append(float(vn1))
            vn.append(float(vn2))
        if i[:2] == "f ":
            tmp = i[2:-1].split(" ")
            for ii in tmp:
                a = ii.split("/")[0]
                a = int(a) - 1
                face.append(a)
    if not with_vn:
        vn = [0 for i in v_]
    if not with_vt:
        vt = [0 for i in range(len(v_)//3*2)]
    return v_, vt, vn, face

def writeObjFile(filepath, verts, face, vt = None):
    with open(filepath, "w") as f:
        for i in range(len(verts)//3):
            f.write("v {:.1f} {:.1f} {:.1f}\n".format(verts[3*i], verts[3*i+1], verts[3*i+2]))
        if vt is not None:
            for i in range(len(vt) // 2):
                f.write("vt {:.4f} {:.4f}\n".format(vt[2 * i], vt[2 * i + 1]))
            for i in range(len(face) // 3):
                f.write("f {0}/{0} {1}/{1} {2}/{2}\n".format(face[3 * i] + 1, face[3 * i + 1] + 1, face[3 * i + 2] + 1))
        else:
            for i in range(len(face)//3):
                f.write("f {} {} {}\n".format(face[3*i]+1, face[3*i+1]+1, face[3*i+2]+1))

def writeObjFile2(filepath, render_verts, render_face):
    num__ = 11
    with open(filepath, "w") as f:
        for i in range(len(render_verts)//num__):
            f.write("v {:.1f} {:.1f} {:.1f} {:.4f} {:.4f} {:.1f} {:.1f} {:.2f} {:.2f} {:.2f} {:.2f}\n".format(
                render_verts[num__ * i + 0], render_verts[num__ * i + 1], render_verts[num__ * i + 2],
                render_verts[num__ * i + 3], render_verts[num__ * i + 4], render_verts[num__ * i + 5], render_verts[num__ * i + 6],
                render_verts[num__ * i + 7], render_verts[num__ * i + 8], render_verts[num__ * i + 9], render_verts[num__ * i + 10]
            ))
        for i in range(len(render_face) // 3):
            f.write("f {} {} {}\n".format(render_face[3 * i] + 1, render_face[3 * i + 1] + 1, render_face[3 * i + 2] + 1))

def adjust_verts(verts):
    # 脸眉心10  鼻尖4  左侧21   右侧251
    tmp = (verts[10] - verts[4]) * 0.32
    vert_index_ = [21, 54,103,67,109,10,338,297,332,284, 251]
    verts[vert_index_] = verts[vert_index_] + tmp
    return verts

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

def get_image(A_path, crop_coords, input_type, resize= 512):
    (x_min, y_min, x_max, y_max) = crop_coords
    size = x_max - x_min

    if input_type == 'mediapipe':
        pose_pts = (A_path - np.array([x_min, y_min, 0])) * resize / size
        pose_pts[:,2] = pose_pts[:,2] - np.max(pose_pts[:,2])
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

def generateRenderInfo():
    v_face, vt_face, vn_face, face_face = readObjFile(os.path.join(current_dir,"../obj_utils/face3D.obj"))
    v_teeth, vt_teeth, vn_teeth, face_teeth = readObjFile(os.path.join(current_dir,"../obj_utils/modified_teeth_upper.obj"))
    v_teeth2, vt_teeth2, vn_teeth2, face_teeth2 = readObjFile(os.path.join(current_dir,"../obj_utils/modified_teeth_lower.obj"))
    print(len(v_face)//3, len(vt_face), len(vn_face), len(face_face))
    print(len(v_teeth)//3, len(vt_teeth), len(vn_teeth), len(face_teeth))
    print(len(v_teeth2)//3, len(vt_teeth2), len(vn_teeth2), len(face_teeth2))
    print(len(v_face)//3 + len(v_teeth)//3 + len(v_teeth2)//3)
    # 17504
    # exit(-1)

    v_, vt, vn, face = (
        v_face + v_teeth + v_teeth2, vt_face + vt_teeth + vt_teeth2, vn_face + vn_teeth + vn_teeth2,
        face_face + [i + len(v_face)//3 for i in face_teeth] + [i + len(v_face)//3 + len(v_teeth)//3 for i in face_teeth2])
    v_ = np.array(v_).reshape(-1, 3)

    # v_[:, 1] = -v_[:, 1]

    # 0-2: verts   3-4: t  5:whether to rotate 6: index
    vertices = np.zeros([len(v_), 7])
    # vertices = np.zeros([len(pts_array_), 6])

    vertices[:, :3] = v_
    vertices[:, 3:5] = np.array(vt).reshape(-1, 2)
    # vertices[:, 5] = 0
    # 脸部为0，脸部可旋转边缘为1，脸部不可旋转边缘为2，牙齿为3
    vertices[478:(len(v_face) // 3), 5] = 1.
    vertices[len(v_face) // 3:, 5] = 3.
    # vertices[-18:, 5] = 4.

    vertices[:, 6] = list(range(len(v_)))
    return vertices, face

def normalizeFaceAndTeeth(face_pts, face_pts_mean, pca, number_verts_lower_teeth = 1121, number_verts_upper_teeth = 988):
    # teeth_verts[:, 1] = teeth_verts[:, 1] + (
    #         np.mean(face_pts_mean_personal[main_keypoints_index][INDEX_LIPS_INNER[11:], 1]) - np.mean(
    #     teeth_verts[9:27, 1])) + 2

    face_pts*0.33 + face_pts_mean*0.66

    return face_pts

def translation_matrix(point):
    """生成平移矩阵"""
    return np.array([
        [1, 0, 0, point[0]],
        [0, 1, 0, point[1]],
        [0, 0, 1, point[2]],
        [0, 0, 0, 1]
    ])
def rotate_around_point(point, theta, phi, psi):
    """围绕点P旋转"""
    # 将点P平移到原点
    T1 = translation_matrix(-point)

    # 定义欧拉角
    theta = np.radians(theta)  # 俯仰角
    phi = np.radians(phi)  # 偏航角
    psi = np.radians(psi)  # 翻滚角

    # 创建旋转矩阵
    tmp = [theta, phi, psi]
    matX = np.array([[1.0,            0,               0,               0],
                     [0.0,            np.cos(tmp[0]), -np.sin(tmp[0]),  0],
                     [0.0,            np.sin(tmp[0]),  np.cos(tmp[0]),  0],
                     [0,              0,               0,               1]
                     ])
    matY = np.array([[np.cos(tmp[1]), 0,               np.sin(tmp[1]),  0],
                     [0,              1,               0,               0],
                     [-np.sin(tmp[1]),0,               np.cos(tmp[1]),  0],
                     [0,              0,               0,               1]
                     ])
    matZ = np.array([[np.cos(tmp[2]), -np.sin(tmp[2]), 0,               0],
                     [np.sin(tmp[2]), np.cos(tmp[2]),  0,               0],
                     [0,              0,               1,               0],
                     [0,              0,               0,               1]
                     ])

    R = matZ @ matY @ matX

    # 将点P移回其原始位置
    T2 = translation_matrix(point)

    # 总的变换矩阵
    total_transform = T2 @ R @ T1

    return total_transform

def rodrigues_rotation_formula(axis, theta):
    """Calculate the rotation matrix using Rodrigues' rotation formula."""
    axis = np.asarray(axis) / np.linalg.norm(axis)  # Normalize the axis
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + sin_theta * K + (1 - cos_theta) * np.dot(K, K)
    return R
def RotateAngle2Matrix(center, axis, theta):
    """Rotate around a center point."""
    # Step 1: Translate the center to the origin
    translation_to_origin = np.eye(4)
    translation_to_origin[:3, 3] = -center

    # Step 2: Apply the rotation
    R = rodrigues_rotation_formula(axis, theta)
    R_ = np.eye(4)
    R_[:3,:3] = R
    R = R_

    # Step 3: Translate back to the original position
    translation_back = np.eye(4)
    translation_back[:3, 3] = center

    # Combine the transformations
    rotation_matrix = translation_back @ R @ translation_to_origin

    return rotation_matrix

# def RotateAngle2Matrix(center, axis, theta):
#         # 将axis向量标准化
#         axis = np.asarray(axis)
#         axis = axis / np.sqrt(np.dot(axis, axis))
#
#         # 使用罗德里格斯公式计算旋转矩阵
#         a = np.cos(theta / 2.0)
#         b, c, d = -axis * np.sin(theta / 2.0)
#         aa, bb, cc, dd = a * a, b * b, c * c, d * d
#         bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
#
#         # 构建旋转矩阵
#         rotation_matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), 0],
#                                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), 0],
#                                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, 0],
#                                     [0, 0, 0, 1]])
#
#         # 构建平移矩阵
#         translation_matrix = np.eye(4)
#         translation_matrix[:3, 3] = -np.array(center)
#
#         # 构建逆平移矩阵
#         inverse_translation_matrix = np.eye(4)
#         inverse_translation_matrix[:3, 3] = np.array(center)
#
#         # 结合平移矩阵、旋转矩阵和逆平移矩阵
#         combined_matrix = np.dot(inverse_translation_matrix, np.dot(rotation_matrix, translation_matrix))
#
#         return combined_matrix

INDEX_FLAME_LIPS = [
1,26,23,21,8,155,83,96,98,101,
73,112,123,124,143,146,71,52,51,40,
2,25,24,22,7,156,82,97,99,100,
74,113,122,125,138,148,66,53,50,41,
30,31,32,38,39,157,111,110,106,105,
104,120,121,126,137,147,65,54,49,48,
4,28,33,20,19,153,94,95,107,103,
76,118,119,127,136,149,64,55,47,46,

3,27,35,17,18,154,93,92,109,102,
75,114,115,128,133,151,61,56,43,42,
6,29, 13, 12, 11, 158, 86, 87, 88, 79,
80,117, 116, 135, 134, 150, 62, 63, 44, 45,
5,36,14,9,10,159,85,84,89,78,
77,141,130,131,139,145,67,59,58,69,
0,37,34,15,16,152,91,90,108,81,72,
142,129,132,140,144,68,60,57,70,
]
INDEX_MP_LIPS = [
291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61,
146, 91, 181, 84, 17, 314, 405, 321, 375,
306, 408, 304, 303, 302, 11, 72, 73, 74, 184, 76,
77, 90, 180, 85, 16, 315, 404, 320, 307,
292, 407, 272, 271, 268, 12, 38, 41, 42, 183, 62,
96, 89, 179, 86, 15, 316, 403, 319, 325,
308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78,
95, 88, 178, 87, 14, 317, 402, 318, 324,
]
floor_ = 20
def faceModeling(source_crop_pts, render_verts, face_rect, w, h):
    INDEX_FACE_EDGE = [
        93, 234, 127, 162, 21,
        54, 103, 67, 109, 10, 338, 297, 332, 284, 251,
        389, 356,
        454, 323, 361, 288, 397, 365,
        379, 378, 400, 377, 152, 148, 176, 149, 150,
        136, 172, 58, 132
    ]
    square_edge = []
    edge_points = [5, 7, 5, 19]
    div_ = 6.5
    # square_edge = square_edge + [[0, (4 - i) / div_] for i in range(edge_points[0])]
    # square_edge = square_edge + [[(i + 1) / 8, 0] for i in range(edge_points[1])]
    # square_edge = square_edge + [[1, i / div_] for i in range(edge_points[2])]
    square_edge = [[0.013, 0.671],
                   [0.035, 0.533],
                   [0.070, 0.390],
                   [0.117, 0.267],
                   [0.183, 0.166],
                   [0.256, 0.122],
                   [0.333, 0.094],
                   [0.418, 0.080],
                   [0.500, 0.074],
                   [0.582, 0.080],
                   [0.667, 0.094],
                   [0.744, 0.122],
                   [0.817, 0.166],
                   [0.883, 0.267],
                   [0.930, 0.390],
                   [0.965, 0.533],
                   [0.987, 0.671],
                   ]
    square_edge = square_edge + [[1 - i/18, 1 - 1/6 * (i - 9)**2/9**2 - 0.03] for i in range(edge_points[3])]



    square_edge0 = np.zeros([len(INDEX_FACE_EDGE), 3])
    out_size = face_rect[2] - face_rect[0]
    square_edge0[:36, :2] = np.array(square_edge) * out_size + np.array([face_rect[0], face_rect[1]])

    pts_array_new = source_crop_pts.copy()
    for floor in range(1, floor_ + 1):
        square_edge1 = np.zeros_like(square_edge0)
        for i in range(len(INDEX_FACE_EDGE)):
            square_edge1[i] = square_edge0[i] * floor / floor_ + source_crop_pts[INDEX_FACE_EDGE[i]] * (floor_ - floor) / floor_

        pts_array_new = np.concatenate([pts_array_new, square_edge1], axis = 0)
        print(pts_array_new.shape)

    square_edge1 = np.array([[0, out_size, 0],
              [out_size, out_size, 0]])
    pts_array_new = np.concatenate([pts_array_new, square_edge1], axis=0)


    render_verts[:478 + 36 * floor_ + 2, :3] = pts_array_new

    # render_verts[:478 + 80 + 36 * 3, 3:5] = render_verts[:478 + 80 + 36 * 3, :2] / out_size
    render_verts[:478 + 36 * floor_ + 2, 3] = render_verts[:478 + 36 * floor_ + 2, 0] / w
    render_verts[:478 + 36 * floor_ + 2, 4] = render_verts[:478 + 36 * floor_ + 2, 1] / h

    render_verts[-2, :5] = np.array([0, h, 0, 0, 1])
    render_verts[-1, :5] = np.array([w, h, 0, 1, 1])
    # print(render_verts.shape)
    # exit(-1)
    return render_verts
from talkingface.mediapipe_utils import detect_face_mesh
render_verts_ = None
face_pts_mean = None
def NewFaceInfo(source_image, ):
    global render_verts_, face_pts_mean
    if render_verts_ is None:
        render_verts_, render_face = generateRenderInfo()
        face_pts_mean = render_verts_[:478, :3]
    render_verts = render_verts_.copy()
    source_pts = detect_face_mesh([source_image[:, :, :3]])[0]
    source_pts[:, 2] = source_pts[:, 2] - np.max(source_pts[:, 2])
    # source_pts = adjust_verts(source_pts)
    print(source_pts.shape)
    # source_image,source_crop_pts = rand_crop_and_resize(source_image, source_pts, crop_rotio = [0.4,0.4,0.4,0.4])
    source_image, source_crop_pts = rand_crop_and_resize(source_image, source_pts, crop_rotio=[0.5, 0.5, 0.65, 1.35])

    face_rect = crop_(source_crop_pts[np.newaxis, :, :], 10000, 10000)

    (h, w) = source_image.shape[:2]
    render_verts = faceModeling(source_crop_pts, render_verts, face_rect, w, h)
    # # print(render_verts[-4:])
    # # exit()
    point_size = 1
    point_color = (0, 0, 255)  # BGR
    thickness = 1  # 0 、4、8
    source_image2 = source_image.copy()
    # for coor in render_verts:
    #     cv2.circle(source_image2, (int(coor[0]), int(coor[1])), point_size, point_color, thickness)
    INDEX_LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,409,270,269,267,0,37,39,40,185]
    INDEX_LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
    INDEX_LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
    INDEX_RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    INDEX_LEFT_IRIS = [474, 475, 476, 477, 474]
    INDEX_RIGHT_IRIS = [469, 470, 471, 472, 469, 469]
    for ii in range(len(INDEX_LIPS_OUTER)):
        pt1 = [int(flt) for flt in render_verts[INDEX_LIPS_OUTER[ii]]][:2]
        pt2 = [int(flt) for flt in render_verts[INDEX_LIPS_OUTER[(ii + 1) % len(INDEX_LIPS_OUTER)]]][:2]
        cv2.line(source_image2, tuple(pt1), tuple(pt2), (255, 0, 0), 1)
    for ii in range(len(INDEX_LIPS_INNER)):
        pt1 = [int(flt) for flt in render_verts[INDEX_LIPS_INNER[ii]]][:2]
        pt2 = [int(flt) for flt in render_verts[INDEX_LIPS_INNER[(ii + 1) % len(INDEX_LIPS_INNER)]]][:2]
        cv2.line(source_image2, tuple(pt1), tuple(pt2), (255, 0, 0), 1)
    for ii in range(len(INDEX_LEFT_EYE)):
        pt1 = [int(flt) for flt in render_verts[INDEX_LEFT_EYE[ii]]][:2]
        pt2 = [int(flt) for flt in render_verts[INDEX_LEFT_EYE[(ii + 1) % len(INDEX_LEFT_EYE)]]][:2]
        cv2.line(source_image2, tuple(pt1), tuple(pt2), (255, 0, 0), 1)
    for ii in range(len(INDEX_RIGHT_EYE)):
        pt1 = [int(flt) for flt in render_verts[INDEX_RIGHT_EYE[ii]]][:2]
        pt2 = [int(flt) for flt in render_verts[INDEX_RIGHT_EYE[(ii + 1) % len(INDEX_RIGHT_EYE)]]][:2]
        cv2.line(source_image2, tuple(pt1), tuple(pt2), (255, 0, 0), 1)
    for ii in range(len(INDEX_LEFT_IRIS)):
        pt1 = [int(flt) for flt in render_verts[INDEX_LEFT_IRIS[ii]]][:2]
        pt2 = [int(flt) for flt in render_verts[INDEX_LEFT_IRIS[(ii + 1) % len(INDEX_LEFT_IRIS)]]][:2]
        cv2.line(source_image2, tuple(pt1), tuple(pt2), (255, 0, 0), 1)
    for ii in range(len(INDEX_RIGHT_IRIS)):
        pt1 = [int(flt) for flt in render_verts[INDEX_RIGHT_IRIS[ii]]][:2]
        pt2 = [int(flt) for flt in render_verts[INDEX_RIGHT_IRIS[(ii + 1) % len(INDEX_RIGHT_IRIS)]]][:2]
        cv2.line(source_image2, tuple(pt1), tuple(pt2), (255, 0, 0), 1)
    cv2.imwrite("output/" + str(uuid.uuid1()) + ".png", source_image2)
    # cv2.imshow("a", source_image)
    # cv2.waitKey(-1)
    # continue
    # print("0000000")
    # 找到标准人脸，求出旋转矩阵
    from talkingface.run_utils import calc_face_mat
    mat_list, _, face_pts_mean_personal_primer = calc_face_mat(source_crop_pts[np.newaxis, :478, :],
                                                               face_pts_mean)
    # print(face_pts_mean_personal_primer.shape)
    mat_list__ = mat_list[0]
    # 标准人脸旋转回去找到标准嘴巴部分的顶点
    mouth_pts = face_pts_mean[INDEX_MP_LIPS] * 0.4 + face_pts_mean_personal_primer[INDEX_MP_LIPS] * 0.6
    # mouth_pts = np.zeros([80, 3])
    # # mouth_pts[-20:] = face_pts_mean[INDEX_MP_LIPS][-20:] * 0.8 + face_pts_mean_personal_primer[INDEX_MP_LIPS][-20:] * 0.2
    # # mouth_pts[-40:-20] = mouth_pts[-20:] + (face_pts_mean[INDEX_MP_LIPS][-40:-20] - face_pts_mean[INDEX_MP_LIPS][-20:]) * 0.6 + (face_pts_mean_personal_primer[INDEX_MP_LIPS][-40:-20] - face_pts_mean_personal_primer[INDEX_MP_LIPS][-20:]) * 0.4
    # # mouth_pts[-60:-40] = mouth_pts[-40:-20] + (face_pts_mean[INDEX_MP_LIPS][-60:-40] - face_pts_mean[INDEX_MP_LIPS][-40:-20]) * 0.6 + (face_pts_mean_personal_primer[INDEX_MP_LIPS][-60:-40] - face_pts_mean_personal_primer[INDEX_MP_LIPS][-40:-20]) * 0.4
    # # mouth_pts[-80:-60] = mouth_pts[-60:-40] + (face_pts_mean[INDEX_MP_LIPS][-80:-60] - face_pts_mean[INDEX_MP_LIPS][-60:-40]) * 0.6 + (face_pts_mean_personal_primer[INDEX_MP_LIPS][-80:-60] - face_pts_mean_personal_primer[INDEX_MP_LIPS][-60:-40]) * 0.4
    # mouth_pts[-20:] = face_pts_mean[INDEX_MP_LIPS][-20:] * 0.8 + face_pts_mean_personal_primer[INDEX_MP_LIPS][-20:] * 0.2
    # mouth_pts[-40:-20] = mouth_pts[-20:] + (face_pts_mean_personal_primer[INDEX_MP_LIPS][-40:-20] - face_pts_mean_personal_primer[INDEX_MP_LIPS][-20:]) * 1
    # mouth_pts[-60:-40] = mouth_pts[-40:-20] + (face_pts_mean_personal_primer[INDEX_MP_LIPS][-60:-40] - face_pts_mean_personal_primer[INDEX_MP_LIPS][-40:-20]) * 1
    # mouth_pts[-80:-60] = mouth_pts[-60:-40] + (face_pts_mean_personal_primer[INDEX_MP_LIPS][-80:-60] - face_pts_mean_personal_primer[INDEX_MP_LIPS][-60:-40]) * 1

    start_index = 0
    num_ = len(mouth_pts)
    keypoints = np.ones([4, num_])
    keypoints[:3, :] = mouth_pts.T
    keypoints = mat_list__.dot(keypoints).T
    source_crop_pts_obj = source_crop_pts.copy()
    source_crop_pts_obj[INDEX_MP_LIPS] = keypoints[:, :3]

    render_verts[INDEX_MP_LIPS, :3] = source_crop_pts_obj[INDEX_MP_LIPS]

    # print("0000001")

    # 嘴巴=face_pts_mean*0.5 + face_pts_mean*0.5

    #  牙齿部分旋转
    start_index = 478 + 36 * 20 + 2
    num_ = len(render_verts) - start_index - 2
    keypoints = np.ones([4, num_])
    keypoints[:3, :] = render_verts[start_index:start_index + num_, :3].T
    keypoints = mat_list__.dot(keypoints).T
    render_verts[start_index:start_index + num_, :3] = keypoints[:, :3]
    return render_verts,render_face,source_image

def rand_crop_and_resize(img, kp2d, crop_rotio = [0.5,0.5,0.65,1.35]):
    h = img.shape[0]
    w = img.shape[1]
    # random_crop and resize
    x2d = kp2d[:, 0]
    y2d = kp2d[:, 1]
    w_span = x2d.max() - x2d.min()
    h_span = y2d.max() - y2d.min()
    crop_size = int(2*max(h_span, w_span))
    center_x = (x2d.max() + x2d.min()) / 2.
    center_y = (y2d.max() + y2d.min()) / 2.
    # 确定裁剪区域上边top和左边left坐标，中心点是(x2d.max() + x2d.min()/2, y2d.max() + y2d.min()/2)
    top = int(center_y - crop_size*crop_rotio[2])
    left = int(center_x - crop_size*crop_rotio[0])
    # 裁剪区域与原图的重合区域
    top_coincidence = int(max(top, 0))
    bottom_coincidence = int(min(top + crop_size*(crop_rotio[2] + crop_rotio[3]), h))
    left_coincidence = int(max(left, 0))
    right_coincidence = int(min(left + crop_size*(crop_rotio[0] + crop_rotio[1]), w))
    # img_new = np.zeros([int(crop_size*2), crop_size, 3], dtype=np.uint8)
    img_new = np.zeros([int(crop_size*(crop_rotio[2] + crop_rotio[3])), int(crop_size*(crop_rotio[0] + crop_rotio[1])), 3], dtype=np.uint8)
    print(top_coincidence - top,bottom_coincidence - top, left_coincidence - left,right_coincidence - left)
    # print(top_coincidence, bottom_coincidence, left_coincidence, right_coincidence)
    # print(img[
    #       top_coincidence:bottom_coincidence,
    #       left_coincidence:right_coincidence,
    #       :].shape)
    # print(img_new[top_coincidence - top:bottom_coincidence - top, left_coincidence - left:right_coincidence - left, :].shape)
    print(img_new.shape)
    img_new[top_coincidence - top:bottom_coincidence - top, left_coincidence - left:right_coincidence - left, :] = img[
                                                                                                                   top_coincidence:bottom_coincidence,
                                                                                                                   left_coincidence:right_coincidence,
                                                                                                                   :]
    out_size = 500
    img_new = cv2.resize(img_new, (out_size, out_size*2))

    factor = out_size*2/(crop_size*(crop_rotio[0] + crop_rotio[1]))
    kp2d[:, 0] = (kp2d[:, 0] - left) * factor
    kp2d[:, 1] = (kp2d[:, 1] - top) * factor
    kp2d[:, 2] = (kp2d[:, 2] - 0) * factor
    return img_new, kp2d

def crop_mouth(mouth_pts, mat_list__):
    """
    x_ratio: 裁剪出一个正方形，边长根据keypoints的宽度 * x_ratio决定
    """
    num_ = len(mouth_pts)
    keypoints = np.ones([4, num_])
    keypoints[:3, :] = mouth_pts.T
    keypoints = mat_list__.dot(keypoints).T
    keypoints = keypoints[:, :3]

    x_min, y_min, x_max, y_max = np.min(keypoints[:, 0]), np.min(keypoints[:, 1]), np.max(keypoints[:, 0]), np.max(keypoints[:, 1])
    border_width_half = max(x_max - x_min, y_max - y_min) * 0.66
    y_min = y_min + border_width_half * 0.3
    center_x = (x_min + x_max) /2.
    center_y = (y_min + y_max) /2.
    x_min, y_min, x_max, y_max = int(center_x - border_width_half), int(center_y - border_width_half*0.75), int(
        center_x + border_width_half), int(center_y + border_width_half*0.75)
    print([x_min, y_min, x_max, y_max])

    # pts = np.array([
    #     [x_min, y_min],
    #     [x_max, y_min],
    #     [x_max, y_max],
    #     [x_min, y_max]
    # ])
    return [x_min, y_min, x_max, y_max]

def drawMouth(keypoints, source_texture, out_size = (700, 1400)):
    INDEX_LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
    INDEX_LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, ]
    INDEX_LIPS_LOWWER = INDEX_LIPS_INNER[:11] + INDEX_LIPS_OUTER[:11][::-1]
    INDEX_LIPS_UPPER = INDEX_LIPS_INNER[10:] + [INDEX_LIPS_INNER[0], INDEX_LIPS_OUTER[0]] + INDEX_LIPS_OUTER[10:][::-1]
    INDEX_LIPS = INDEX_LIPS_INNER + INDEX_LIPS_OUTER
    # keypoints = keypoints[INDEX_LIPS]
    keypoints[:, 0] = keypoints[:, 0] * out_size[0]
    keypoints[:, 1] = keypoints[:, 1] * out_size[1]
    # pts = keypoints[20:40]
    # pts = pts.reshape((-1, 1, 2)).astype(np.int32)
    # cv2.fillPoly(source_texture, [pts], color=(255, 0, 0,))
    # pts = keypoints[:20]
    # pts = pts.reshape((-1, 1, 2)).astype(np.int32)
    # cv2.fillPoly(source_texture, [pts], color=(0, 0, 0,))

    pts = keypoints[INDEX_LIPS_OUTER]
    pts = pts.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(source_texture, [pts], color=(0, 0, 0))
    pts = keypoints[INDEX_LIPS_UPPER]
    pts = pts.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(source_texture, [pts], color=(255, 0, 0))
    pts = keypoints[INDEX_LIPS_LOWWER]
    pts = pts.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(source_texture, [pts], color=(127, 0, 0))

    prompt_texture = np.zeros_like(source_texture)
    pts = keypoints[INDEX_LIPS_UPPER]
    pts = pts.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(prompt_texture, [pts], color=(255, 0, 0))
    pts = keypoints[INDEX_LIPS_LOWWER]
    pts = pts.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(prompt_texture, [pts], color=(127, 0, 0))
    return source_texture, prompt_texture
