import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import time
import pickle
import cv2
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

vertex_src = """
# version 330

layout(location = 0) in vec2 a_position;
layout(location = 1) in vec3 a_texture;
out vec3 v_TexCoord;

void main()
{
    gl_Position =  vec4(a_position, 0.0, 1.0);
    v_TexCoord = a_texture;

    //v_texture = 1 - a_texture;                      // Flips the texture vertically and horizontally
    //v_texture = vec2(a_texture.s, 1 - a_texture.t); // Flips the texture vertically
}
"""

fragment_src = """
# version 330

in vec3 v_TexCoord;

out vec4 fragColor;

uniform sampler2D texture0;
uniform sampler2D texture1;
void main()
{
    if (v_TexCoord.z < 0.5)
    {
        fragColor = texture(texture0, v_TexCoord.xy);
    }
    else if (v_TexCoord.z < 1.5)
    {
        fragColor = vec4(0.0, 0.0, 1.0, 1.0);
    }
    else if (v_TexCoord.z < 2.5)
    {
        fragColor = vec4(0.0, 1.0, 0.0, 1.0);
    }
    else
    {
        fragColor = vec4(0.0, 0.0, 1.0, 1.0);
    }
}
"""


# glfw callback functions
def window_resize(window, width, height):
    glViewport(0, 0, width, height)


# initializing glfw library
if not glfw.init():
    raise Exception("glfw can not be initialized!")

# creating the window
window_size = 384
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
window = glfw.create_window(window_size, window_size, "My OpenGL window", None, None)

# check if window was created
if not window:
    glfw.terminate()
    raise Exception("glfw window can not be created!")

# set window's position
glfw.set_window_pos(window, 100, 100)
# set the callback function for window resize
glfw.set_window_size_callback(window, window_resize)

# make the context current
glfw.make_context_current(window)

from obj_utils.utils import generateRenderInfo
render_verts_, render_face_ = generateRenderInfo()
vt_ = render_verts_[:, 3:5].flatten().tolist()
vertices = np.zeros([len(render_verts_), 5], dtype=np.float32).flatten()
indices = np.array(render_face_, dtype=np.uint32)


shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))

# Step2: 创建并绑定VBO 对象 传送数据
VBO = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

# Step3: 创建并绑定EBO 对象 传送数据
EBO = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
# Step4: 指定解析方式  并启用顶点属性
# 顶点位置属性
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, vertices.itemsize * 5, ctypes.c_void_p(0))
# 顶点纹理属性
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertices.itemsize * 5, ctypes.c_void_p(8))
glActiveTexture(GL_TEXTURE0)
texture = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, texture)

# Set the texture wrapping parameters
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
# Set texture filtering parameters
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

image2 = cv2.imread(os.path.join(current_dir, "face_mask.png"), cv2.IMREAD_UNCHANGED)
b, g, r, a = cv2.split(image2)
image2 = cv2.merge([r, g, b, a])
img_data2 = image2.tobytes()
image_height,image_width = image2.shape[:2]
# img_data = np.array(image.getdata(), np.uint8) # second way of getting the raw image data
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data2)


glUseProgram(shader)
# glUniform1i(glGetUniformLocation(shader, "texture0"), 0)
# glUniform1i(glGetUniformLocation(shader, "texture1"), 1)
glClearColor(0, 0.1, 0.1, 1)
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
verts_loc = glGetUniformLocation(shader, "vertArray")
frame_index = 0
INDEX_LEFT_IRIS = [474, 475, 476, 477]
INDEX_RIGHT_IRIS = [469, 470, 471, 472, 469]

glfw.make_context_current(window)
glfw.swap_interval(0)
start_time = time.time()
glActiveTexture(GL_TEXTURE1)
texture1 = glGenTextures(1)
ref_vt = None
def set_ref_texture(vt, ref_image):
    global ref_vt
    ref_vt = vt
    glBindTexture(GL_TEXTURE_2D, texture1)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    # image = cv2.imread(os.path.join(current_dir, "face_mask2.png"), cv2.IMREAD_UNCHANGED)
    image = ref_image
    img_data = image.tobytes()
    image_height, image_width = image.shape[:2]
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

def render2cv(pts_array_, w = 384, h = 384):
    global verts_loc,VBO,window
    if glfw.get_window_size(window)[0] != w:
        glfw.set_window_size(window, w, h)
    glfw.poll_events()
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glClearColor(0, 0, 0, 0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    # glClearColor(0,0,0,0)
    pts_array = pts_array_.copy()

    if len(pts_array) == 478:
        # 加入牙齿
        pts_array = np.concatenate([pts_array, np.zeros([36, 3])], axis = 0)
    else:
        assert len(pts_array) == 478 + 36

    pts_array = pts_array[:, :2]/ np.array([w, h]) * 2 - 1
    glUniform1i(glGetUniformLocation(shader, "texture0"), 0)
    # glUniform1i(glGetUniformLocation(shader, "texture1"), 1)
    vertices = []
    # print("vt_", len(vt_))
    for i in range(468):
        vertices.extend([pts_array[i, 0], -pts_array[i, 1], vt_[2 * i], 1 - vt_[2 * i + 1], 0])
    for i in range(468, 478):
        vertices.extend([pts_array[i, 0], -pts_array[i, 1], 0, 0, 1])
    for i in range(478, 478 + 18):
        vertices.extend([pts_array[i, 0], -pts_array[i, 1], 0, 0, 2])
    for i in range(478 + 18, 478 + 18*2):
        vertices.extend([pts_array[i, 0], -pts_array[i, 1], 0, 0, 3])
    # for i in range(478, 478 + 18):
    #     vertices.extend([0, 0, 0, 0, 2])
    # for i in range(478 + 18, 478 + 18*2):
    #     vertices.extend([0, 0, 0, 0, 3])
    vertices = np.array(vertices, dtype=np.float32)
    glUniform2fv(verts_loc, 478 + 36, vertices)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
    # 绘制第一个矩形
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    if w == h*2:
        glUniform1i(glGetUniformLocation(shader, "texture0"), 1)
        vertices = []
        # print("vt_", len(vt_))
        for i in range(468):
            vertices.extend([pts_array[i, 0] + 1.0, -pts_array[i, 1], ref_vt[i, 0], ref_vt[i, 1], 0])
        for i in range(468, 478):
            vertices.extend([pts_array[i, 0] + 1.0, -pts_array[i, 1], 0, 0, 1])
        # for i in range(478, 478 + 18):
        #     vertices.extend([pts_array[i, 0] + 1.0, -pts_array[i, 1], 0, 0, 2])
        # for i in range(478 + 18, 478 + 18 * 2):
        #     vertices.extend([pts_array[i, 0] + 1.0, -pts_array[i, 1], 0, 0, 3])
        for i in range(478, 478 + 18):
            vertices.extend([0, 0, 0, 0, 2])
        for i in range(478 + 18, 478 + 18 * 2):
            vertices.extend([0, 0, 0, 0, 3])
        vertices = np.array(vertices, dtype=np.float32)
        glUniform2fv(verts_loc, 478, vertices)

        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
        # 绘制第一个矩形
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    glfw.swap_buffers(window)
    glReadBuffer(GL_FRONT)
    # 从缓冲区中的读出的数据是字节数组
    data = glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, outputType=None)
    rgba = data.reshape(h, w, -1)
    rgba = np.flip(rgba, 0).astype(np.uint8)
    return rgba
