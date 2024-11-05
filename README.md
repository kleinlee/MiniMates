# MiniMates

MiniMates 是一款轻量级的图片数字人驱动算法，比liveportrait、EchoMimic、MuseTalk等算法快10-100倍，支持语音驱动和表情驱动两种模式，并嵌入普通电脑实时运行，让用户能够定制自己的ai伙伴。

## 🔥 更新日志
- **`2024/10/17`**：发布了最新release软件包吗，支持一键数字人实时对话，招呼你的AI伙伴！
- **`2024/10/06`**：更新了相机实时表情驱动，使用mediapipe完成ARkit表情捕捉，请尝试interface/interface_face.py！
- **`2024/10/04`**：发布了面部推理代码，支持旋转驱动、音频驱动和混合驱动。
- **`2024/09/24`**：发布了大模型语音对话原始程序及release包，在普通电脑上使用llama.cpp和edgeTTS完成实时语音对话。
## Demo
release image
![1730800572065](https://github.com/user-attachments/assets/9a836ad0-0446-4fe6-a7fb-9b801440bbc0)
release video

https://github.com/user-attachments/assets/d42b7893-34f1-422e-9027-b69110b97efa

旋转驱动-0

https://github.com/user-attachments/assets/787837b9-1c18-4303-82fa-a4c23cbd0e63

旋转驱动-0

https://github.com/user-attachments/assets/1a18e531-69c8-4b64-88a9-2b13bfb1c6fc

面部重演+语音驱动

https://github.com/user-attachments/assets/3bde6132-e541-4f4f-85b7-0a22bd2d97d1

## 亮点
- **极速体验**：开源最快的数字人表情&语音驱动算法，没有之一。独立显卡、集成显卡，乃至CPU都可以实时。
- **个性化定制**：one-shot单图驱动，最低只需要一张图片。
- **嵌入终端**：摆脱python和cuda依赖。AI女友新生！

## To Do List
- 集成至桌面软件
- 发布训练代码
- 实时相机表情驱动
- 人脸优化
- 上半身驱动
## Usage
可以在这里获取预训练模型，并将其放在checkpoint目录下

百度网盘 https://pan.baidu.com/s/18stswLIZ0zyCcVWF7kTV7g?pwd=zosn  (提取码：zosn)
### 创建环境
```bash
conda create -n MiniMates python=3.12
conda activate MiniMates
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```
### 人像抠图（可选）
```bash
python interface/matting.py <img_path> <output_path>
# eg: python interface/matting.py assets/01.jpg assets/01_rgba.png
```
output_path是你要保存的RGBA图片的位置。
例如: python interface/matting.py assets/01.jpg assets/01_rgba.png
### 用摄像头快速尝试

```bash
python interface/interface_face.py <img_path>
# eg: python interface/interface_face_rotation.py assets/01_rgba.png
```
等待几秒钟让相机启动，你可以观测到图片的人物跟随你的头部来运动。

注意img_path必须是包含RGBA四通道的图片。

目前的表情驱动还不完善，所以谨慎使用，可能会获得不稳定但有趣的结果。使用的blendshape模版在checkpoint/bs_dict.pkl文件中，欢迎提出修改意见。
### 用一个人物的视频当做表情模版
```bash
python interface/generate_move_template.py  <video_path> <template_path>
# eg: python interface/generate_move_template.py assets/driving.mp4 assets/driving.template
```
video_path是你找的模版视频，template_path则是要生成的模版文件位置。

### 让图片人物按照语音文件和表情模版来生成视频
```bash
python interface/interface_audio.py  <img_path> <wav_path> <output_path> <template_path>
# eg: python interface/interface_audio.py  assets/01_rgba.png assets/audio.wav assets/output.mp4 assets/driving.template
```
template_path是可选项，若template_path不存在，那么人物就会在头部静止状态下说话。
## 算法介绍
MiniMates 采用 coarse-to-fine 的 wrap network 架构，取代传统的 dense motion 方法，以实现在 CPU 上的性能提升。
此外，我们还使用 显式的 UV map 技术来提高人像的精度。
![40ae6207dd3cbee4c0df7e6474fe1c5](https://github.com/user-attachments/assets/efb0e665-4b0b-4954-b4cc-e11b35651b2c)

## 速度
以下是 MiniMates 数字人算法在不同设备和推理框架下的fps表现(纯粹推理耗时)：

| 设备                 | 推理框架           | fps |
|--------------------|----------------|-----|
| Intel i5 12600k    | ncnn-cpu       | 11  |
| AMD Ryzen7 7735H   | ncnn-cpu       | 10  |
| RTX4050 laptop     | ncnn-vulkan    | 119 |
| mac m1             | ncnn-cpu       | 36  |
| RTX3080            | ncnn-vulkan    | 100 |
| Intel graphics 770 | ncnn-vulkan    | 18  |
| mac m1             | ncnn-vulkan    | 66  |
| RTX3080            | pytorch-gpu    | 374 |



## 致谢
我们感谢以下开源项目的支持：
- [face2face-rho](#)
- [DH_live](#)
- [sherpa-onnx](#)

## License
MiniMates 数字人算法遵循 MIT 协议。

---



## 联系
| 进入QQ群聊，分享看法和最新咨询。 | 加我好友，请备注“进群”，拉你进去微信交流群。 |
|-------------------|----------------------|
| ![QQ群聊](https://github.com/user-attachments/assets/29bfef3f-438a-4b9f-ba09-e1926d1669cb) | ![微信交流群](https://github.com/user-attachments/assets/b1f24ebb-153b-44b1-b522-14f765154110) |




