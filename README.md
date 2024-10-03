# MiniMates 数字人算法

MiniMates 是一款轻量级的图片数字人驱动算法，比liveportrait、EchoMimic、MuseTalk等算法快10-100倍，支持语音驱动和表情驱动两种模式，并嵌入普通电脑实时运行，让用户能够定制自己的ai伙伴。

## 亮点
- **极速体验**：开源最快的数字人表情&语音驱动算法，没有之一。独立显卡、集成显卡，乃至CPU都可以实时。
- **个性化定制**：one-shot单图驱动，最低只需要一张图片。
- **嵌入终端**：摆脱python和cuda依赖。AI女友新生！

## Demo 视频
![Video](assets/rotate0.mp4)

## To Do List
- 集成至桌面软件
- 发布训练代码
- 实时相机表情驱动
- 人脸优化
- 上半身驱动
## Usage
可以在这里获取预训练模型，并将其放在checkpoint目录下

百度网盘 https://pan.baidu.com/s/1xqraihlOW4kzvpBjuoD4Yw?pwd=d48z 
提取码：d48z 

### 创建环境
```bash
conda create -n MiniMates python=3.12
conda activate MiniMates
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```
### 用摄像头快速尝试

```bash
python interface/interface_face_rotation.py <img_path>
```
你可以观测到图片的人物跟随你的头部来运动

注意：interface/interface_face.py有相同的用法，但目前的表情驱动还不完善，所以谨慎使用，可能会获得不稳定但有趣的结果。
### 用一个人物的视频当做表情模版
```bash
python interface/generate_move_template.py  <video_path> <template_path>
```
video_path是你找的模版视频，template_path则是要生成的模版文件位置
### 让图片人物按照语音文件和表情模版来生成视频
```bash
python interface/interface_audio.py  <img_path> <wav_path> <output_path> <template_path>
```
template_path是可选项，若template_path不存在，那么人物就会在头部静止状态下说话。
## 算法介绍
MiniMates 采用 coarse-to-fine 的 wrap network 架构，取代传统的 dense motion 方法，以实现在 CPU 上的性能提升。此外，我们还使用 UV map 技术来提高人像的精度。
![算法架构图](#)
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
- [SadTalker](#)

## License
MiniMates 数字人算法遵循 [LICENSE](#) 协议。

---



## 邮箱
[kleinlee1@outlook.com].
