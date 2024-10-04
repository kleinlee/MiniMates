import cv2
import sys
import numpy as np
import onnxruntime as rt
import os

class Matting:
    def __init__(self, input_size=(512, 512)):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, "../checkpoint/modnet.onnx")
        self.sess = rt.InferenceSession(self.model_path)
        self.input_name = self.sess.get_inputs()[0].name
        self.label_name = self.sess.get_outputs()[0].name
        self.input_size = input_size
        self.txt_font = cv2.FONT_HERSHEY_PLAIN

    def normalize(self, im, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        im = im.astype(np.float32, copy=False) / 255.0
        im -= mean
        im /= std
        return im

    def resize(self, im, target_size=608, interp=cv2.INTER_LINEAR):
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            w = target_size[0]
            h = target_size[1]
        else:
            w = target_size
            h = target_size
        im = cv2.resize(im, (w, h), interpolation=interp)
        return im

    def preprocess(self, image, target_size=(512, 512), interp=cv2.INTER_LINEAR):
        image = self.normalize(image)
        image = self.resize(image, target_size=target_size, interp=interp)
        image = np.transpose(image, [2, 0, 1])
        image = image[None, :, :, :]
        return image

    def predict_frame(self, bgr_image):
        assert len(bgr_image.shape) == 3, "Please input RGB image."
        raw_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        h, w, c = raw_image.shape
        image = self.preprocess(raw_image, target_size=self.input_size)

        pred = self.sess.run(
            [self.label_name],
            {self.input_name: image.astype(np.float32)}
        )[0]
        pred = pred[0, 0]
        matte_np = self.resize(pred, target_size=(w, h), interp=cv2.INTER_NEAREST)
        matte_np = np.expand_dims(matte_np, axis=-1)
        return matte_np

    def predict_image(self, source_image_path, save_image_path):
        bgr_image = cv2.imread(source_image_path)
        assert len(bgr_image.shape) == 3, "Please input RGB image."
        matte_np = self.predict_frame(bgr_image)
        # bgr_image = np.ones_like(bgr_image)*255
        matting_frame = matte_np * bgr_image + (1 - matte_np) * np.full(bgr_image.shape, 0.0)
        matting_frame = matting_frame.astype('uint8')
        cv2.imwrite(save_image_path, matting_frame)
    def predict_image_rgba(self, source_image_path, save_image_path):
        bgr_image = cv2.imread(source_image_path)
        assert len(bgr_image.shape) == 3, "Please input RGB image."
        matte_np = self.predict_frame(bgr_image)
        # bgr_image = np.ones_like(bgr_image)*255
        matting_frame = matte_np * bgr_image + (1 - matte_np) * np.full(bgr_image.shape, 0.0)
        matte_np = (matte_np*255).astype('uint8')
        matting_frame = matting_frame.astype('uint8')
        matting_frame = np.concatenate([matting_frame, matte_np],axis = 2)
        cv2.imwrite(save_image_path, matting_frame)

def main():
    # 检查命令行参数的数量
    if len(sys.argv) != 3:
        print("Usage: python interface/matting.py <img_path> <output_path>")
        sys.exit(1)  # 参数数量不正确时退出程序

    img_path = sys.argv[1]
    output_path = sys.argv[2]
    print(f"img path is set to: {img_path}, output path is set to: {output_path}")
    model = Matting()
    model.predict_image_rgba(img_path, output_path)

if __name__ == "__main__":
    main()