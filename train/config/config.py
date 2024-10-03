import argparse

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def add_base(self):
        self.parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
        self.parser.add_argument('--source_channel', type=int, default=8, help='channels of input image')


class TrainingOptions(BaseOptions):
    def parse_args(self):
        self.add_base()
        self.parser.add_argument('--ref_channel', type=int, default=8, help='num of 2d keypoints')
        self.parser.add_argument('--train_data', type=str, default=r".",
                            help='json path of train data')
        self.parser.add_argument('--train_img_dir', type=str, default=r"",
                            help='img dir of train data')
        self.parser.add_argument('--start_epoch', default=1, type=int, help='start epoch in training stage')
        self.parser.add_argument('--non_decay', default=10, type=int, help='num of epoches with fixed learning rate')
        self.parser.add_argument('--decay', default=10, type=int, help='num of linearly decay epochs')
        self.parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
        self.parser.add_argument('--checkpoint', type=int, default=2, help='num of checkpoints in training stage')
        self.parser.add_argument('--lamb_perception', type=int, default=5, help='weight of perception loss')
        self.parser.add_argument('--lr_g', type=float, default=0.00008, help='learning rate of generator')
        self.parser.add_argument('--lr_d', type=float, default=0.00008, help='learning rate of discriminator')
        self.parser.add_argument('--result_path', type=str, default=r"./checkpoint",
                                 help='result path to save model')
        self.parser.add_argument('--resume', type=str, default='None',
                                 help='If true, load model and training.')
        self.parser.add_argument('--resume_path',
                                 default='',
                                 type=str,
                                 help='Save data (.pth) of previous training')
        # =========================  Discriminator ==========================
        self.parser.add_argument('--D_num_blocks', type=int, default=4, help='num of down blocks in discriminator')
        self.parser.add_argument('--D_block_expansion', type=int, default=64, help='block expansion in discriminator')
        self.parser.add_argument('--D_max_features', type=int, default=256, help='max channels in discriminator')
        return self.parser.parse_args()




