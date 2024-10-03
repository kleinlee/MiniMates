from models.common.Discriminator import Discriminator
from models.common.VGG19 import Vgg19
from models.FreeFace import FreeFace
from utils import get_scheduler, update_learning_rate
from config.config import TrainingOptions
from utils import GANLoss
import cv2
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
import os
import glob
import uuid
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from data.dataset_faceReenactment3 import Dataset_Face2facerho,data_preparation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()


def showTensor(source_tensor, target_tensor, fake_out, bg):
    bg = bg * 255
    bg = bg.cpu().permute(1, 2, 0).float().detach().numpy().astype(np.uint8)
    # bg = cv2.cvtColor(bg, cv2.COLOR_BGRA2RGBA)

    inference_out = fake_out * 255
    inference_out = inference_out.cpu().permute(1, 2, 0).float().detach().numpy().astype(np.uint8)
    # inference_out = cv2.cvtColor(inference_out, cv2.COLOR_BGRA2RGBA)
    inference_in = target_tensor * 255
    inference_in = inference_in.cpu().permute(1, 2, 0).float().detach().numpy().astype(np.uint8)
    # inference_in = cv2.cvtColor(inference_in, cv2.COLOR_BGRA2RGBA)
    source_img = source_tensor * 255
    source_img = source_img.cpu().permute(1, 2, 0).float().detach().numpy().astype(np.uint8)
    # source_img = cv2.cvtColor(source_img, cv2.COLOR_BGRA2RGBA)
    inference_out = np.concatenate([source_img, inference_in, inference_out, bg], axis=1).astype(np.uint8)
    return inference_out

if __name__ == "__main__":
    # load config
    opt = TrainingOptions().parse_args()
    model_name = "FreeFace"
    size_ = 384
    opt.batch_size = 4
    opt.source_channel = 4
    opt.ref_channel = 4
    opt.result_path = "checkpoint/{}".format(model_name)
    opt.resume = False
    opt.resume_path = "checkpoint/{}/epoch_40.pth".format(model_name)
    train_log_path = os.path.join("checkpoint/{}/log".format(model_name), "")
    opt.seed = 1003
    scaler = GradScaler()
    scaler2 = GradScaler()
    device = "cuda"
    bg_mask = cv2.imread("../bg_mask.png")
    bg_mask[np.where(bg_mask < 128)] = 0
    bg_mask = cv2.resize(bg_mask, (size_, size_))
    bg_mask = (bg_mask > 0).astype(int)
    bg_mask = np.concatenate([bg_mask[:,:,0:1], bg_mask[:,:,0:1], bg_mask[:,:,0:1], bg_mask[:,:,0:1]], axis = 2)
    bg_mask = torch.from_numpy(bg_mask).float().permute(2, 0, 1).unsqueeze(0).to(device)


    # set seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    # load training data in memory
    # video_list = r"F:/preparation3"
    path_ = r"../../preparation"
    video_list = glob.glob(r"{}/*/*".format(path_))
    video_list.sort()
    video_list = random.sample(video_list, 5200)
    print(video_list)
    train_dict_info = data_preparation(video_list)
    train_set = Dataset_Face2facerho(train_dict_info, is_train=True,out_size=size_)
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batch_size, shuffle=True)
    train_data_length = len(training_data_loader)
    # init network
    net_g = FreeFace(opt.source_channel * 2, opt.ref_channel * 2, size_).cuda()
    net_d = Discriminator(opt.source_channel + opt.ref_channel, opt.D_block_expansion, opt.D_num_blocks,
                           opt.D_max_features).cuda()
    net_vgg = Vgg19().cuda()

    # set optimizer
    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr_g)
    optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr_d)
    # set scheduler
    net_g_scheduler = get_scheduler(optimizer_g, opt.non_decay, opt.decay)
    net_d_scheduler = get_scheduler(optimizer_d, opt.non_decay, opt.decay)
    # state_dict = torch.load(r"F:\C\AI\CV\TalkingFace\face2face_rho\src\checkpoints\voxceleb_face2facerho\33_motion_net.pth", map_location={'cpu':'cuda:0'})
    # net_g.motion_net.load_state_dict(state_dict)
    # state_dict = torch.load(
    #     r"F:\C\AI\CV\TalkingFace\face2face_rho\src\checkpoints\voxceleb_face2facerho\33_rendering_net.pth",
    #     map_location={'cpu': 'cuda:0'})
    # net_g.rendering_net.load_state_dict(state_dict)
    if opt.resume:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        # opt.start_epoch = checkpoint['epoch']
        net_g_static = checkpoint['state_dict']['net_g']
        net_g.load_state_dict(net_g_static, strict=False)
        # net_g.load_state_dict(net_g_static)
        net_d.load_state_dict(checkpoint['state_dict']['net_d'])
        optimizer_g.load_state_dict(checkpoint['optimizer']['net_g'])

        optimizer_d.load_state_dict(checkpoint['optimizer']['net_d'])

        net_g_scheduler.load_state_dict(checkpoint['scheduler']['net_g'])
        net_d_scheduler.load_state_dict(checkpoint['scheduler']['net_d'])

    # set criterion
    criterionGAN = GANLoss().cuda()
    criterionL1 = nn.L1Loss().cuda()


    from util.log_board import log


    os.makedirs(train_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    tag_index = 0
    # start train
    for epoch in range(opt.start_epoch, opt.non_decay + opt.decay + 1):
        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_d_scheduler, optimizer_d)
        net_g.train()
        avg_loss_g_perception = 0
        avg_loss_g_perception_face = 0
        avg_Loss_DI = 0
        avg_Loss_GI = 0
        avg_Loss_wrap = 0
        for iteration, data in enumerate(training_data_loader):
            # read data
            source_img = data['source_img'].float().cuda()
            source_prompt = data['source_prompt'].float().cuda()
            drving_img = data['drving_img'].float().cuda()
            driving_img_face = data['driving_img_face'].float().cuda()
            drving_prompt = data['drving_prompt'].float().cuda()

            driving_bg_mask = bg_mask.repeat(len(source_img), 1, 1, 1)
            # network forward
            fake_out = net_g(source_img, source_prompt, drving_img, driving_img_face, drving_prompt, driving_bg_mask)
            # down sample output image and real image
            fake_out_half = F.avg_pool2d(fake_out, 3, 2, 1, count_include_pad=False)
            target_tensor_half = F.interpolate(drving_img, scale_factor=0.5, mode='bilinear')
            # (1) Update D network
            optimizer_d.zero_grad()
            with autocast():
                # compute fake loss
                condition_fake_d = torch.cat([fake_out, drving_prompt], 1)
                # print(fake_out.size(), drving_prompt.size(), condition_fake_d.size())
                _, pred_fake_d = net_d(condition_fake_d)
                loss_d_fake = criterionGAN(pred_fake_d, False)
                # compute real loss
                condition_real_d = torch.cat([drving_img, drving_prompt], 1)
                _, pred_real_d = net_d(condition_real_d)
                loss_d_real = criterionGAN(pred_real_d, True)
                # Combine D loss
                loss_dI = (loss_d_fake + loss_d_real) * 0.5
            scaler.scale(loss_dI).backward(retain_graph=True)
            scaler.step(optimizer_d)
            scaler.update()
            # (2) Update G network
            optimizer_g.zero_grad()
            with autocast():
                _, pred_fake_dI = net_d(condition_fake_d)

                # 初始化四个通道的索引
                channels = [0, 1, 2, 3]
                random.shuffle(channels)
                selected_channels = channels[:3]

                # compute perception loss
                perception_real = net_vgg(drving_img[:, selected_channels])
                # print(target_tensor.size(), len(perception_real), perception_real[0].size())
                perception_fake = net_vgg(fake_out[:, selected_channels])
                perception_real_half = net_vgg(target_tensor_half[:, selected_channels])
                perception_fake_half = net_vgg(fake_out_half[:, selected_channels])
                loss_g_perception = 0
                for i in range(len(perception_real)):
                    loss_g_perception += criterionL1(perception_fake[i], perception_real[i])
                    loss_g_perception += criterionL1(perception_fake_half[i], perception_real_half[i])
                # if epoch > 50:
                #     for iii in range(opt.batch_size):
                #         tmp = criterionL1(perception_fake[0][iii], perception_real[0][iii])
                #         # print("XXXX", float(tmp), iii)
                #         if float(tmp) > 0.17:
                #             inference_out = showTensor(source_img[iii], drving_img[iii], fake_out[iii],
                #                                        driving_img_face[iii])
                #             video_name = os.path.split(data["video_name"][iii])[0]
                #             video_name = os.path.split(video_name)[0]
                #             video_name = os.path.split(video_name)[1]
                #             cv2.imwrite(os.path.join("../../ERROR", "{}_{:0>5d}.jpg".format(video_name, tag_index)),
                #                         inference_out)
                #             tag_index += 1

                warped_src1 = net_g.deform_img(source_img, net_g.deformation[0])
                warped_src2 = net_g.deform_img(source_img, net_g.deformation[1])
                warped_src3 = net_g.deform_img(source_img, net_g.deformation[2])

                loss_warp = net_g.calculate_warp_loss(warped_src1, drving_img) + \
                            net_g.calculate_warp_loss(warped_src2, drving_img) + \
                            net_g.calculate_warp_loss(warped_src3, drving_img)
                loss_warp = loss_warp*10

                loss_g_perception = (loss_g_perception / len(perception_real)) * opt.lamb_perception

                loss_init_field = 0
                if epoch < 5:
                    init_field_loss_weight = 5*(5 - epoch)
                    loss_init_field = criterionL1(net_g.init_motion_field_2, net_g.deformation[2].permute(0, 3, 1, 2)) + \
                                      criterionL1(net_g.init_motion_field_1, net_g.deformation[1].permute(0, 3, 1, 2)) + \
                                      criterionL1(net_g.init_motion_field_0, net_g.deformation[0].permute(0, 3, 1, 2))
                    loss_init_field = loss_init_field * init_field_loss_weight

                # gan dI loss
                loss_g_dI = criterionGAN(pred_fake_dI, True)
                # combine perception loss and gan loss
                loss_g = loss_g_perception + loss_g_dI + loss_warp + loss_init_field
            scaler2.scale(loss_g).backward()
            scaler2.step(optimizer_g)
            scaler2.update()
            message = "===> Epoch[{}]({}/{}): Loss_DI: {:.4f} Loss_GI: {:.4f} Loss_perception: {:.4f} Loss_wrap: {:.4f}  loss_init_field: {:.4f} lr_g = {:.7f} lr_d = {:.7f}".format(
                    epoch, iteration, len(training_data_loader), float(loss_dI), float(loss_g_dI),
                    float(loss_g_perception), float(loss_warp), float(loss_init_field), optimizer_g.param_groups[0]['lr'], optimizer_d.param_groups[0]['lr'])
            print(message)


            if iteration%100 == 0:
                inference_out = showTensor(source_img[0], drving_img[0], fake_out[0], driving_img_face[0])
                # print("SSS", inference_out.shape)
                # inference_out = cv2.cvtColor(inference_out, cv2.COLOR_BGR2RGB)
                video_name = os.path.split(data["video_name"][0])[0]
                video_name = os.path.split(video_name)[0]
                video_name = os.path.split(video_name)[1]
                log(train_logger, fig=inference_out, tag="Training/epoch_{}_{}_{}".format(epoch, iteration, video_name))
                # log(train_logger, fig=inference_out, tag="Training/epoch_{}_{}".format(epoch, iteration))
                real_iteration = epoch * len(training_data_loader) + iteration
                message1 = "Step {}/{}, ".format(real_iteration, (epoch + 1) * len(training_data_loader))
                message2 = ""
                losses = [loss_dI.item(), loss_g_perception.item(), loss_g_dI.item(), loss_warp.item(), loss_init_field.item() if epoch < 5 else 0]
                # losses = [loss_dI.item(), loss_g_perception.item(), loss_g_dI.item(), loss_warp.item(), 0 if epoch < 5 else 0]
                train_logger.add_scalar("Loss/loss_dI", losses[0], real_iteration)
                train_logger.add_scalar("Loss/loss_g_perception", losses[1], real_iteration)
                train_logger.add_scalar("Loss/loss_g_dI", losses[2], real_iteration)
                train_logger.add_scalar("Loss/loss_g_wrap", losses[3], real_iteration)
                train_logger.add_scalar("Loss/loss_init_field", losses[4], real_iteration)

            avg_loss_g_perception += loss_g_perception.item()
            avg_Loss_DI += loss_dI.item()
            avg_Loss_GI += loss_g_dI.item()
            avg_Loss_wrap += loss_warp.item()

        train_logger.add_scalar("Loss/{}".format("epoch_g_perception"),
                                avg_loss_g_perception / len(training_data_loader), epoch)
        train_logger.add_scalar("Loss/{}".format("epoch_DI"),
                                avg_Loss_DI / len(training_data_loader), epoch)
        train_logger.add_scalar("Loss/{}".format("epoch_GI"),
                                avg_Loss_GI / len(training_data_loader), epoch)
        train_logger.add_scalar("Loss/{}".format("epoch_wrap"),
                                avg_Loss_wrap / len(training_data_loader), epoch)

        # update_learning_rate(net_g_scheduler, optimizer_g)
        # update_learning_rate(net_d_scheduler, optimizer_d)

        # checkpoint
        if epoch % opt.checkpoint == 0:
            if not os.path.exists(opt.result_path):
                os.mkdir(opt.result_path)
            model_out_path = os.path.join(opt.result_path, 'epoch_{}.pth'.format(epoch))
            states = {
                'epoch': epoch + 1,
                'state_dict': {'net_g': net_g.state_dict(), 'net_d': net_d.state_dict()},
                'optimizer': {'net_g': optimizer_g.state_dict(), 'net_d': optimizer_d.state_dict()},
                'scheduler': {'net_g': net_g_scheduler.state_dict(), 'net_d': net_d_scheduler.state_dict()},
            }
            torch.save(states, model_out_path)
            print("Checkpoint saved to {}".format(epoch))


