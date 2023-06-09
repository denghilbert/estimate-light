import os
import cv2
import argparse
import time
import imageio
import torch
import numpy as np
from tqdm import tqdm
from data.dataset_real import RealDataset
from torch.utils.data import DataLoader

from model.Estimator import SplitLightEstimator
from utils.tonemapping import GammaTMO, InverseGamma
from utils.loss import mae, mse
from utils.loss.pytorch_ssim import ssim
from utils.loss import CosineSimilarity, NormalNLLLoss
from utils.metrics import calc_azimuth_error
from utils.mapping.log_mapping import linear2log, log2linear
from utils.logger import *
from utils.postprocess_mask import *

colors = [(0, 0, 255),  # RED
          (255, 255, 0),  # CYAN
          (255, 0, 255),  # PURPLE
          (0, 255, 255)]  # YELLOW

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="debug mode")
parser.add_argument("--override", action="store_true")

parser.add_argument("--dataroot", type=str, default="../data/real")

parser.add_argument("--log_image", action="store_true", help="use image in log space")
parser.add_argument("--log_mu", type=float, default=16.0)
parser.add_argument("--load_estimator_path", type=str, required=True)

parser.add_argument("--batch_size", type=int, required=True)

parser.add_argument("--tmo_gamma", type=float, default=2.2)
parser.add_argument("--tmo_log_exposure", type=float, default=-2)
parser.add_argument("--exposure_compensate", type=float, default=0)
parser.add_argument("--num_loader", type=int, default=1)
parser.add_argument("--result_dir", type=str, required=True)

parser.add_argument("--dump_all", action="store_true")
parser.add_argument("--postprocess_mask", action="store_true")


def main():
    args = parser.parse_args()

    dataroot = args.dataroot

    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(os.path.join(args.result_dir), exist_ok=True)
    output_dir = os.path.join(args.result_dir, 'real')
    os.makedirs(output_dir, exist_ok=(args.debug or args.override))
    output_image_dir = os.path.join(output_dir, 'image')
    os.makedirs(output_image_dir, exist_ok=(args.debug or args.override))
    output_image_ldr_dir = os.path.join(output_dir, 'image_ldr')
    os.makedirs(output_image_ldr_dir, exist_ok=(args.debug or args.override))
    if args.dump_all:
        output_dump_dir = os.path.join(output_dir, 'dump')
        os.makedirs(output_dump_dir, exist_ok=(args.debug or args.override))

    save_options_cmdline(output_dir, args)
    logger = set_logger(output_dir)
    tb_logger = set_tb_logger(output_dir)
    tb_save_options_cmdline(tb_logger, args)

    # initialize model and load checkpoint
    estimator = SplitLightEstimator().to('cuda')
    print('loading estimator from ', args.load_estimator_path)
    estimator.load_state_dict(torch.load(args.load_estimator_path, map_location='cuda'))
    estimator.eval()
    MSE = torch.nn.MSELoss(reduction='mean')
    MAE = torch.nn.L1Loss(reduction='mean')
    BCE = torch.nn.BCELoss(reduction='mean')
    CE = torch.nn.CrossEntropyLoss(reduction='mean')
    COS = CosineSimilarity()
    NNLL = NormalNLLLoss()

    print("output path: ", output_dir)

    testSet = RealDataset(opt=args, dataroot=dataroot)
    testLoader = DataLoader(testSet, batch_size=args.batch_size, shuffle=False, num_workers=args.num_loader,
                            drop_last=False)

    all_dict = {}
    all_dict["all"] = {}
    all_dict["stat"] = {}
    mse_list = []
    mae_list = []
    rmse_list = []
    ssim_list = []
    ldr_mse_list = []
    ldr_mae_list = []
    ldr_rmse_list = []
    ldr_ssim_list = []
    az = []
    el = []

    with torch.no_grad():
        for i, test_data in enumerate(tqdm(testLoader)):
            iter_start_time = time.time()
            # retrieve the data
            persp_tensor = test_data['color'].to('cuda')  # B, 3, 240, 320
            # local_pano = test_data['local_pano'].to('cuda')  # B, num_local, 3, 64, 128
            # local_pos = test_data['local_pos'].numpy()  # B, num_local, 2
            # sun_vis = test_data['is_sunny']  # B
            # sun_pos = test_data['sun_pos']  # B
            meta = test_data['meta']  # B
            origin_size = [size.numpy() for size in test_data['origin_size']]
            B, D = len(meta), 2

            # exposure compensation
            if args.exposure_compensate != 0:
                linear_persp_tensor = InverseGamma(persp_tensor, args.tmo_gamma)
                persp_tensor = GammaTMO(linear_persp_tensor, args.tmo_gamma, args.exposure_compensate)

            # extract feats and estimates codes
            img_feat = estimator.forward_img(persp_tensor)
            [mask_est, pos_est, sky_code_est, sun_code_est] = estimator.forward_global(img_feat)
            local_feat_list = estimator.get_local_feat(
                img_feat)  # NOTE: may further use stacked hourglass supervision here
            local_feat = local_feat_list[-1]  # B, 128, 60, 80

            # B, num_local, D = local_pos.shape  # B, num_local, 2
            # patch_feat = estimator.get_patch_feat(local_feat, local_pos)  # B*num_local, 128, 9
            # local_code_est = estimator.forward_local(patch_feat)  # B*num_local, 64

            # calcuate azimuth, elevation, pos mask and cosine mask predicts
            # azimuth_deg_gt = (sun_pos[:, 0] - 63.5) / 64 * 180
            # elevation_deg_gt = (31.5 - sun_pos[:, 1]) / 64 * 180
            azimuth_deg_est, elevation_deg_est, pos_est_fine, cosine_mask_fine_est = calc_pos_cos_mask(pos_est, B)



            for im_idx in range(len(meta)):
                meta_idx = meta[im_idx]

                if not os.path.exists(os.path.join(output_image_ldr_dir, '{}_sun_angle.txt').format(meta_idx)):
                    os.mknod(os.path.join(output_image_ldr_dir, '{}_sun_angle.txt').format(meta_idx))
                f = open(os.path.join(output_image_ldr_dir, '{}_sun_angle.txt').format(meta_idx), 'a')
                angel = str(azimuth_deg_est[im_idx]) + ',' + str(elevation_deg_est[im_idx])
                f.write(angel)
                f.close()

                ldr_est_mask_path = os.path.join(output_image_ldr_dir, '{}s_mask_est.png'.format(meta_idx))
                persp_pos_vis_path = os.path.join(output_image_ldr_dir, '{}s_pos_vis.png'.format(meta_idx))

                persp_sample = persp_tensor[im_idx].cpu().numpy()
                persp_sample = persp_sample.transpose(1, 2, 0)  # 240, 320, 3
                persp_loc_vis_sample = cv2.cvtColor((persp_sample * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
                persp_loc_vis_sample = cv2.cvtColor(persp_loc_vis_sample, cv2.COLOR_BGR2RGB)  # 240, 320, 3

                mask_est_single = (np.transpose(mask_est[im_idx].expand([3, mask_est[im_idx].shape[1], mask_est[im_idx].shape[2]]).cpu().numpy(),(1, 2, 0)) * 255.0).astype(np.uint8)
                mask_est_single = cv2.resize(mask_est_single, (origin_size[1][im_idx], origin_size[0][im_idx]), interpolation=cv2.INTER_CUBIC)
                persp_loc_vis_sample = cv2.resize(persp_loc_vis_sample, (origin_size[1][im_idx], origin_size[0][im_idx]), interpolation=cv2.INTER_CUBIC)
                imageio.imwrite(ldr_est_mask_path, mask_est_single)
                imageio.imwrite(persp_pos_vis_path, persp_loc_vis_sample)
            continue

            # decode codes to images
            sky_est = log2linear(estimator.sky_decoder(sky_code_est).clamp_min(0.0).clamp_max(4.5),
                                 args.log_mu)  # B, 3, 32, 128
            sun_est = log2linear(estimator.sun_decoder(sun_code_est, pos_est_fine).clamp_min(0.0).clamp_max(4.5),
                                 args.log_mu)  # B, 3, 32, 128

            # local_app_est = log2linear(
            #     estimator.local_app_render(local_code_est, sky_code_est.repeat_interleave(num_local, 0),
            #                                sun_code_est.repeat_interleave(num_local, 0),
            #                                cosine_mask_fine_est.repeat_interleave(num_local, 0)).clamp_min(0.0),
            #     args.log_mu)  # B*num_local, 3, 64, 128
            # local_sil_est = estimator.local_sil_decoder(local_code_est)  # B*num_local, 1, 64, 128
            # local_app_est = local_app_est.view(B, num_local, 3, 64, 128)  # B, num_local, 3, 64, 128
            # local_sil_est = local_sil_est.view(B, num_local, 1, 64, 128)  # B, num_local, 1, 64, 128

            # combine to the full local lighting map
            if args.postprocess_mask:
                sun_est_np = sun_est.cpu().numpy()
                sky_est_np = sky_est.cpu().numpy()
                local_app_est_np = local_app_est.cpu().numpy()
                pos_est_fine_np = pos_est_fine.cpu().numpy()  # B, 3, 32, 128
                mask_est_np = mask_est.cpu().numpy()  # B, 1, 240, 320
                local_sil_est_np = local_sil_est.cpu().numpy()  # B, num_local, 1, 64, 128
                post_pos_est_fine_np = np.zeros_like(pos_est_fine_np)
                post_local_sil_est_np = np.zeros_like(local_sil_est_np)
                local_est_np = np.zeros((B, num_local, 3, 64, 128))
                for _b in range(B):
                    post_pos_est_fine_np[_b] = postprocess_sun_pos(pos_est_fine_np[_b], sun_est_np[_b], sun_thres=1.0)
                    for _l in range(num_local):
                        _sun_vis_est = pred_sun_vis(mask_est_np[_b], local_pos[_b][_l], shadow_thres=0.75,
                                                    sun_thres=0.4)
                        if _sun_vis_est == "shadowed":
                            post_local_sil_est_np[_b][_l] = postprocess_local_sil(local_sil_est_np[_b][_l])
                            _global_est_np = sky_est_np[_b]
                        elif _sun_vis_est == "non-shadowed":
                            post_local_sil_est_np[_b][_l] = postprocess_local_sil(local_sil_est_np[_b][_l],
                                                                                  post_pos_est_fine_np[_b])
                            _global_est_np = sky_est_np[_b] * (1 - post_pos_est_fine_np[_b]) + sun_est_np[_b] * \
                                             post_pos_est_fine_np[_b]
                        else:  # "not-sure"
                            post_local_sil_est_np[_b][_l] = postprocess_local_sil(local_sil_est_np[_b][_l])
                            _global_est_np = sky_est_np[_b] * (1 - post_pos_est_fine_np[_b]) + sun_est_np[_b] * \
                                             post_pos_est_fine_np[_b]
                        local_est_np[_b][_l][:, :32, :] = _global_est_np
                        local_est_np[_b][_l] = local_est_np[_b][_l] * (1 - post_local_sil_est_np[_b][_l]) + \
                                               local_app_est_np[_b][_l] * post_local_sil_est_np[_b][_l]

                local_sil_est = torch.Tensor(post_local_sil_est_np).cuda()
                pos_est_fine = torch.Tensor(post_pos_est_fine_np).cuda()
                global_est = sky_est * (1 - pos_est_fine) + sun_est  # B, 3, 32, 128
                local_est = torch.Tensor(local_est_np).cuda()
            else:
                global_est = sky_est * (1 - pos_est_fine) + sun_est  # B, 3, 32, 128
                local_est = torch.nn.functional.pad(global_est, (0, 0, 0, 32)).unsqueeze(1) * (
                            1 - local_sil_est) + local_app_est * local_sil_est  # B, 4, 3, 64, 128

            ldr_local = GammaTMO(local_pano.cpu(), args.tmo_gamma, args.tmo_log_exposure)

            ldr_global_est = GammaTMO(global_est.detach().cpu(), args.tmo_gamma, args.tmo_log_exposure)
            ldr_local_est = GammaTMO(local_est.detach().cpu(), args.tmo_gamma, args.tmo_log_exposure)

            # loss stats
            for im_idx in range(B):
                meta_idx = meta[im_idx]
                gt_local_sample = local_pano[im_idx]  # num_local, 3, 64, 128
                est_local_sample = local_est[im_idx]
                ldr_gt_local_sample = ldr_local[im_idx]
                ldr_est_local_sample = ldr_local_est[im_idx]

                persp_sample = persp_tensor[im_idx].cpu().numpy()
                local_pos_sample = local_pos[im_idx]  # num_local, 2
                persp_sample = persp_sample.transpose(1, 2, 0)  # 240, 320, 3
                persp_loc_vis_sample = cv2.cvtColor((persp_sample * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

                for local_idx in range(gt_local_sample.shape[0]):
                    if local_pos_sample[local_idx][0] < 0 or local_pos_sample[local_idx][1] < 0:
                        local_pos_sample[local_idx] = -local_pos_sample[local_idx] - 1
                    cv2.circle(persp_loc_vis_sample,
                               (int(local_pos_sample[local_idx][1]), int(local_pos_sample[local_idx][0])), 1,
                               colors[local_idx], 4)
                persp_loc_vis_sample = cv2.cvtColor(persp_loc_vis_sample, cv2.COLOR_BGR2RGB)  # 240, 320, 3

                tmp_dict = {}
                for local_idx in range(gt_local_sample.shape[0]):
                    _tmp_dict = {}
                    _tmp_dict['MAE'] = mae(gt_local_sample[local_idx], est_local_sample[local_idx]).cpu().item()
                    _tmp_dict['MSE'] = mse(gt_local_sample[local_idx], est_local_sample[local_idx]).cpu().item()
                    _tmp_dict['SSIM'] = ssim(gt_local_sample[local_idx:local_idx + 1],
                                             est_local_sample[local_idx:local_idx + 1]).cpu().item()
                    _tmp_dict['RMSE'] = np.sqrt(_tmp_dict['MSE'])
                    _tmp_dict['LDR_MAE'] = mae(ldr_gt_local_sample[local_idx],
                                               ldr_est_local_sample[local_idx]).cpu().item()
                    _tmp_dict['LDR_MSE'] = mse(ldr_gt_local_sample[local_idx],
                                               ldr_est_local_sample[local_idx]).cpu().item()
                    _tmp_dict['LDR_SSIM'] = ssim(ldr_gt_local_sample[local_idx:local_idx + 1],
                                                 ldr_est_local_sample[local_idx:local_idx + 1]).cpu().item()
                    _tmp_dict['LDR_RMSE'] = np.sqrt(_tmp_dict['LDR_MSE'])
                    tmp_dict[local_idx] = _tmp_dict
                    mae_list.append(_tmp_dict['MAE'])
                    mse_list.append(_tmp_dict['MSE'])
                    rmse_list.append(_tmp_dict['RMSE'])
                    ssim_list.append(_tmp_dict['SSIM'])
                    ldr_mae_list.append(_tmp_dict['LDR_MAE'])
                    ldr_mse_list.append(_tmp_dict['LDR_MSE'])
                    ldr_rmse_list.append(_tmp_dict['LDR_RMSE'])
                    ldr_ssim_list.append(_tmp_dict['LDR_SSIM'])

                    gt_img_path = os.path.join(output_image_dir, '%s_local_%d.hdr' % (meta_idx, local_idx))
                    est_img_path = os.path.join(output_image_dir, '%s_local_%d_est.hdr' % (meta_idx, local_idx))
                    ldr_gt_img_path = os.path.join(output_image_ldr_dir, '%s_local_%d.png' % (meta_idx, local_idx))
                    ldr_est_img_path = os.path.join(output_image_ldr_dir, '%s_local_%d_est.png' % (meta_idx, local_idx))

                    imageio.imwrite(gt_img_path, np.transpose(gt_local_sample[local_idx].cpu().numpy(), (1, 2, 0)))
                    imageio.imwrite(est_img_path, np.transpose(est_local_sample[local_idx].cpu().numpy(), (1, 2, 0)))
                    imageio.imwrite(ldr_gt_img_path, (
                                np.transpose(ldr_gt_local_sample[local_idx].cpu().numpy(), (1, 2, 0)) * 255.0).astype(
                        np.uint8))
                    imageio.imwrite(ldr_est_img_path, (
                                np.transpose(ldr_est_local_sample[local_idx].cpu().numpy(), (1, 2, 0)) * 255.0).astype(
                        np.uint8))

                    if args.dump_all:
                        est_img_path = os.path.join(output_dump_dir, '%s_local_%d_est.hdr' % (meta_idx, local_idx))
                        ldr_est_img_path = os.path.join(output_dump_dir, '%s_local_%d_est.png' % (meta_idx, local_idx))
                        ldr_est_sil_path = os.path.join(output_dump_dir,
                                                        '%s_local_sil_%d_est.png' % (meta_idx, local_idx))

                        local_app_path = os.path.join(output_dump_dir,
                                                      '%s_local_app_%d_est.hdr' % (meta_idx, local_idx))
                        ldr_local_app_path = os.path.join(output_dump_dir,
                                                          '%s_local_app_%d_est.png' % (meta_idx, local_idx))

                        local_app_est_hdr = local_app_est[im_idx][local_idx].cpu().numpy()
                        local_app_est_ldr = GammaTMO(local_app_est_hdr, args.tmo_gamma, args.tmo_log_exposure)

                        imageio.imwrite(est_img_path,
                                        np.transpose(est_local_sample[local_idx].cpu().numpy(), (1, 2, 0)))
                        imageio.imwrite(ldr_est_img_path, (np.transpose(ldr_est_local_sample[local_idx].cpu().numpy(),
                                                                        (1, 2, 0)) * 255.0).astype(np.uint8))
                        imageio.imwrite(ldr_est_sil_path, (np.transpose(local_sil_est[im_idx][local_idx].cpu().numpy(),
                                                                        (1, 2, 0)) * 255.0).astype(np.uint8))
                        imageio.imwrite(local_app_path, np.transpose(local_app_est_hdr, (1, 2, 0)))
                        imageio.imwrite(ldr_local_app_path,
                                        (np.transpose(local_app_est_ldr, (1, 2, 0)) * 255.0).astype(np.uint8))

                if sun_vis[im_idx] == 1:
                    tmp_dict['az'] = calc_azimuth_error(azimuth_deg_est[im_idx], azimuth_deg_gt[im_idx], unit='deg')
                    tmp_dict['el'] = elevation_deg_est[im_idx] - elevation_deg_gt[im_idx].cpu().item()
                    az.append(tmp_dict['az'])
                    el.append(tmp_dict['el'])

                ldr_est_mask_path = os.path.join(output_image_ldr_dir, '%s_mask_est.png' % (meta_idx))
                persp_pos_vis_path = os.path.join(output_image_ldr_dir, '%s_pos_vis.png' % (meta_idx))
                # import ipdb
                # ipdb.set_trace()
                imageio.imwrite(ldr_est_mask_path, (np.transpose(
                    mask_est[im_idx].expand([3, mask_est[im_idx].shape[1], mask_est[im_idx].shape[2]]).cpu().numpy(),
                    (1, 2, 0)) * 255.0).astype(np.uint8))
                imageio.imwrite(persp_pos_vis_path, persp_loc_vis_sample)

                if args.dump_all:
                    ldr_est_mask_path = os.path.join(output_dump_dir, '%s_mask_est.png' % (meta_idx))

                    sky_est_path = os.path.join(output_dump_dir, '%s_sky_est.hdr' % (meta_idx))
                    ldr_sky_est_path = os.path.join(output_dump_dir, '%s_sky_est.png' % (meta_idx))
                    sun_est_path = os.path.join(output_dump_dir, '%s_sun_est.hdr' % (meta_idx))
                    ldr_sun_est_path = os.path.join(output_dump_dir, '%s_sun_est.png' % (meta_idx))
                    pos_est_path = os.path.join(output_dump_dir, '%s_pos_est.png' % (meta_idx))
                    global_est_path = os.path.join(output_dump_dir, '%s_global_est.hdr' % (meta_idx))
                    ldr_global_est_path = os.path.join(output_dump_dir, '%s_global_est.png' % (meta_idx))
                    cosine_est_path = os.path.join(output_dump_dir, '%s_cosine_est.png' % (meta_idx))

                    sky_est_hdr = sky_est[im_idx].cpu().numpy()
                    sun_est_hdr = sun_est[im_idx].cpu().numpy()
                    sky_est_ldr = GammaTMO(sky_est_hdr, args.tmo_gamma, args.tmo_log_exposure)
                    sun_est_ldr = GammaTMO(sun_est_hdr, args.tmo_gamma, args.tmo_log_exposure)
                    pos_est_fine_ldr = pos_est_fine[im_idx].cpu().numpy()
                    global_est_hdr = global_est[im_idx].cpu().numpy()
                    global_est_ldr = ldr_global_est[im_idx].cpu().numpy()
                    cosine_mask_fine_est_ldr = cosine_mask_fine_est[im_idx].cpu().numpy()

                    imageio.imwrite(ldr_est_mask_path,
                                    (np.transpose(mask_est[im_idx].cpu().numpy(), (1, 2, 0)) * 255.0).astype(np.uint8))
                    imageio.imwrite(sky_est_path, np.transpose(sky_est_hdr, (1, 2, 0)))
                    imageio.imwrite(sun_est_path, np.transpose(sun_est_hdr, (1, 2, 0)))
                    imageio.imwrite(ldr_sky_est_path, (np.transpose(sky_est_ldr, (1, 2, 0)) * 255.0).astype(np.uint8))
                    imageio.imwrite(ldr_sun_est_path, (np.transpose(sun_est_ldr, (1, 2, 0)) * 255.0).astype(np.uint8))
                    imageio.imwrite(pos_est_path, (np.transpose(pos_est_fine_ldr, (1, 2, 0)) * 255.0).astype(np.uint8))
                    imageio.imwrite(global_est_path, np.transpose(global_est_hdr, (1, 2, 0)))
                    imageio.imwrite(ldr_global_est_path,
                                    (np.transpose(global_est_ldr, (1, 2, 0)) * 255.0).astype(np.uint8))
                    imageio.imwrite(cosine_est_path,
                                    (np.transpose(cosine_mask_fine_est_ldr, (1, 2, 0)) * 255.0).astype(np.uint8))

                all_dict['all'][meta_idx] = tmp_dict

    print()
    print("All done. Results have been saved to %s" % (output_dir))


def calc_pos_cos_mask(pos_est, batch_size):
    # use max confidence point as sun pos est and calculate cosine mask
    pos_est_conf = pos_est.clone().detach().view(batch_size, -1)  # B, 256
    max_idx = torch.argmax(pos_est_conf, dim=1)  # B
    pos_est_conf[:, :] = 0.0
    pos_est_conf[torch.arange(batch_size), max_idx] = 1.0
    pos_est_conf = pos_est_conf.view(batch_size, 1, 8, 32)  # B, 1, 8, 32
    idx_y, idx_x = np.unravel_index(max_idx.cpu().numpy(), (8, 32))
    azimuth_rad_est = (idx_x - 15.5) / 16.0 * np.pi  # B
    elevation_rad_est = (7.5 - idx_y) / 16.0 * np.pi  # B
    sun_unit_vec = np.array([np.cos(elevation_rad_est) * np.sin(azimuth_rad_est),  # x
                             np.cos(elevation_rad_est) * np.cos(azimuth_rad_est),  # y
                             np.sin(elevation_rad_est)])  # z
    # print(sun_unit_vec, sun_unit_vec.shape, type(sun_unit_vec))
    sun_unit_vec = sun_unit_vec.reshape(3, -1)  # 3, B
    _tmp = np.mgrid[63:-1:-1, 0:128:1]
    elevation_mask = _tmp[0][np.newaxis, :]
    azimuth_mask = _tmp[1][np.newaxis, :]
    elevation_mask = (elevation_mask - 31.5) / 32 * (np.pi / 2)  # 1, 64, 128
    azimuth_mask = (azimuth_mask - 63.5) / 64 * (np.pi)  # 1, 64, 128
    unit_mask = np.stack([np.cos(elevation_mask) * np.sin(azimuth_mask),
                          np.cos(elevation_mask) * np.cos(azimuth_mask),
                          np.sin(elevation_mask)], axis=-1)  # 1, 64, 128, 3
    cosine_mask_fine_est = -np.einsum('ijkl,lm->ijkm', unit_mask, sun_unit_vec)  # 1, 64, 128, B
    cosine_mask_fine_est = np.clip(cosine_mask_fine_est, 0.0, 1.0).astype(np.float32)
    cosine_mask_fine_est = np.transpose(cosine_mask_fine_est, (3, 0, 1, 2))  # B, 1, 64, 128
    cosine_mask_fine_est = torch.tensor(cosine_mask_fine_est).to('cuda')
    sun_pos_y = (idx_y + 0.5) / 8.0  # B
    sun_pos_x = (idx_x + 0.5) / 32.0  # B
    sun_pos_y = np.clip(sun_pos_y, 0, 1)
    sun_pos_x = np.clip(sun_pos_x, 0, 1)
    pos_est_fine = np.zeros((batch_size, 1, 32, 128), dtype=np.float32)  # B, 1, 32, 128
    idx_y = np.clip(sun_pos_y * 32, 0, 31.99).astype(int)
    idx_x = np.clip(sun_pos_x * 128, 0, 127.99).astype(int)
    pos_left_ind = np.maximum(int(0), idx_x - 3)
    pos_right_ind = np.minimum(int(128), pos_left_ind + 8)
    pos_left_ind = pos_right_ind - 8
    pos_upper_ind = np.maximum(int(0), idx_y - 3)
    pos_lower_ind = np.minimum(int(32), pos_upper_ind + 8)
    pos_upper_ind = pos_lower_ind - 8
    for _i in range(batch_size):
        pos_est_fine[_i, 0, pos_upper_ind[_i]:pos_lower_ind[_i], pos_left_ind[_i]:pos_right_ind[_i]] = 1.0
    pos_est_fine = torch.tensor(pos_est_fine).to('cuda')

    azimuth_deg = azimuth_rad_est / np.pi * 180.0
    elevation_deg = elevation_rad_est / np.pi * 180.0

    return (azimuth_deg, elevation_deg, pos_est_fine, cosine_mask_fine_est)


if __name__ == '__main__':
    main()