import argparse
import mmcv
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet3d.utils import collect_env, get_root_logger
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Rectangle
import rdp


def perspective(cam_coords, proj_mat):
    pix_coords = proj_mat @ cam_coords
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    return pix_coords


def parse_args():
    parser = argparse.ArgumentParser(description='vis hdmaptr map gt label')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--samples', default=2000, help='samples to visualize')
    parser.add_argument(
        '--log-interval', default=50, help='interval of logging')
    parser.add_argument(
        '--show-dir', help='directory where visualizations will be saved')
    parser.add_argument('--show-cam', action='store_true', help='show camera pic')
    parser.add_argument(
        '--gt-format',
        type=str,
        nargs='+',
        default=['se_points', ],
        help='vis format, default should be "points",'
             'support ["se_pts","bbox","fixed_num_pts","polyline_pts"]')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.show_dir is None:
        args.show_dir = osp.join('./work_dirs',
                                 osp.splitext(osp.basename(args.config))[0],
                                 'vis_pred')
    # create vis_label dir
    mmcv.mkdir_or_exist(osp.abspath(args.show_dir))
    cfg.dump(osp.join(args.show_dir, osp.basename(args.config)))
    logger = get_root_logger()
    logger.info(f'DONE create vis_pred dir: {args.show_dir}')

    dataset = build_dataset(cfg.data.test)
    dataset.is_vis_on_test = True
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        # workers_per_gpu=cfg.data.workers_per_gpu,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    logger.info('Done build test data set')

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    cfg.model.pts_bbox_head.bbox_coder.max_num = 20  # TODO this is a hack
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    logger.info('loading check point')
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE
    logger.info('DONE load check point')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    img_norm_cfg = cfg.img_norm_cfg

    # get denormalized param
    mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
    std = np.array(img_norm_cfg['std'], dtype=np.float32)
    to_bgr = img_norm_cfg['to_rgb']

    # get pc_range
    pc_range = cfg.point_cloud_range

    # get car icon
    car_img = Image.open('./figs/lidar_car.png')

    # get color map: divider->r, ped->b, boundary->g
    colors_plt = ['r', 'b', 'g']

    logger.info('BEGIN vis test dataset samples gt label & pred')

    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    count = 0
    for i, data in enumerate(data_loader):
        count += 1
        if count <= 300:
            continue
        if ~(data['gt_labels_3d'].data[0][0] != -1).any():
            logger.error(f'\n empty gt for index {i}, continue')
            prog_bar.update()
            continue
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        img = data['img'][0].data[0]
        img_metas = data['img_metas'][0].data[0]
        gt_bboxes_3d = data['gt_bboxes_3d'].data[0]
        gt_labels_3d = data['gt_labels_3d'].data[0]

        pts_filename = img_metas[0]['pts_filename']
        pts_filename = osp.basename(pts_filename)
        pts_filename = pts_filename.replace('__LIDAR_TOP__', '_').split('.')[0]

        img_list = [img[0, i].permute(1, 2, 0).contiguous().numpy() for i in range(int(img.size(1)))]
        img_list = [mmcv.imdenormalize(img, mean, std, to_bgr=to_bgr) for img in img_list]

        for vis_format in args.gt_format:
            fig = plt.figure(figsize=(10, 10))
            grid = plt.GridSpec(5, 8, wspace=0.5, hspace=0.5)
            ax = plt.subplot(grid[:, :6])
            ax.axis('off')
            new_img_list = [img_list[2], img_list[0], img_list[1], img_list[4], img_list[3], img_list[5]]
            img = np.zeros([900, 2400, 3])
            for i in range(2):
                for j in range(3):
                    img[i*450:(i+1)*450, 800*j:800*(j+1), :] = new_img_list[i*3+j][:450, :, ::-1]
            img = img.astype(np.uint8)
            ax.imshow(img)
            ax = plt.subplot(grid[:, 6])
            if vis_format == 'fixed_num_pts':  # fixed_num_pts
                # visualize pred
                ax.set_title("ScalableMap")
                ax.axis(xmin=pc_range[0], xmax=pc_range[3])
                ax.axis(ymin=pc_range[1], ymax=pc_range[4])
                ax.axis('off')
                result_dic = result[0]['pts_bbox']
                boxes_3d = result_dic['boxes_3d']
                scores_3d = result_dic['scores_3d']
                labels_3d = result_dic['labels_3d']
                pts_3d = result_dic['pts_3d']
                for pred_score_3d, pred_bbox_3d, pred_label_3d, pred_pts_3d in zip(scores_3d, boxes_3d, labels_3d, pts_3d):
                    pred_pts_3d = pred_pts_3d.numpy()
                    pts_x = pred_pts_3d[:, 0]
                    pts_y = pred_pts_3d[:, 1]
                    ax.quiver(pts_x[:-1], pts_y[:-1], pts_x[1:]-pts_x[:-1], pts_y[1:]-pts_y[:-1], scale_units='xy', angles='xy', scale=1,
                               color=colors_plt[pred_label_3d])
                #
                # gt_lines_instance = gt_bboxes_3d[0].instance_list
                # for gt_line_instance, gt_label_3d in zip(gt_lines_instance, gt_labels_3d[0]):
                #     pts = np.array(list(gt_line_instance.coords))
                #     x = np.array([pt[0] for pt in pts])
                #     y = np.array([pt[1] for pt in pts])
                #     plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1,
                #                color=colors_plt[gt_label_3d], alpha=0.5)

                ax.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

                # visualize polyline_pts
                ax = plt.subplot(grid[:, 7])
                ax.set_title("GT")
                ax.axis(xmin=pc_range[0], xmax=pc_range[3])
                ax.axis(ymin=pc_range[1], ymax=pc_range[4])
                ax.axis('off')
                gt_lines_instance = gt_bboxes_3d[0].instance_list

                for gt_line_instance, gt_label_3d in zip(gt_lines_instance, gt_labels_3d[0]):
                    pts = np.array(list(gt_line_instance.coords))
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1,
                               color=colors_plt[gt_label_3d], headwidth=1, headlength=1)
                ax.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])


                map_path = osp.join(args.show_dir, 'GT_fixednum_pts_MAP_' + pts_filename + '.jpg')
            elif vis_format == 'polyline_pts':  # origint gt
                ax.set_title("ground_truth_pts")
                ax.axis(xmin=pc_range[0], xmax=pc_range[3])
                ax.axis(ymin=pc_range[1], ymax=pc_range[4])
                gt_lines_instance = gt_bboxes_3d[0].instance_list
                for gt_line_instance, gt_label_3d in zip(gt_lines_instance, gt_labels_3d[0]):
                    pts = np.array(list(gt_line_instance.coords))
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])

                    plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=0.5,
                               color=colors_plt[gt_label_3d])
                ax.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
                map_path = osp.join(args.show_dir, 'GT_origin_polyline_MAP_' + pts_filename + '.jpg')
            else:
                logger.error(f'WRONG visformat for GT: {vis_format}')
                raise ValueError(f'WRONG visformat for GT: {vis_format}')

        plt.savefig(map_path, bbox_inches='tight', dpi=400)
        plt.close()
        prog_bar.update()

    logger.info('\n DONE vis test dataset samples gt label')


if __name__ == '__main__':
    main()