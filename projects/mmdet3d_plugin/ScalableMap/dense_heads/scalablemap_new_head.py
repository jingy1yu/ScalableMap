import copy
import torch
import torch.nn as nn
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmcv.runner import force_fp32
from mmcv.cnn import Linear, bias_init_with_prob
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
from mmdet.core import (multi_apply, reduce_mean)
from mmcv.utils import TORCH_VERSION, digit_version


def allocate_new2(ratio, total_num):
    """ allocate auxiliary points between control points version"""
    t_ratio = ratio.clone()
    l = ratio.clone() * total_num
    l = torch.floor(l).to(torch.int64)
    left_num = min(total_num - int(l.sum()), 17)
    for i in range(0, left_num):
        c = torch.max(ratio, dim=0)[1]
        l[c] = l[c] + 1
        ratio[c] = t_ratio[c] / (l[c] + 1)
    return l


def normalize_2d_bbox(bboxes, pc_range):
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes)
    cxcywh_bboxes[..., 0:1] = cxcywh_bboxes[..., 0:1] - pc_range[0]
    cxcywh_bboxes[..., 1:2] = cxcywh_bboxes[..., 1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h, patch_w, patch_h])
    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes


def normalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] - pc_range[1]
    factor = pts.new_tensor([pc_range[3] - pc_range[0], pc_range[4] - pc_range[1]])
    normalized_pts = new_pts / factor
    return normalized_pts


def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[..., 0:1] = (pts[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0])
    new_pts[..., 1:2] = (pts[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1])
    return new_pts


@HEADS.register_module()
class SMPNewHead(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 num_vec=20,
                 num_pts_per_vec=2,
                 num_pts_per_gt_vec=2,
                 transform_method='minmax',
                 gt_shift_pts_pattern='v0',
                 dir_interval=1,
                 loss_pts=dict(type='ChamferDistance', loss_src_weight=1.0, loss_dst_weight=1.0),
                 loss_dir=dict(type='PtsDirCosLoss', loss_weight=2.0),
                 loss_seg=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=5.0),
                 loss_angle=dict(type='L1Loss', loss_weight=0.005),
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

        self.transform_method = transform_method
        self.gt_shift_pts_pattern = gt_shift_pts_pattern
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.dir_interval = dir_interval
        self.embed_dims = 256

        super(SMPNewHead, self).__init__(*args, transformer=transformer, **kwargs)
        self.loss_pts = build_loss(loss_pts)
        self.loss_dir = build_loss(loss_dir)
        self.loss_angle = build_loss(loss_angle)
        self.loss_seg = build_loss(loss_seg)
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.per_layer_index = []
        self.number_iterations = 4
        for iter_idx in range(self.number_iterations):
            if iter_idx == 0:
                idxs = torch.arange(0, self.num_pts_per_gt_vec + 1,
                                    self.num_pts_per_gt_vec // (self.num_pts_per_vec - 1))
                self.per_layer_index.append(idxs)
            else:
                new_idxs = ((idxs[:-1] + idxs[1:]) / 2).long()
                idxs = torch.sort(torch.cat((idxs, new_idxs)))[0]
                self.per_layer_index.append(idxs)

        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])

        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
        self.instance_embedding = nn.Embedding(self.num_vec, self.embed_dims * 2)
        self.pts_embedding = nn.Embedding(self.num_vec * self.num_pts_per_vec, self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    # @auto_fp16(apply_to=('mlvl_feats'))
    @force_fp32(apply_to=('mlvl_feats', 'prev_bev'))
    def forward(self, mlvl_feats, lidar_feat, img_metas, prev_bev=None, only_bev=False):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        instance_embeds = self.instance_embedding.weight.unsqueeze(1)
        pts_embeds = self.pts_embedding.weight.view(self.num_vec, self.num_pts_per_vec, -1)
        object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1)

        bev_queries = self.bev_embedding.weight
        bev_mask = bev_queries.new_zeros((bs, self.bev_h, self.bev_w))
        bev_pos = self.positional_encoding(bev_mask)

        outputs = self.transformer(
            mlvl_feats,
            bev_queries,
            object_query_embeds,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
            img_metas=img_metas,
            prev_bev=prev_bev,
            pts_embeds=pts_embeds,
            instance_embeds=instance_embeds
        )

        bev_embed, hs, init_reference, inter_references, global_bev = outputs
        outputs_classes = []
        outputs_coords = []
        outputs_box_coords = []
        for lvl in range(len(hs)):
            hs[lvl] = hs[lvl].permute(1, 0, 2)
            outputs_class = self.cls_branches[lvl](hs[lvl].view(bs, self.num_vec, -1, self.embed_dims).mean(2))
            xy = self.reg_branches[lvl](hs[lvl])
            outputs_coord = (inverse_sigmoid(inter_references[lvl]) + xy).sigmoid()
            outputs_box_coord, outputs_coord = self.transform_box(outputs_coord)
            outputs_coords.append(outputs_coord)
            outputs_box_coords.append(outputs_box_coord)
            outputs_classes.append(outputs_class)

        outputs_classes = torch.stack(outputs_classes)
        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_box_coords,
            'all_pts_preds': outputs_coords,
            'drivable_area': global_bev
        }

        return outs

    def transform_box(self, pts, y_first=False):
        """
        Converting the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [cx, cy, w, h] transformed from points.
        """
        bs, _, dim = pts.shape
        pts_reshape = pts.view(bs, self.num_vec, -1, 2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.transform_method == 'minmax':
            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           pts_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_shifts_pts,
                           gt_bboxes_ignore=None,
                           gt_control_pts=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result, order_index = self.assigner.assign(bbox_pred, cls_score, pts_pred.clone(),
                                                          gt_bboxes, gt_labels, gt_shifts_pts, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        if order_index is None:
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
        pts_targets = pts_pred.new_zeros((pts_pred.size(0), pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        # point-to-line ground truth
        pr_shape = pts_pred.size(1)
        gt_control = torch.cat([gt_control_pts[ind][ass] for ind, ass in
            zip(sampling_result.pos_assigned_gt_inds, assigned_shift)], dim=0).view(-1, self.num_pts_per_gt_vec)
        if pr_shape < self.num_pts_per_gt_vec:
            if pr_shape == 3:
                current_index = self.per_layer_index[0]
            elif pr_shape == 5:
                current_index = self.per_layer_index[1]
            elif pr_shape == 9:
                current_index = self.per_layer_index[2]
            current_index = gt_control[:, current_index]
        else:
            current_index = gt_control
        # if original number of points >= predicted number of points, control points haven't been predicted completely
        insert_flag = (current_index.sum(-1) == gt_control.sum(-1)) & (gt_control.sum(-1) < pr_shape)
        if insert_flag.any():
            new_gt = []
            current_index = current_index.bool()
            pts_pred_clone = denormalize_2d_pts(pts_pred[pos_inds].detach(), self.pc_range)
            # allocate auxiliary points to each area according the ratios
            gt_clone = gt_shifts_pts[sampling_result.pos_assigned_gt_inds, assigned_shift, :, :].clone()
            pr_index = torch.arange(0, pr_shape, dtype=torch.int64, device=gt_clone.device)
            for pr, gt, curr_ind, fg in zip(pts_pred_clone, gt_clone, current_index, insert_flag):
                if not fg:
                    new_gt.append(gt)
                else:
                    gt_c_pts = gt[curr_ind].clone()
                    num_gt = gt_c_pts.shape[0]
                    sep_dis_btw_control = gt_c_pts[1:, :] - gt_c_pts[:-1, :]  # distance between two control points
                    dis_between_control = torch.linalg.norm(sep_dis_btw_control, dim=-1)
                    ratio = torch.div(dis_between_control, dis_between_control.sum()+1e-5)
                    allo_pts_num_per_area = allocate_new2(ratio, pr_shape - num_gt)
                    control_index = torch.zeros_like(pr_index)
                    control_index[0] = control_index[-1] = 1
                    true_index = pr_index[1:num_gt] + allo_pts_num_per_area.cumsum(0)
                    control_index[true_index] = 1
                    area_index = (control_index.cumsum(0) - 1)[:-1]  # each points belong to which area
                    insert_area_flag = allo_pts_num_per_area > 0
                    if insert_area_flag.any():  # need to interpolate or not
                        relative_offset = pr[:-1, :] - gt_c_pts[area_index]  # offset from predocted pt to control pt
                        gt_control_dis = sep_dis_btw_control.clone()
                        inside_area_flag = (pr[:-1, 0] > gt_c_pts[area_index, 0]) & (
                                pr[:-1, 0] < gt_c_pts[area_index + 1, 0]) \
                                           & (pr[:-1, 1] > gt_c_pts[area_index, 1]) & (
                                                   pr[:-1, 1] < gt_c_pts[area_index + 1, 1])  # pts in this area or not
                        inside_area_flag[control_index[:-1] > 0] = True
                        inside_flag = torch.stack([relative_offset.new_tensor(bool(inside_area_flag[area_index == ai].all())) for ai in range(num_gt - 1)], dim=0)
                        total_flag = torch.mul(insert_area_flag, inside_flag)
                        if total_flag.any():
                            vector = torch.mul(relative_offset, gt_control_dis[area_index]).sum(-1)  # ratio computation
                            dis_norm = torch.linalg.norm(gt_control_dis[area_index], dim=-1)
                            pred_ratio = torch.div(vector, torch.mul(dis_norm, dis_norm)+1e-5)
                            pred_ratio = pred_ratio.clamp(min=0, max=1)
                            p_ratio = [
                                torch.sort(pred_ratio[area_index == ai])[0].unsqueeze(-1).expand(-1,
                                                                                                 2) if insert_flag else
                                ((torch.linspace(0, 10, g_n + 2, device=pr_index.device) / 10)[:-1]).unsqueeze(
                                    -1).expand(-1, 2)
                                for ai, (insert_flag, g_n) in enumerate(zip(total_flag, allo_pts_num_per_area))]
                            temp_gt = [(torch.mul(p_ratio[ai],
                                                  gt_control_dis[ai]) + gt_c_pts[ai].unsqueeze(0)).view(-1, 2)
                                       for ai in range(num_gt - 1)]
                            if sum([tgt.shape[0] for tgt in temp_gt]) < pr_shape:
                                temp_gt.append(gt_c_pts[-1, :].unsqueeze(0))
                            temp_gt = torch.cat(temp_gt, dim=0)
                            temp_gt[curr_ind] = gt_c_pts.clone()
                            new_gt.append(temp_gt)
                        else:
                            new_gt.append(gt)
                    else:
                        new_gt.append(gt)
            new_gt = torch.stack(new_gt, dim=0)
            pts_targets[pos_inds] = new_gt
        else:
            pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds, assigned_shift, :, :]
        return (labels, label_weights, bbox_targets, bbox_weights, pts_targets, pts_weights, pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    pts_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None,
                    gt_control_pts_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, 'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, pts_targets_list, pts_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list, pts_preds_list,
            gt_labels_list, gt_bboxes_list, gt_shifts_pts_list, gt_bboxes_ignore_list, gt_control_pts_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, pts_targets_list, pts_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    pts_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None,
                    gt_control_pts_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_pts_list (list[Tensor]): Ground truth pts for each image
                with shape (num_gts, fixed_num, 2) in [x,y] format.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list, pts_preds_list,
                                           gt_bboxes_list, gt_labels_list, gt_shifts_pts_list,
                                           gt_bboxes_ignore_list, gt_control_pts_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        # normalized_bbox_targets = bbox_targets
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)

        # num_samples, num_order, num_pts, num_coords
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)

        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        loss_pts = self.loss_pts(pts_preds[isnotnan, :, :], normalized_pts_targets[isnotnan, :, :],
                                 pts_weights[isnotnan, :, :], avg_factor=num_total_pos)

        dir_weights = pts_weights[:, :-self.dir_interval, 0]
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        denormed_pts_preds_dir = denormed_pts_preds[:, self.dir_interval:, :] - denormed_pts_preds[:,
                                                                                :-self.dir_interval, :]
        pts_targets_dir = pts_targets[:, self.dir_interval:, :] - pts_targets[:, :-self.dir_interval, :]
        loss_dir = self.loss_dir(
            denormed_pts_preds_dir[isnotnan, :, :], pts_targets_dir[isnotnan, :, :],
            dir_weights[isnotnan, :], avg_factor=num_total_pos)

        angle_weights = dir_weights[:, :-self.dir_interval]
        pts_preds_first = pts_preds.clone()
        pts_preds_second = pts_preds_first.clone()
        pred_angle_first = pts_preds_first[:, 1:-1, :] - pts_preds_first[:, :-2, :]
        pred_angle_second = pts_preds_second[:, 2:, :] - pts_preds_second[:, 1:-1, :]
        pred_angle_similarity = self.cos(pred_angle_first, pred_angle_second)

        pts_target_first = normalized_pts_targets
        pts_target_second = pts_target_first.clone()
        target_angle_first = pts_target_first[:, 1:-1, :] - pts_target_first[:, :-2, :]
        target_angle_second = pts_target_second[:, 2:, :] - pts_target_second[:, 1:-1, :]
        target_angle_similarity = self.cos(target_angle_first, target_angle_second)
        loss_angle = self.loss_angle(
            pred_angle_similarity[isnotnan, ...], target_angle_similarity[isnotnan, ...],
            angle_weights[isnotnan, :], avg_factor=num_total_pos)

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)
            loss_angle = torch.nan_to_num(loss_angle)
        return loss_cls, loss_pts, loss_dir, loss_angle

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_drivable=None,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        gt_vecs_list = copy.deepcopy(gt_bboxes_list)
        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        all_pts_preds = preds_dicts['all_pts_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_origin = [gt_bboxes.fixed_num_sampled_points.to(device) for gt_bboxes in gt_vecs_list]

        gt_bboxes_list = [gt_bboxes.bbox.to(device) for gt_bboxes in gt_vecs_list]
        gt_shifts_pts_list = [gt_bboxes.shift_4stage_fixed_num_control_points_index for gt_bboxes in gt_vecs_list]
        control_index = [gt[1] for gt in gt_shifts_pts_list]
        gt_shifts_pts_list = [gt[0].to(device) for gt in gt_shifts_pts_list]
        all_gt_shifts_pts_list = []
        for num_layer, pt_num in enumerate([3, 5, 9, 17]):
            if pt_num < self.num_pts_per_gt_vec:
                temp_shift_pts = [gt_shifts_pts[..., self.per_layer_index[num_layer], :]
                                  for gt_shifts_pts in gt_shifts_pts_list]
                all_gt_shifts_pts_list.append(temp_shift_pts)
            else:
                all_gt_shifts_pts_list.append(gt_shifts_pts_list)
                all_gt_shifts_pts_list.append(gt_shifts_pts_list)
                all_gt_shifts_pts_list.append(gt_shifts_pts_list)
        all_control_index = [control_index for _ in range(num_dec_layers)]
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]

        losses_cls, losses_pts, losses_dir, losses_angle = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, all_pts_preds,
            all_gt_bboxes_list, all_gt_labels_list, all_gt_shifts_pts_list,
            all_gt_bboxes_ignore_list, all_control_index)

        drivable_area = preds_dicts['drivable_area']
        gt_drivable = torch.stack([gt_drive.to(drivable_area.device)
                                   for gt_drive in gt_drivable['gt_drivable']], dim=0)
        loss_drivable = self.loss_seg(drivable_area.permute(0, 2, 3, 1).reshape(-1, 2), gt_drivable.view(-1))

        loss_dict = dict()
        num_dec_layer = 0
        for loss_cls_i, loss_pts_i, loss_dir_i, loss_angle_i in zip(losses_cls, losses_pts, losses_dir, losses_angle):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
            loss_dict[f'd{num_dec_layer}.loss_dir'] = loss_dir_i
            loss_dict[f'd{num_dec_layer}.loss_angle'] = loss_angle_i
            num_dec_layer += 1
        loss_dict['loss_drivable'] = loss_drivable
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        # bboxes: xmin, ymin, xmax, ymax
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            scores = preds['scores']
            labels = preds['labels']
            pts = preds['pts']

            ret_list.append([bboxes, scores, labels, pts])

        return ret_list

