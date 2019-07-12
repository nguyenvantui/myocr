import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, gradcheck
from torch.autograd.gradcheck import gradgradcheck
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import torchvision.utils as vutils
from model.utils.config import cfg
from model.rpn.rpn_fpn import _rpn_fpn
# from model.roi_module import RoIPooling2D as ROIPool
from model.roi_pooling.modules.roi_pool_py import RoIPool as ROIPool
from model.rpn.proposal_target_layer import _ProposalTargetLayer
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
import time
import pdb
from ipdb import set_trace as bp
from torch.autograd import Function
from torch.nn.modules.module import Module
from model.roi_pool import roi_pool_cuda
from torch.autograd.function import once_differentiable
# pass
#
class c_roi(Function):
    def __init__(ctx, pooled_height, pooled_width, spatial_scale):
        ctx.pooled_width = pooled_width
        ctx.pooled_height = pooled_height
        ctx.spatial_scale = spatial_scale
        ctx.feature_size = None

    @once_differentiable
    def forward(ctx, features, rois):
        ctx.feature_size = features.size()
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        num_rois = rois.size(0)
        output = features.new(num_rois, num_channels, ctx.pooled_height, ctx.pooled_width).zero_()
        ctx.argmax = features.new(num_rois, num_channels, ctx.pooled_height, ctx.pooled_width).zero_().int()
        ctx.rois = rois
        if len(ctx.rois.shape) == 1:
            ctx.rois = ctx.rois.view(1, -1)
        roi_pool_cuda.forward(features, ctx.rois, ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale,  output, ctx.argmax)
        # print("")

        # int roi_pooling_forward_cuda(at::Tensor features, at::Tensor rois,
        # int pooled_height, int pooled_width,
        # float spatial_scale, at::Tensor output, at::Tensor argmax) {
        return output
#
    def backward(ctx, grad_output):
        assert(ctx.feature_size is not None and grad_output.is_cuda)
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        grad_input = grad_output.new(batch_size, num_channels, data_height, data_width).zero_()

        # roi_pool_cuda.roi_pool_backward(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale,
        #                                       grad_output, ctx.rois, grad_input, ctx.argmax)

        # num_rois = ctx.rois.size(0)
        roi_pool_cuda.backward(grad_output, ctx.rois,ctx.argmax, ctx.spatial_scale, grad_input)

        # pass
        # ROIPoolBackwardLaucher(top_grad, rois, argmax, spatial_scale, batch_size,
        #                        channels, height, width, num_rois, pooled_height,
        #                        pooled_width, bottom_grad);
#       int roi_pooling_backward_cuda(at::Tensor top_grad, at::Tensor rois,
        #                               at::Tensor argmax, float spatial_scale,
        #                               at::Tensor bottom_grad) {
        return grad_input, None
#
class roi_pool(Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(roi_pool, self).__init__()

        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois, scale):
        return c_roi(self.pooled_height, self.pooled_width, scale)(features, rois)

class myfpn(nn.Module):
    def __init__(self, classes, class_agnostic):
        super(myfpn, self).__init__()
        self.n_classes = classes + 1
        self.class_agnostic = class_agnostic
        self.rcnn_loss_cls = 0
        self.rcnn_loss_bbox = 0

        self.maxpool2d = nn.MaxPool2d(1, stride=2)
        self.rcnn_rpn = _rpn_fpn(self.dout_base_model)
        self.rcnn_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.rcnn_roi_pool = roi_pool(7, 7, 1.0 / 16.0)

    def _init_weights(self, range):
        def normal_init(m, mean, stddev, truncated=False):

            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        def weights_init(m, mean, stddev, truncated=False):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

        # print()
        # bp()
        # print("--====")
        # print(False)
        # print("--====")

        normal_init(self.rcnn_toplayer, 0, range, False)
        normal_init(self.rcnn_smooth1, 0, range, False)
        normal_init(self.rcnn_smooth2, 0, range, False)
        normal_init(self.rcnn_smooth3, 0, range, False)
        normal_init(self.rcnn_latlayer1, 0, range, False)
        normal_init(self.rcnn_latlayer2, 0, range, False)
        normal_init(self.rcnn_latlayer3, 0, range, False)
        normal_init(self.rcnn_rpn.rpn_Conv, 0, range, False)
        normal_init(self.rcnn_rpn.rpn_cls_score, 0, range, False)
        normal_init(self.rcnn_rpn.rpn_bbox_pred, 0, range, False)
        normal_init(self.rcnn_cls_score, 0, range, False)
        normal_init(self.rcnn_bbox_pred, 0, range/10, False)

        weights_init(self.rcnn_top, 0, range, False)

    def create_architecture(self):
        self._init_modules()
        self._init_weights(range=0.01)

    def merge(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def pyramid_roi(self, feat_maps, rois, im_info, canonical_size = 224.0):

        mg_area = im_info[0][0] * im_info[0][1]
        h = rois.data[:, 4] - rois.data[:, 2] + 1
        w = rois.data[:, 3] - rois.data[:, 1] + 1
        roi_level = torch.log(torch.sqrt(h * w) / canonical_size) / np.log(2)
        roi_level = torch.round(roi_level + 4)
        roi_level[roi_level < 2] = 2
        roi_level[roi_level > 5] = 5

        roi_pool_feats = []
        box_to_levels = []
        num_rois = rois.shape[0]
        num_channels = feat_maps[0].shape[1]
        dtype, device = feat_maps[0].dtype, feat_maps[0].device
        roi_pool_feats = torch.zeros(
            (num_rois, num_channels, 7, 7),
            dtype=dtype,
            device=device,
        )
        for i, l in enumerate(range(2, 6)):
            if (roi_level == l).sum() == 0:
                continue
            idx_l = (roi_level == l).nonzero().squeeze(1)
            # box_to_levels.append(idx_l)
            scale = feat_maps[i].size(2) / im_info[0][0]
            roi_pool_feats[idx_l] = self.rcnn_roi_pool(feat_maps[i], rois[idx_l], scale)
            # roi_pool_feats.append(feat)
        # roi_pool_feat = torch.cat(roi_pool_feats, 0)

        # print(box_to_level[0].shape)
        # print("================================")
        # for bb in box_to_levels:
        #     print(bb)
        #     print(bb.shape)
        # box_to_level = torch.cat(box_to_levels, 0)
        # idx_sorted, order = torch.sort(box_to_level)
        # roi_pool_feat = roi_pool_feat[order]

        return roi_pool_feats
    # def pyramid(self, p, c):
    #     pp = self.merge(p, self.rcnn_latlayer3(c))
    #     pp = self.rcnn_smooth3(pp)
    #     return pp

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        c1 = self.rcnn_layer0(im_data)
        c2 = self.rcnn_layer1(c1)
        c3 = self.rcnn_layer2(c2)
        c4 = self.rcnn_layer3(c3)
        c5 = self.rcnn_layer4(c4)


        # p4 = self.pyramid(p5, c4)
        # p3 = self.pyramid(p4, c3)
        # p2 = self.pyramid(p3, c2)
        # =====================================================
        p5 = self.rcnn_toplayer(c5)
        p4 = self.merge(p5, self.rcnn_latlayer1(c4))
        p4 = self.rcnn_smooth1(p4)
        p3 = self.merge(p4, self.rcnn_latlayer2(c3))
        p3 = self.rcnn_smooth2(p3)
        p2 = self.merge(p3, self.rcnn_latlayer3(c2))
        p2 = self.rcnn_smooth3(p2)
        p6 = self.maxpool2d(p5)
        # ==========================================================
        # c6 = self.rcnn_layer5(c5)
        # p6 = self.rcnn_toplayer(c6)
        # p5 = self.rcnn_latlayer1(c5) + p6
        # p4 = self.rcnn_latlayer2(c4) + p5
        # p3 = self.merge(p4, self.rcnn_latlayer3(c3))
        # p3 = self.rcnn_smooth1(p3)
        # p2 = self.merge(p3, self.rcnn_latlayer4(c2))
        # p2 = self.rcnn_smooth2(p2)
        # =============================================================

        rpn_feature_maps = [p2, p3, p4, p5, p6]
        mrcnn_feature_maps = [p2, p3, p4, p5]

        rois, rpn_loss_cls, rpn_loss_bbox = self.rcnn_rpn(rpn_feature_maps, im_info, gt_boxes, num_boxes)

        if self.training == True:
            roi_data = self.rcnn_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, gt_assign, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois = rois.view(-1, 5)
            rois_label = rois_label.view(-1).long()
            gt_assign = gt_assign.view(-1).long()
            pos_id = rois_label.nonzero().squeeze()
            gt_assign_pos = gt_assign[pos_id]
            rois_label_pos = rois_label[pos_id]
            rois_label_pos_ids = pos_id

            rois_pos = Variable(rois[pos_id])
            rois = Variable(rois)
            rois_label = Variable(rois_label)

            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:

            rois_label, gt_assign,rois_target,rois_inside_ws,rois_outside_ws = None, None, None, None, None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
            rois = rois.view(-1, 5)
            pos_id = torch.arange(0, rois.size(0)).long().type_as(rois).long()
            rois_label_pos_ids = pos_id
            rois_pos = Variable(rois[pos_id])
            rois = Variable(rois)

        roi_pool_feat = self.pyramid_roi(mrcnn_feature_maps, rois, im_info)
        pooled_feat = self._head_to_tail(roi_pool_feat)
        bbox_pred = self.rcnn_bbox_pred(pooled_feat)

        if self.training == True:

            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.long().view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)


        cls_score = self.rcnn_cls_score(pooled_feat)
        objectiness = F.softmax(cls_score, dim =1)

        rcnn_loss_cls = 0
        rcnn_loss_bbox = 0

        if self.training == True:
            rcnn_loss_cls = F.cross_entropy(cls_score, rois_label)
            rcnn_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        rois = rois.view(batch_size, -1, rois.size(1))
        objectiness = objectiness.view(batch_size, -1, objectiness.size(1))
        # bp()
        bbox_pred = bbox_pred.view(batch_size, -1, bbox_pred.size(1))

        if self.training:
            rois_label = rois_label.view(batch_size, -1)

        loss = rpn_loss_cls + rpn_loss_bbox + rcnn_loss_cls + rcnn_loss_bbox
        return rois, objectiness, bbox_pred, rois_label, loss
