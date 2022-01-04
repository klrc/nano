import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits, one_hot, binary_cross_entropy
from torchvision.ops import box_iou
from nano.ops.box2d import completely_box_iou


def iou_loss(pred, target, reduction="mean"):
    """
    calculate IoU loss with both shaped as xyxy
    """
    assert pred.shape == target.shape
    loss = 1 - completely_box_iou(pred, target)
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def label_smoothing(target, eps, num_classes, inplace=False):
    if not inplace:
        target = target.clone()
    label_indexes = target > 0
    target[label_indexes] = 1 - eps
    target[~label_indexes] = eps / (num_classes - 1)
    return target


def compute_loss(assigned_batch, num_classes=3, eps=0.05):
    """
    compute loss from collected reg (bbox), obj, cls targets,
    all anchors are flatten to batch dimension.
    reg loss is processed with GIoU loss,
    while obj & cls losses are processed with BCE_with_logits loss,

    * Note that preds should be logits. (do not need to sigmoid())
    * specially, obj loss is valid on ALL anchors.

    returns:
        loss: loss for backward
        detached_loss: detached loss, for printing usage
    """
    preds, (reg_targets, obj_targets, cls_targets) = assigned_batch
    device = preds.device
    # bbox regression loss, objectness loss, classification loss (batched)
    loss = torch.zeros(3, device=device)
    # label smoothing
    if eps > 0:
        cls_targets = label_smoothing(cls_targets, eps, num_classes)
    lbox = iou_loss(preds[obj_targets, :4], reg_targets, reduction="mean")
    lobj = binary_cross_entropy_with_logits(preds[obj_targets, 5:], cls_targets, reduction="mean")
    lcls = binary_cross_entropy_with_logits(preds[:, 4], obj_targets.float(), reduction="mean")
    # loss gain
    loss += torch.stack((lbox, lobj, lcls))
    # loss, loss items (for printing)
    return lbox + lobj + lcls, loss.detach()


class SimOTA(nn.Module):
    """
    https://github.com/Megvii-BaseDetection/YOLOX/blob/0cce4a6f4ed6b7772334a612cdcc51aa16eb0591/yolox/models/yolo_head.py#L425
    https://blog.csdn.net/Megvii_tech/article/details/120030518
    TODO: optimize with https://zhuanlan.zhihu.com/p/405789762?ivk_sa=1024320u

    * Takes about 680MiB CUDA Memory on batch_size=16
    """

    def __init__(self, num_classes, with_loss, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.with_loss = with_loss
        self.loss_kwargs = kwargs

    def forward(self, result, targets):
        targets = self.assign_batch(result, targets)
        torch.cuda.empty_cache()
        if self.with_loss:
            preds = result[0].flatten(0, 1)
            assigned_batch = (preds, targets)
            return compute_loss(assigned_batch, num_classes=self.num_classes, **self.loss_kwargs)
        else:
            return targets

    @torch.no_grad()
    def assign_batch(self, result, targets):
        """
        (Tuple) result:
            preds: N - A - < (abs)xyxy | (sigmoid) objectness, c1, c2, ... >
            grid_mask: N - A - 2
            stride_mask: N - A
        target: N(collate) - < collate_id, cid, (abs)xyxy >
        """
        preds, grid_mask, stride_mask = result
        # collect info
        cls_targets = []
        obj_targets = []
        reg_targets = []
        batch_size = preds.size(0)
        num_classes = self.num_classes
        # process per image
        for bi in range(batch_size):
            # process batch ----------------------------------------------------------------
            # get targets & preds batch
            im_index = targets[:, 0] == bi
            targets_per_image = targets[im_index]
            preds_per_image = preds[bi]

            if targets_per_image.size(0) == 0:  # no targets alive
                cls_target = preds_per_image.new_zeros((0, num_classes))
                reg_target = preds_per_image.new_zeros((0, 4))
                obj_target = preds_per_image.new_zeros(preds_per_image.size(0)).bool()
                reg_targets.append(reg_target)
                obj_targets.append(obj_target)
                cls_targets.append(cls_target)
                continue

            is_in_centers_any, is_matched = self.center_sampling(preds_per_image, targets_per_image, grid_mask, stride_mask)
            true_preds, true_targets, pred_ious = self.dynamic_topk(preds_per_image[is_in_centers_any], targets_per_image, is_matched, num_classes)
            # update matched anchor indexes
            match_mask = is_in_centers_any
            match_mask[match_mask.clone()] = true_preds
            # collect reg_target, obj_target, cls_target
            reg_target = targets_per_image[true_targets, 2:]
            obj_target = is_in_centers_any
            cls_target = (
                one_hot(
                    targets_per_image[true_targets, 1].to(torch.int64),
                    num_classes,
                )
                * pred_ious.unsqueeze(-1).detach()
            )  # cls_target = onehot_class * iou
            reg_targets.append(reg_target)
            obj_targets.append(obj_target)
            cls_targets.append(cls_target)
            # batch finished --------------------------------------------------------------
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        cls_targets = torch.cat(cls_targets, 0)
        return reg_targets, obj_targets, cls_targets

    @torch.no_grad()
    def center_sampling(self, preds_per_image, targets_per_image, grid_mask, stride_mask):
        """
        perform center sampling for targets.
        returns:
            matched_anchors: (A,) -> (Am,)  # Am is the number of matched anchors
            is_in_boxes_and_center: (T, Am)
        """
        # build match_matrix with shape (num_targets, num_grids)
        num_targets = targets_per_image.size(0)
        num_anchors = preds_per_image.size(0)
        # set positive samples (anchors inside bbox or 5x5 of box_center)
        # assert center is in box
        x_centers_per_image = (grid_mask[..., 0] + 0.5).repeat(num_targets, 1) * stride_mask  # (1, na) -> (nt, na)
        y_centers_per_image = (grid_mask[..., 1] + 0.5).repeat(num_targets, 1) * stride_mask  # (1, na) -> (nt, na)
        bboxes_x1_per_image = targets_per_image[..., 2].unsqueeze(1).repeat(1, num_anchors)  # (nt, 1) -> (nt, na)
        bboxes_y1_per_image = targets_per_image[..., 3].unsqueeze(1).repeat(1, num_anchors)  # (nt, 1) -> (nt, na)
        bboxes_x2_per_image = targets_per_image[..., 4].unsqueeze(1).repeat(1, num_anchors)  # (nt, 1) -> (nt, na)
        bboxes_y2_per_image = targets_per_image[..., 5].unsqueeze(1).repeat(1, num_anchors)  # (nt, 1) -> (nt, na)
        b_l = x_centers_per_image - bboxes_x1_per_image
        b_t = y_centers_per_image - bboxes_y1_per_image
        b_r = bboxes_x2_per_image - x_centers_per_image
        b_b = bboxes_y2_per_image - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # assert center is in 5x5 in box_center
        center_radius = 2.5
        bboxes_xc_per_image = (bboxes_x1_per_image + bboxes_x2_per_image) / 2
        bboxes_yc_per_image = (bboxes_y1_per_image + bboxes_y2_per_image) / 2
        c_x = (x_centers_per_image - bboxes_xc_per_image).abs()
        c_y = (y_centers_per_image - bboxes_yc_per_image).abs()
        center_deltas = torch.stack([c_x, c_y], 2)
        is_in_centers = center_deltas.max(dim=-1).values < center_radius * stride_mask
        is_in_centers_all = is_in_centers.sum(dim=0) > 0
        # in boxes and in centers
        is_in_boxes_or_center = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = is_in_boxes[:, is_in_boxes_or_center] & is_in_centers[:, is_in_boxes_or_center]
        return is_in_boxes_or_center, is_in_boxes_and_center

    @torch.no_grad()
    def dynamic_topk(self, matched_preds, targets, is_matched, num_classes):
        """
        perform dynamic topk algorithm on matched anchors,
        firstly, each target is assigned to k anchors, (according to sum of ranked iou)
        then multiple targets will be purged,
        which means each anchor should have <= 1 targets.
        returns:
            true_matched_preds: (D,)     # D is the number of matched anchors
            true_matched_targets: (D,)
            pred_ious: (D,)
        """
        # dynamic k algorithm
        # get l_reg & l_cls of all possible samples
        device = targets.device
        num_targets = targets.size(0)
        num_anchors = matched_preds.size(0)
        pair_wise_iou = box_iou(targets[..., 2:], matched_preds[..., :4])  # T*D
        pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)
        gt_cls_per_image = one_hot(targets[:, 1].to(torch.int64), num_classes)
        gt_cls_per_image = gt_cls_per_image.float().unsqueeze(1).repeat(1, num_anchors, 1)
        cls_preds = matched_preds[..., 4:5].sigmoid() * matched_preds[..., 5:].sigmoid()
        cls_preds = cls_preds.float().unsqueeze(0).repeat(num_targets, 1, 1).sqrt()
        with torch.cuda.amp.autocast(enabled=False):
            pair_wise_cls_loss = binary_cross_entropy(cls_preds, gt_cls_per_image, reduction="none").sum(-1)
        del cls_preds
        # get dynamic topk
        cost = pair_wise_cls_loss + 3.0 * pair_wise_iou_loss + 100000.0 * (~is_matched)
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8).to(device)
        n_candidate_k = min(10, pair_wise_iou.size(1))
        topk_ious, _ = torch.topk(pair_wise_iou, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()
        for t in range(num_targets):
            _, A = torch.topk(cost[t], k=dynamic_ks[t], largest=False)
            matching_matrix[t, A] = 1
        del topk_ious, dynamic_ks, A
        # purge duplicated assignment
        targets_per_anchor = matching_matrix.sum(0)
        if (targets_per_anchor > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, targets_per_anchor > 1], dim=0)
            matching_matrix[:, targets_per_anchor > 1] *= 0
            matching_matrix[cost_argmin, targets_per_anchor > 1] = 1
        true_matched_preds = matching_matrix.sum(0) > 0
        true_matched_targets = matching_matrix[:, true_matched_preds].argmax(0)
        pred_ious = (matching_matrix * pair_wise_iou).sum(0)[true_matched_preds]
        del pair_wise_iou, pair_wise_iou_loss, pair_wise_cls_loss
        return true_matched_preds, true_matched_targets, pred_ious


class AssignGuidanceSimOTA(SimOTA):
    """
    SimOTA with assign guidance module (from NanoDet-plus)
    https://zhuanlan.zhihu.com/p/449912627
    """

    def __init__(self, num_classes, with_loss, **kwargs):
        super().__init__(num_classes, with_loss, **kwargs)

    def forward(self, result, targets):
        """
        result: * result should be with additional guidance
        """
        guides, preds, grid_mask, stride_mask = result
        targets = self.assign_batch((guides, grid_mask, stride_mask), targets)
        assigned_batch = (preds.flatten(0, 1), targets)
        guides_batch = (guides.flatten(0, 1), targets)
        torch.cuda.empty_cache()
        if self.with_loss:
            loss_g, _ = compute_loss(guides_batch, num_classes=self.num_classes, **self.loss_kwargs)
            loss_p, detached_loss_p = compute_loss(guides_batch, num_classes=self.num_classes, **self.loss_kwargs)
            return loss_g + loss_p, detached_loss_p
        else:
            return assigned_batch
