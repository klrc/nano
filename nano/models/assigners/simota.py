import torch
from torch import nn
from torch.nn.functional import one_hot, binary_cross_entropy, binary_cross_entropy_with_logits
from torchvision.ops.boxes import box_iou
from nano.models.multiplex.box2d import completely_box_iou


class SimOTA(nn.Module):
    """
    https://github.com/Megvii-BaseDetection/YOLOX/blob/0cce4a6f4ed6b7772334a612cdcc51aa16eb0591/yolox/models/yolo_head.py#L425
    https://blog.csdn.net/Megvii_tech/article/details/120030518
    optimize with https://zhuanlan.zhihu.com/p/405789762?ivk_sa=1024320u

    Anchors + GT -> Positive Anchors

    """

    def __init__(self, class_balance=None):
        super().__init__()
        self.class_balance = class_balance

    @torch.no_grad()
    def center_sampling(self, pred_per_image, target_per_image):
        """
        perform center sampling for targets.
        pred_per_image:    (A,  2+1+4+C (grid, stride, box_xyxy, cls)   )
        target_per_image:  (N,  1+1+4   (collate_id, cid, xyxy)         )

        returns:
            is_in_boxes_anchor: (A,)
            is_in_boxes_and_center: (T, Am)
        """
        # build match_matrix with shape (num_targets, num_grids)
        T = target_per_image.size(0)
        A = pred_per_image.size(0)
        # set positive samples (anchors inside bbox or 5x5 of box_center)
        # assert center is in box
        x_centers_per_image = (pred_per_image[..., 0] + 0.5).repeat(T, 1) * pred_per_image[..., 2]  # (1, na) -> (nt, na)
        y_centers_per_image = (pred_per_image[..., 1] + 0.5).repeat(T, 1) * pred_per_image[..., 2]  # (1, na) -> (nt, na)
        bboxes_x1_per_image = target_per_image[..., 2].unsqueeze(1).repeat(1, A)  # (nt, 1) -> (nt, na)
        bboxes_y1_per_image = target_per_image[..., 3].unsqueeze(1).repeat(1, A)  # (nt, 1) -> (nt, na)
        bboxes_x2_per_image = target_per_image[..., 4].unsqueeze(1).repeat(1, A)  # (nt, 1) -> (nt, na)
        bboxes_y2_per_image = target_per_image[..., 5].unsqueeze(1).repeat(1, A)  # (nt, 1) -> (nt, na)
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
        is_in_centers = center_deltas.max(dim=-1).values < center_radius * pred_per_image[..., 2]  # stride mask
        is_in_centers_all = is_in_centers.sum(dim=0) > 0
        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        return is_in_boxes_anchor, is_in_boxes_and_center

    @torch.no_grad()
    def dynamic_topk(self, pred_per_image, target, in_box, in_box_center):
        """
        perform dynamic topk algorithm on matched anchors,
        firstly, each target is assigned to k anchors, (according to sum of ranked iou)
        then multiple targets will be purged,
        which means each anchor should have <= 1 targets.

        pred_per_image:    (A,  2+1+4+C (grid, stride, box_xyxy, cls)   )
        target:            (N,  1+1+4   (collate_id, cid, xyxy)         )

        returns:
            mp: (P,)   matched anchors (bool)
            tp: (P,)   target index of each matched anchor (long)
            p_iou: (P,)  sum iou for each matched anchor (float)
        """
        # dynamic k algorithm
        paired = pred_per_image[in_box, 3:]

        # get (iou+obj*cls)cost for all paired-target
        T = target.size(0)
        P = paired.size(0)
        C = paired.size(1) - 4
        device = paired.device

        pair_wise_iou = box_iou(target[..., 2:], paired[..., :4])  # (T, P)
        pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

        cls_target = one_hot(target[:, 1].to(torch.int64), C)
        cls_target = cls_target.float().unsqueeze(1).repeat(1, P, 1)
        cls_pred = paired[..., 4:].sigmoid().float().unsqueeze(0).repeat(T, 1, 1)
        with torch.cuda.amp.autocast(enabled=False):
            pair_wise_cls_loss = binary_cross_entropy(cls_pred, cls_target, reduction="none").sum(-1)  # (T, P)

        cost = pair_wise_cls_loss + 3.0 * pair_wise_iou_loss + 100000.0 * (~in_box_center)
        del cls_target, cls_pred, pair_wise_iou_loss, pair_wise_cls_loss

        # get dynamic topk
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8).to(device)  # (T, P)
        n_candidate_k = min(10, P)
        topk_ious, _ = torch.topk(pair_wise_iou, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        # max_topk = float(topk_ious.sum(1).mean().item())  # record dynamic K
        dynamic_ks = dynamic_ks.tolist()

        # select topk paired pred
        for t in range(T):
            _, p = torch.topk(cost[t], k=dynamic_ks[t], largest=False)
            matching_matrix[t, p] = 1
        del topk_ious, dynamic_ks, p

        # purge duplicated assignment
        targets_per_anchor = matching_matrix.sum(0)
        if (targets_per_anchor > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, targets_per_anchor > 1], dim=0)
            matching_matrix[:, targets_per_anchor > 1] *= 0
            matching_matrix[cost_argmin, targets_per_anchor > 1] = 1

        # collect results
        mp = matching_matrix.sum(0) > 0  # (P, )
        tp = matching_matrix[:, mp].argmax(0)  # (P, )
        p_iou = pair_wise_iou.max(0).values  # (P, )
        del pair_wise_iou, cost
        return mp, tp, p_iou

    @torch.no_grad()
    def assign_batch(self, input, target):
        """
        input:   (N, A,  2+1+4+C (    grid_xy,  stride,   xyxy,  conf1, conf2, ... ))
        target:  (N,     1+1+4   ( collate_id,     cid,   xyxy)                     )
        matched: (N, A,  1+4+C   (anchor_type,            xyxy,  conf1, conf2, ... ))    NEG=0, POS=1
        """

        N, A, C = input.shape
        C = C - 7
        matched = torch.zeros(N, A, C + 5).to(target.device)  # negative by default

        for bi in range(N):
            # process batch ----------------------------------------------------------------
            # get targets & preds batch
            idx_mask = target[:, 0] == bi
            target_per_image = target[idx_mask]
            if target_per_image.size(0) == 0:  # no targets alive
                continue
            pred_per_image = input[bi]

            # get positive candidates
            in_box, in_box_center = self.center_sampling(pred_per_image, target_per_image)
            mm, target_mask, target_iou = self.dynamic_topk(pred_per_image, target_per_image, in_box, in_box_center)

            # set matched xyxy
            pos_mask = in_box.clone()
            pos_mask[in_box] = mm
            matched[bi, pos_mask, 1:5] = target_per_image[target_mask, 2:]

            # set matched cls_conf
            quality = one_hot(target_per_image[target_mask, 1].to(torch.int64), C).float()
            quality *= target_iou[mm].unsqueeze(-1)
            matched[bi, pos_mask, 5:] = quality

            # set POS / OOD type in matched output
            om = matched[bi, pos_mask, -1] > 0
            ood_mask = pos_mask.clone()
            ood_mask[pos_mask] = om
            matched[bi, pos_mask, 0] = 1  # <POS> == 1
            matched[bi, ood_mask, 0] = 2  # <OOD> == 2
        return matched

    def forward(self, input, target, debug=False):
        """
        input:   (N, A,  2+1+4+C (    grid_x,  grid_y,  stride,   xyxy,  conf1, conf2, ... ))
        matched: (N, A,  1+4+C   (                 anchor_type,   xyxy,  conf1, conf2, ... ))    NEG=0, POS=1
        """
        matched = self.assign_batch(input, target)

        if debug:  # debug output for visualization ------------------
            N = matched.size(0)
            boxes = []
            for bi in range(N):
                pos_mask = matched[bi, :, 0] > 0
                box_pred, box_target = input[bi, pos_mask][..., 3:7], matched[bi, pos_mask][..., 1:5]
                boxes.append((box_pred, box_target))
            return boxes
        # ------------------------------------------------------------

        device = matched.device
        input = input.flatten(0, 1)  # (N*A,  2+1+4+C)
        matched = matched.flatten(0, 1)  # (N*A,  1+4+C)
        pos_mask = matched[..., 0] == 1
        box_pred, box_target = input[pos_mask, 3:7], matched[pos_mask, 1:5]
        cls_pred, cls_target = input[:, 7:], matched[:, 5:]
        loss = torch.zeros(2, device=device)

        nt = box_target.size(0)
        if nt == 0:
            lbox = 0
            lqfl = quality_focal_loss_with_ood(cls_pred, cls_target, beta=2)
        else:
            # reference: https://arxiv.org/pdf/2111.00902.pdf
            # loss = loss_vfl + 2 * loss_giou + 0.25 * loss_dfl
            lbox = 1 - completely_box_iou(box_pred, box_target)
            lqfl = quality_focal_loss_with_ood(cls_pred, cls_target, beta=2)
            # class balance using manual weights
            if self.class_balance is not None:
                class_balance = torch.tensor(self.class_balance, device=device)
                balance_mask = torch.matmul((matched[pos_mask, 5:] > 0).float(), class_balance).detach()
                lbox *= balance_mask
                lqfl[pos_mask] *= balance_mask.unsqueeze(-1)
        lbox = 0.2 * lbox.mean()
        lqfl = 0.2 * lqfl.sum() / max(nt, 1)
        loss += torch.stack((lbox, lqfl))
        # loss, loss items (for printing)
        return lbox + lqfl, loss.detach()


def quality_focal_loss_with_ood(pred, target, beta=2.0):
    r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    the Quality Focal Loss is defined as:
        QFL(p) = - |y-p|^{\beta} * ((1-y)*log(1-p) + y*log(p))
    """
    sigmoid_pred = pred.sigmoid()
    scale_factor = (target - sigmoid_pred).abs().pow(beta)
    out_of_distribution = target[..., -1] > 0
    scale_factor[out_of_distribution, -1] *= 2 * sigmoid_pred[out_of_distribution, :-1].max(dim=-1)[0].clamp(0, 1)
    scale_factor = scale_factor.detach()
    # negatives are supervised by 0 quality score
    # positives are supervised by bbox quality (IoU) score
    loss = binary_cross_entropy_with_logits(pred, target, reduction="none") * scale_factor
    return loss
