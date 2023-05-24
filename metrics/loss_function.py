import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class LossRecL1(torch.nn.Module):
    def __init__(self):
        super(LossRecL1, self).__init__()
        self.L1Loss = torch.nn.L1Loss(reduction='mean')
    def forward(self, pred_v, gt_v):
        loss = self.L1Loss(pred_v, gt_v)
        return loss

class LossRecL2(torch.nn.Module):
    def __init__(self):
        super(LossRecL2, self).__init__()
        self.L2Loss = torch.nn.MSELoss(reduction='mean')
    def forward(self, pred_v, gt_v):
        loss = self.L2Loss(pred_v, gt_v)
        return loss

class LossEdge(torch.nn.Module):
    def __init__(self):
        super(LossEdge, self).__init__()

    def forward(self,
                pred_v,
                edge_index_id,
                gt_v,
                edge_index_gt):
        if pred_v.dim() > 2:
            pred_v = pred_v.reshape([-1, 3])
            gt_v = gt_v.reshape([-1, 3])
        edge_len_pred = torch.sum((pred_v[edge_index_id[0]]-pred_v[edge_index_id[1]])**2, dim=1)    # E,1
        edge_len_gt = torch.sum((gt_v[edge_index_gt[0]]-gt_v[edge_index_gt[1]])**2, dim=1)         # E,1
        loss = torch.mean(torch.abs(edge_len_pred / edge_len_gt - 1))
        return loss



# import metrics.chamfer3D.dist_chamfer_3D as dist_chamfer_3D
# class LossChamferDistance(torch.nn.Module):
#     """
#     chamfer损失函数,最近邻点对的距离之和\n
#     注意：需要提供batch,因为不同批次之间不可求chamfer\n
#     Args:
#         pred_v: Tensor[BV,3]    预测点集
#         gt_v: Tensor[BV,3]      正确点集
#         batch: Tensor[BV]       批索引
#     Returns:
#         loss: chamfer损失
#     """
#     def __init__(self):
#         super(LossChamferDistance, self).__init__()
#         self.chamfer_dist = dist_chamfer_3D.chamfer_3DDist()
#     def forward(self, pred_v, gt_v, batch):
#         loss = torch.Tensor(0)
#         for i in range(batch.max()+1):
#             cur_batch = torch.argwhere(batch == i).squeeze(1)
#             pred_v_batch = pred_v[cur_batch].unsqueeze(0).cuda()      # V,3
#             gt_v_batch = gt_v[cur_batch].unsqueeze(0).cuda()          # V,3
#             dist1, dist2, idx1, idx2 = self.chamfer_dist(gt_v_batch, pred_v_batch)
#             loss += torch.mean(dist1) + torch.mean(dist2)
#         return loss

if __name__=='__main__':
    import torch.profiler as profiler
    loss_reconstruction = LossRecL1()
    # loss_chamfer = LossChamferDistance()
    loss_reconstruction.cuda()
    # loss_chamfer.cuda()
    pred_v = torch.randn(6890, 3)
    gt_v = torch.randn(6890, 3)
    batch = torch.ones(6890, dtype=torch.long)
    batch[0:10] = 0
    with profiler.profile(with_stack=True, profile_memory=True, record_shapes=True) as prof:
        with profiler.record_function("loss_reconstruction"):
            print("loss_reconstruction: %.6f" % loss_reconstruction(pred_v, gt_v))
        # with profiler.record_function("loss_chamfer"):
        #     print("loss_chamfer: %.6f" % loss_chamfer(pred_v, gt_v, batch))
    print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=5))






