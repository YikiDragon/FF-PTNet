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





