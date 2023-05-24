import torch

class EvalPMD(torch.nn.Module):
    def __init__(self):
        super(EvalPMD, self).__init__()
        self.Eval = torch.nn.MSELoss(reduction='mean')

    def forward(self, pred_v, gt_v):
        return self.Eval(pred_v, gt_v)
