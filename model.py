import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class PoseFeature(torch.nn.Module):
    def __init__(self, num_points=6890):
        super(PoseFeature, self).__init__()

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.norm1 = torch.nn.InstanceNorm1d(64)
        self.norm2 = torch.nn.InstanceNorm1d(128)
        self.norm3 = torch.nn.InstanceNorm1d(1024)
        self.num_points = num_points

    def forward(self, v):  # x: 3,BVp
        x = torch.relu(self.norm1(self.conv1(v)))  # B,64,V
        x = torch.relu(self.norm2(self.conv2(x)))  # B,128,V
        x = torch.relu(self.norm3(self.conv3(x)))  # B,1024,V
        x, _ = torch.max(x, dim=-1, keepdim=True)  # B,1024,1
        return x  # B,1024,1

class FFAdaIN(torch.nn.Module):
    def __init__(self, C):
        super(FFAdaIN, self).__init__()
        self.norm = torch.nn.InstanceNorm1d(C)
        self.conv_id1 = torch.nn.Conv1d(3, C, 1)
        self.conv_id2 = torch.nn.Conv1d(3, C, 1)
        self.conv_po1 = torch.nn.Conv1d(1027, C, 1)
        self.conv_po2 = torch.nn.Conv1d(1027, C, 1)
        self.alpha = torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=True)

    def forward(self, x, x_id, x_po):
        miu_id = self.conv_id1(x_id)
        sigma_id = self.conv_id2(x_id)
        miu_po = self.conv_po1(x_po)
        sigma_po = self.conv_po2(x_po)
        sigma = self.alpha * sigma_po + (1 - self.alpha) * sigma_id
        miu = self.beta * miu_po + (1 - self.beta) * miu_id
        out = self.norm(x) * sigma + miu
        return out


class FFAdaINResBlock(torch.nn.Module):
    def __init__(self, C):
        super(FFAdaINResBlock, self).__init__()
        self.conv1 = torch.nn.Conv1d(C, C, 1)
        self.conv2 = torch.nn.Conv1d(C, C, 1)
        self.ffadain1 = FFAdaIN(C)
        self.ffadain2 = FFAdaIN(C)

    def forward(self, x, x_id, x_po):
        x1 = self.ffadain1(x, x_id, x_po)
        x1 = self.conv1(torch.relu(x1))
        x2 = self.ffadain2(x1, x_id, x_po)
        x2 = self.conv2(torch.relu(x2))
        return x1 + x2

class FFAdaINDecoder(torch.nn.Module):
    def __init__(self):
        super(FFAdaINDecoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 256, 1)
        self.ffadainres1 = FFAdaINResBlock(256)
        self.conv2 = torch.nn.Conv1d(256, 1024, 1)
        self.ffadainres2 = FFAdaINResBlock(1024)
        self.conv3 = torch.nn.Conv1d(1024, 256, 1)
        self.ffadainres3 = FFAdaINResBlock(256)
        self.conv4 = torch.nn.Conv1d(256, 3, 1)

    def forward(self, x, x_id, x_po):
        x_m = self.conv1(x)
        x_m = self.ffadainres1(x_m, x_id, x_po)
        x_m = self.conv2(x_m)
        x_m = self.ffadainres2(x_m, x_id, x_po)
        x_m = self.conv3(x_m)
        x_m = self.ffadainres3(x_m, x_id, x_po)
        x_m = self.conv4(x_m)
        x_m = 2 * torch.tanh(x_m)
        return x_m


class FF_PTNet(torch.nn.Module):
    def __init__(self):
        super(FF_PTNet, self).__init__()
        self.encoder = PoseFeature()
        self.decoder = FFAdaINDecoder()

    def forward(self, v_id, v_po):
        pose_code = self.encoder(v_po)  # B,1024,N
        x_po = torch.concat([pose_code.repeat(1, 1, v_id.shape[-1]), v_id], dim=1)      # B,1024,N
        x_id = v_id
        x = v_id
        out = self.decoder(x, x_id, x_po)
        return out