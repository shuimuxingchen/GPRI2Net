from DenseUNet import DenseUNet
from ConvLSTM import ConvLSTM
import torch
from torch import nn
import torch.nn.functional as Fn
import numpy as np


class GPRI2Net(nn.Module):
    def __init__(self, num_units=5, num_classes=5):
        super().__init__()
        self.single_BScan_feature_extraction_module = nn.ModuleList([DenseUNet() for i in range(num_units)])
        self.MBIF_F = ConvLSTM(1, 64, "forward", num_units)
        self.MBIF_B = ConvLSTM(1, 64, "backward", num_units)
        self.conv_end_MBIF = nn.ModuleList(
            [nn.Conv2d(128, 64, kernel_size=3, padding="same", bias=True) for i in range(num_units)])
        self.conv_to_sigmoid = nn.ModuleList([nn.Conv2d(64, 1, kernel_size=1) for i in range(num_units)])
        self.conv_to_softmax = nn.ModuleList([nn.Conv2d(64, num_classes, kernel_size=1) for i in range(num_units)])

    def forward(self, list_D):
        F = []
        for i in range(list_D.shape[1]):
            a = list_D[:, i, :, :, :]
            F.append(self.single_BScan_feature_extraction_module[i](list_D[:, i, :, :, :]))

        H_F = self.MBIF_F(F)
        H_B = self.MBIF_B(F)
        H = []
        for i in range(len(H_F)):
            H.append(torch.cat([H_F[i], H_B[i]], dim=1))
        E = []
        for i in range(len(H)):
            E.append(self.conv_end_MBIF[i](H[i]))
        P = []
        # I = []
        for i in range(len(E)):
            P.append(Fn.sigmoid(self.conv_to_sigmoid[i](E[i]))[:, None, :, :, :])
            temp_I = self.conv_to_softmax[i](E[i])
            # I.append(Fn.softmax(temp_I, dim=1)[:, None, :, :, :])
        return torch.cat(P, dim=1)#, torch.cat(I, dim=1)


if __name__ == "__main__":
    model = GPRI2Net(5)
    test_tensor = np.array([
        [np.random.rand(1, 128, 128) for i in range(5)],
        [np.random.rand(1, 128, 128) for i in range(5)]
    ]).astype("float32")
    test_tensor = torch.tensor(test_tensor)
    output = model(test_tensor)
    print(output[0].shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
