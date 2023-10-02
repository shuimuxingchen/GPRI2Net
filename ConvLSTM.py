import torch
from torch import nn
import torch.nn.functional as F


class ConvLSTMUnit(nn.Module):
    def __init__(self, in_filters=1, out_filters=64, image_shape=(128, 128)):
        super().__init__()
        width, height = image_shape
        self.Wfi = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding='same')
        self.Whi = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding='same')
        self.Wff = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding='same')
        self.Whf = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding='same')
        self.Wfo = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding='same')
        self.Who = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding='same')
        self.Wfc = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding='same')
        self.Whc = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding='same')
        self.bi = torch.nn.Parameter(torch.randn((1, out_filters, width, height)))
        self.bf = torch.nn.Parameter(torch.randn((1, out_filters, width, height)))
        self.bo = torch.nn.Parameter(torch.randn((1, out_filters, width, height)))
        self.bc = torch.nn.Parameter(torch.randn((1, out_filters, width, height)))

    def forward(self, F_in, H_in, C_in):
        i = F.sigmoid(self.Wfi(F_in) + self.Whi(H_in) + self.bi)
        f = F.sigmoid(self.Wff(F_in) + self.Whf(H_in) + self.bf)
        o = F.sigmoid(self.Wfo(F_in) + self.Who(H_in) + self.bo)
        C = torch.mul(f, C_in) + torch.mul(i, F.tanh(self.Wfc(F_in) + self.Whc(H_in) + self.bc))
        H = torch.mul(o, F.tanh(C))
        return C, H


class ConvLSTM(nn.Module):
    def __init__(self, in_filters, out_filters, direction, num_units, image_shape=(128, 128)):
        super().__init__()
        self.direction = direction
        self.num_units = num_units
        self.out_filters = out_filters
        self.conv_unit_list = nn.ModuleList(
            [ConvLSTMUnit(in_filters, out_filters, image_shape) for i in range(num_units)])

    def forward(self, list_F):
        num_sample, _, width, height = list_F[0].shape
        if not (self.direction == "forward" or self.direction == "backward"):
            raise Exception(f"ConvLSTM only has direction: \"forward\" or \"backward\" but you have: {self.direction}")
        if self.direction == "backward":
            list_F = list_F[::-1]
        C, H = self.conv_unit_list[0](list_F[0], torch.zeros((num_sample, self.out_filters, width, height)),
                                      torch.zeros((num_sample, self.out_filters, width, height)))
        H_list = [C]
        C_list = [H]
        for i in range(1, self.num_units):
            C, H = self.conv_unit_list[i](list_F[i], H_list[i - 1], C_list[i - 1])
            H_list.append(H)
            C_list.append(C)
        if self.direction == "backward":
            H_list = H_list[::-1]
        return H_list


if __name__ == "__main__":
    # model = ConvLSTMUnit().to("cuda")
    # test_tensor = torch.rand(32, 1, 128, 128).to("cuda")
    # out_test_C, out_test_H = model(test_tensor, torch.zeros(test_tensor.shape).to("cuda"),
    #                                torch.zeros(test_tensor.shape).to("cuda"))
    # print(out_test_C.shape, out_test_H.shape)
    model = ConvLSTM(1, 64, "forward", 3).to("cuda")
    test_tensor = [torch.rand(32, 1, 128, 128).to("cuda") for i in range(3)]
    out1 = model(test_tensor)
    print(out1[0].shape)
    model2 = ConvLSTM(1, 64, "backward", 3).to("cuda")
    out2 = model2(test_tensor)
