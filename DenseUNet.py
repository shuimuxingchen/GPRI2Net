import torch
from torch import nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    def __init__(self, filters, num_conv=4):
        super().__init__()
        self.num_conv = num_conv
        self.conv_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        for i in range(self.num_conv):
            self.conv_list.append(nn.Conv2d(filters, filters, 3, padding='same'))
            self.bn_list.append(nn.BatchNorm2d(filters))

    def forward(self, x):
        outs = [x]
        for i in range(self.num_conv):
            temp_out = self.conv_list[i](outs[i])
            if i > 0:
                for j in range(i):
                    temp_out += outs[j]
            outs.append(F.relu(self.bn_list[i](temp_out)))
        out_final = outs[-1]
        del outs
        return out_final


class DownSample(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.layer = nn.MaxPool2d(kernel_size, stride)

    def forward(self, x):
        return self.layer(x)


class UpSampleAndConcat(nn.Module):
    def __init__(self, filters, kernel_size=3, padding=0):
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(filters, filters, kernel_size, padding=padding, stride=2)

    def forward(self, x, y):
        x = self.up_sample(x)
        x = torch.cat([x, y], dim=1)
        return x


class ConvToBNToReUL(nn.Module):
    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.conv = nn.Conv2d(in_filters, out_filters, 3, padding='same')
        self.bn = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class DenseUNet(nn.Module):
    def __init__(self, start_filters=1, filters=64, num_layer=4, end_filters=1):
        super().__init__()
        self.num_layer = num_layer
        self.start_conv = ConvToBNToReUL(start_filters, filters)
        self.down_dense_list = nn.ModuleList([DenseBlock(filters) for i in range(num_layer)])
        self.down_sample_list = nn.ModuleList([DownSample() for i in range(num_layer)])
        self.bottom_dense = DenseBlock(filters)
        self.up_sample_list = nn.ModuleList([UpSampleAndConcat(filters,
                                                               kernel_size=2,
                                                               padding=0)
                                             for i in range(num_layer)])
        self.up_conv_list = nn.ModuleList([ConvToBNToReUL(2 * filters, filters) for i in range(num_layer)])
        self.up_dense_list = nn.ModuleList([DenseBlock(filters) for i in range(num_layer)])
        self.end_conv = ConvToBNToReUL(filters, end_filters)

    def forward(self, x):
        x = self.start_conv(x)
        out = [x]
        temp_x = x
        for i in range(self.num_layer):
            temp_x = self.down_dense_list[i](temp_x)
            out.append(temp_x)
            temp_x = self.down_sample_list[i](temp_x)
        temp_x = self.bottom_dense(temp_x)
        for i in range(self.num_layer):
            temp_x = self.up_sample_list[i](temp_x, out[self.num_layer - i])
            temp_x = self.up_conv_list[i](temp_x)
            temp_x = self.up_dense_list[i](temp_x)
        temp_x = self.end_conv(temp_x)
        return temp_x


if __name__ == '__main__':
    model = DenseUNet().to("cuda")
    test_tensor = torch.rand(1, 1, 128, 128).to("cuda")
    out_test = model(test_tensor)
    print(test_tensor.shape, out_test.shape)
