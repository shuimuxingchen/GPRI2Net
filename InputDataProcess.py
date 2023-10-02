import torch
import torchvision.io as ImageIo
import torchvision.transforms.functional as ImageFun


def resize(pics,size):
    l = []
    for pic in pics:
        print(pic.shape)
        l.append(ImageFun.resize(pic, size, ImageFun.InterpolationMode.NEAREST)[None, :, :, :])
    return torch.cat(l, dim=0)


if __name__ == "__main__":
    test_tensor = ImageIo.read_image("./test.png", ImageIo.ImageReadMode.GRAY)[None, :, :, :]
    input_tensor = torch.cat((test_tensor, test_tensor, test_tensor), dim=0)
    input_tensor = input_process(input_tensor)
