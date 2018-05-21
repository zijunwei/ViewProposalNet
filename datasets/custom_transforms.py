from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import types
from torchvision import transforms
import matplotlib.pyplot as plt


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')


class PIL2Numpy(object):
    """Convert PIL image to Numpy
    """
    def __init__(self):
        pass

    def __call__(self, pil_image):
        return np.array(pil_image)

class Resize(object):
    def __init__(self, w=224, h=224, interpolation=Image.BILINEAR):
        self.w = w
        self.h = h
        self.interpolation = interpolation
    def __call__(self, image):
        return image.resize((self.w, self.h), self.interpolation)


class RandomOrder(object):
    def __init__(self):
        pass

    def __call__(self, pil_image):
        np_image = np.array(pil_image)

        random_num = random.randint(0, 5)
        if random_num == 0:
            np_image = np_image[:,:,(0, 1, 2)]
        if random_num == 1:
            np_image = np_image[:,:,(0, 2, 1)]
        if random_num == 2:
            np_image = np_image[:,:,(1, 0, 2)]
        if random_num == 3:
            np_image = np_image[:,:,(1, 2, 0)]
        if random_num == 4:
            np_image = np_image[:,:,(2, 1, 0)]
        if random_num == 5:
            np_image = np_image[:,:,(2, 1, 1)]

        return Image.fromarray(np_image)


class RandomGray(object):
    def __init__(self, gray_rate=0.2):
        self.gray_rate = gray_rate

    def __call__(self, pil_image):
        random_num = random.uniform(0, 1)
        if random_num < self.gray_rate:
            pil_image = pil_image.convert('LA').convert('RGB')
        return pil_image

class AddGaussianNoise(object):
    """Add gaussian noise to a FloatTensor
    """

    def __init__(self, mean=0, sigma=0.05, random_state=np.random):
        self.sigma = sigma
        self.mean = mean
        self.random_state = random_state

    def __call__(self, image):
        c, h, w = image.size()
        gaussian = np.random.normal(self.mean, self.sigma, [c, w, h])
        t_gaussian = torch.from_numpy(gaussian)
        t_gaussian = t_gaussian.type(torch.FloatTensor)
        return image.add_(t_gaussian)


def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)

    def change_hue(x):
        x += hue * 255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x

    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    # constrain_image(im)
    return im


def random_in_scale(s):
    return random.uniform(1-s, 1+s)


class RandomDistortColor(object):
    def __init__(self, hue=0.1, sat=0.1, val=0.1):
        self.hue = hue
        self.sat = sat
        self.val = val

    def __call__(self, image):
        r_hue = random.uniform(-self.hue, self.hue)
        r_sat = random_in_scale(self.sat)
        r_val = random_in_scale(self.val)
        return distort_image(image, r_hue, r_sat, r_val)


class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_width, target_height)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, min_th, max_th):
        self.min_th = min_th
        self.max_th = max_th

    def __call__(self, img):
        scale_x = random.uniform(self.min_th, self.max_th)
        scale_y = random.uniform(self.min_th, self.max_th)
        shift_x = random.random() * (1 - scale_x)
        shift_y = random.random() * (1 - scale_y)
        w, h = img.size
        x1 = int(shift_x * w)
        y1 = int(shift_y * h)
        x2 = int((shift_x + scale_x) * w)
        y2 = int((shift_y + scale_y) * h)
        return img.crop((x1, y1, x2, y2))


# class RandomScale(object):
#
#     def __init__(self, preset_size=None, interpolation=Image.BILINEAR):
#         self.interpolation = interpolation
#         self.preset = False
#         if preset_size is not None:
#             self.raw_w = preset_size
#             self.raw_h = preset_size
#             self.preset = True
#
#     def __call__(self, img):
#         if self.preset:
#             raw_w = self.raw_w
#             raw_h = self.raw_h
#         else:
#             raw_w, raw_h = img.size
#
#         scale_x = random.uniform(0.5, 1.5)
#         scale_y = random.uniform(0.5, 1.5)
#         new_w = int(raw_w * scale_x)
#         new_h = int(raw_h * scale_y)
#
#         img = img.resize((new_w, new_h), self.interpolation)
#         img = img.resize((raw_w, raw_h), self.interpolation)
#
#         return img


class RandomScale(object):
    """Compared to v1 no longer scale back"""
    def __init__(self, preset_size=None, random_range=0.5, interpolation=Image.BILINEAR):
        self.interpolation = interpolation
        self.preset = False
        if preset_size is not None:
            self.new_w = preset_size
            self.new_h = preset_size
            self.preset = True
        self.random_range = random_range

    def __call__(self, img):
        if self.preset:
            new_w = self.new_w
            new_h = self.new_h
        else:
            new_w, new_h = img.size

        scale_x = random.uniform(1 - self.random_range, 1 + self.random_range)
        scale_y = random.uniform(1 - self.random_range, 1 + self.random_range)
        new_w = int(new_w * scale_x)
        new_h = int(new_h * scale_y)

        img = img.resize((new_w, new_h), self.interpolation)

        return img


class Lighting(object):
     """Lighting noise(AlexNet - style PCA - based noise)"""

     def __init__(self, alphastd, eigval, eigvec):
         self.alphastd = alphastd
         self.eigval = eigval
         self.eigvec = eigvec

     def __call__(self, img):
         if self.alphastd == 0:
             return img

         alpha = img.new().resize_(3).normal_(0, self.alphastd)
         rgb = self.eigvec.type_as(img).clone()\
             .mul(alpha.view(1, 3).expand(3, 3))\
             .mul(self.eigval.view(1, 3).expand(3, 3))\
             .sum(1).squeeze()

         return img.add(rgb.view(3, 1, 1).expand_as(img))

class Grayscale(object):

     def __call__(self, img):
         gs = img.clone()
         gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
         gs[1].copy_(gs[0])
         gs[2].copy_(gs[0])
         return gs


class Saturation(object):

     def __init__(self, var):
         self.var = var

     def __call__(self, img):
         gs = Grayscale()(img)
         alpha = random.uniform(0, self.var)
         return img.lerp(gs, alpha)


class Brightness(object):

     def __init__(self, var):
         self.var = var

     def __call__(self, img):
         gs = img.new().resize_as_(img).zero_()
         alpha = random.uniform(0, self.var)
         return img.lerp(gs, alpha)


class Contrast(object):

     def __init__(self, var):
         self.var = var

     def __call__(self, img):
         gs = Grayscale()(img)
         gs.fill_(gs.mean())
         alpha = random.uniform(0, self.var)
         return img.lerp(gs, alpha)


class RandomOrderTransform(object):
     """ Composes several transforms together in random order.
     """

     def __init__(self, transforms):
         self.transforms = transforms

     def __call__(self, img):
         if self.transforms is None:
             return img
         order = torch.randperm(len(self.transforms))
         for i in order:
             img = self.transforms[i](img)
         return img


class ColorJitter(RandomOrderTransform):

     def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
         self.transforms = []
         if brightness != 0:
             self.transforms.append(Brightness(brightness))
         if contrast != 0:
             self.transforms.append(Contrast(contrast))
         if saturation != 0:
             self.transforms.append(Saturation(saturation))


class MinScale(object):
    """Rescale the input PIL.Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.

        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w >= self.size) or (h <= w and h >= self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)


if __name__ == '__main__':
    image = Image.open('/Users/zwei/Dev/adobe_pytorch/image/img1.jpg').convert('RGB')
    # random_gray = RandomGray(0.5)
    # output_image = random_gray(image)
    # image_tensor = transforms.ToTensor()(image)
    # addGaussian = AddGaussianNoise()
    # output_image = addGaussian(image_tensor)
    print "DB"
