import os  # 导入os模块，用于与操作系统交互，比如文件路径处理、读取环境变量等
import random  # 导入random模块，用于生成随机数，这在数据增强和随机采样时很有用

import numpy as np  # 导入numpy库，并用np作为别名，numpy是Python中用于处理大型多维数组和矩阵的库
import torch.utils.data as data  # 从torch库中导入utils.data模块，通常用于创建自定义的数据集
import torchvision.transforms as transforms  # 从torchvision库中导入transforms模块，用于图像预处理和数据增强
from PIL import Image  # 从PIL（Python Imaging Library，即Pillow）库中导入Image模块，用于图像的加载、保存和基本的图像处理
from PIL import ImageEnhance  # 从PIL库中导入ImageEnhance模块，用于图像增强，比如对比度增强、亮度增强等


# several data augumentation strategies
def cv_random_flip(img, label):
    # 生成一个随机整数，0或1，用于决定是否进行翻转
    flip_flag = random.randint(0, 1)

    # 注释掉的代码行是另一个随机翻转标志，但在这段代码中并未使用
    # flip_flag2 = random.randint(0, 1)

    # 接下来的注释表明下面将进行左右翻转
    # left right flip

    # 如果flip_flag为1，则执行翻转操作
    if flip_flag == 1:
        # 使用PIL的transpose方法对图像进行左右翻转
        # Imgs.FLIP_LEFT_RIGHT是PIL中预定义的翻转方向
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # 假设label也是一个可以进行翻转操作的对象（比如一个分割图），则同样进行翻转
        label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # edge = edge.transpose(Image.FLIP_LEFT_RIGHT)

    # 返回翻转后的图像和标签，如果未进行翻转，则返回原图像和标签
    return img, label


def randomCrop(image, label):
    # 设置一个边界值，确保裁剪后的图像不会过于接近原始图像的边界
    border = 30

    # 获取图像的宽度和高度
    image_width = image.size[0]
    image_height = image.size[1]

    # 随机生成裁剪窗口的宽度，其最小值不得小于（原始宽度-边界），最大值为原始宽度
    crop_win_width = np.random.randint(image_width - border, image_width)

    # 随机生成裁剪窗口的高度，其最小值不得小于（原始高度-边界），最大值为原始高度
    crop_win_height = np.random.randint(image_height - border, image_height)

    # 计算裁剪区域的左上角和右下角的坐标
    # 注意：这里使用右移运算符(>>)进行整数除法，并且加上了crop_win_width/height的一半，
    # 以确保裁剪区域是从中心开始的，而不是从左上角开始
    # 这里的计算方式有误，会导致裁剪的坐标计算不准确。正确的做法应该单独计算x和y坐标的起始点和终点
    # 正确的代码示例如下（请替换以下四行）：
    # x_start = np.random.randint(0, image_width - crop_win_width)
    # y_start = np.random.randint(0, image_height - crop_win_height)
    # x_end = x_start + crop_win_width
    # y_end = y_start + crop_win_height
    # random_region = (x_start, y_start, x_end, y_end)

    # 错误的代码行（保留原样以便解释）
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1
    )

    # 使用PIL的crop方法对图像和标签进行裁剪
    # 由于random_region的计算错误，这里的裁剪可能不是预期的
    return image.crop(random_region), label.crop(random_region)


def randomRotation(image, label):
    # 定义旋转时使用的插值方法，这里选择的是双三次插值（BICUBIC）
    # 它通常能提供比最近邻插值（NEAREST）或双线性插值（BILINEAR）更好的图像质量
    mode = Image.BICUBIC

    # 生成一个[0, 1)之间的随机浮点数
    if random.random() > 0.8:
        # 如果这个随机数大于0.8（即大约20%的概率），则执行旋转操作
        # 生成一个介于-15到15之间的随机整数，作为旋转的角度
        random_angle = np.random.randint(-15, 15)

        # 使用PIL的rotate方法对图像进行旋转，旋转的角度是random_angle，插值方法是BICUBIC
        image = image.rotate(random_angle, mode)

        # 假设label也是一个可以旋转的对象（比如一个分割图），则同样进行旋转
        label = label.rotate(random_angle, mode)

        # edge = edge.rotate(random_angle, mode)

    # 返回旋转后的图像和标签，如果未进行旋转，则返回原图像和标签
    return image, label


def colorEnhance(image):
    # 生成一个介于0.5到1.5之间的随机浮点数，用于调整图像的亮度
    bright_intensity = random.randint(5, 15) / 10.0

    # 使用PIL的ImageEnhance模块中的Brightness类来增强图像的亮度
    # enhance方法接收一个介于0到任意正数之间的浮点数，表示增强的强度
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)

    # 生成另一个介于0.5到1.5之间的随机浮点数，用于调整图像的对比度
    contrast_intensity = random.randint(5, 15) / 10.0

    # 使用PIL的ImageEnhance模块中的Contrast类来增强图像的对比度
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)

    # 生成一个介于0到2之间的随机浮点数，用于调整图像的色彩饱和度
    # 注意：这里将随机数的范围设置为0到20，但通常色彩增强强度的合理范围在0到1之间，
    # 所以这个范围可能需要根据实际情况进行调整
    color_intensity = random.randint(0, 20) / 10.0

    # 使用PIL的ImageEnhance模块中的Color类来增强图像的色彩
    image = ImageEnhance.Color(image).enhance(color_intensity)

    # 生成一个介于0到3之间的随机浮点数，用于调整图像的清晰度
    # 注意：这里将随机数的范围设置为0到30，但通常清晰度增强强度的合理范围在0到1之间，
    # 所以这个范围可能需要根据实际情况进行调整
    sharp_intensity = random.randint(0, 30) / 10.0

    # 使用PIL的ImageEnhance模块中的Sharpness类来增强图像的清晰度
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)

    # 返回经过色彩、亮度、对比度和清晰度增强后的图像
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    # 定义一个名为randomGaussian的函数，它接受一个图像以及高斯噪声的均值和标准差作为参数
    # mean表示噪声的均值，默认为0.1
    # sigma表示噪声的标准差，默认为0.35

    def gaussianNoisy(im, mean=mean, sigma=sigma):
        # 定义了一个内部函数gaussianNoisy，用于在单个像素数组上添加高斯噪声
        # 它也接受均值和标准差作为参数，如果调用时不传入，则使用外部函数的默认值

        for _i in range(len(im)):
            # 遍历像素数组中的每一个像素值

            im[_i] += random.gauss(mean, sigma)
            # 对每个像素值添加服从给定均值和标准差的高斯分布的随机噪声

        return im
        # 返回添加噪声后的像素数组

    img = np.asarray(image)
    # 将输入的图像转换为NumPy数组，方便进行数值操作

    width, height = img.shape
    # 获取图像的宽度和高度，这里假设图像是二维的灰度图像
    # 如果图像是彩色图像，还需要处理额外的通道维度

    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    # 将图像数据展平为一维数组，并调用gaussianNoisy函数添加高斯噪声
    # 注意这里img[:]是创建了一个图像的副本，避免直接修改原始图像数据

    img = img.reshape([width, height])
    # 将添加噪声后的一维数组重新整形为原始的二维图像形状

    return Image.fromarray(np.uint8(img))
    # 将噪声图像数组转换为8位无符号整数格式（因为图像数据通常是这个类型）
    # 并使用PIL库的Image.fromarray方法将其转换回图像对象
    # 注意：这里假设原始图像是8位无符号整数类型，如果不是，可能需要进行适当的类型转换


# 定义一个函数randomPepper，接受一个图像img作为输入
def randomPepper(img):
    # 将输入的图像转换为NumPy数组格式，便于进行后续的数值操作
    # 当这个图像被转换为Numpy数组后，其形状（shape）会保持不变，即仍然为 (C, H, W)，Numpy数组能够直接处理这种多维数据，使得图像处理变得更加方便
    img = np.array(img)

    # 计算要添加的椒盐噪声点的数量。这里通过图像的高度和宽度的乘积的0.0015倍来确定
    # 这个比例可以根据需要调整，以产生不同密度的椒盐噪声
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])

    # 遍历要添加的噪声点数量
    for i in range(noiseNum):

        # 随机生成一个x坐标，确保坐标在图像宽度范围内
        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)
        # 随机生成一个y坐标，确保坐标在图像高度范围内

        if random.randint(0, 1) == 0:
            # 以50%的概率确定噪声点的颜色
            # 如果随机数为0，则将对应位置的像素值设置为0（黑色）
            img[randX, randY] = 0


        else:
            # 如果随机数不为0
            # 则将对应位置的像素值设置为255（白色）
            img[randX, randY] = 255

    # 将添加椒盐噪声后的NumPy数组转换回PIL图像对象，并返回
    return Image.fromarray(img)


# 定义一个名为SalObjDataset的类，继承自data.Dataset，用于加载带有标注的图像数据
class SalObjDataset(data.Dataset):

    # 初始化方法，接收图像根目录、标注根目录和训练时使用的图像大小
    def __init__(self, image_root, gt_root, trainsize):

        # 将训练时使用的图像大小保存在类的实例变量中
        self.trainsize = trainsize

        # 构造图像文件的完整路径列表
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]

        # 构造标注文件的完整路径列表，支持jpg和png两种格式
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        # self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.jpg') or f.endswith('.png')]

        # 对图像文件和标注文件路径列表进行排序，保证文件顺序的一致性
        # 这句话的核心目的是确保图像文件和标注文件的顺序是匹配的，也就是说，排序后的图像文件列表中的第一个文件应该与标注文件列表中的第一个文件相对应，第二个文件与第二个文件相对应
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        # self.edges = sorted(self.edges)

        # self.kernel = np.ones((5, 5), np.uint8)
        # 过滤图像和标注文件，确保它们的尺寸匹配 对于目标检测或语义分割等任务，标注文件（如边界框或像素级别的标签）需要与图像文件的尺寸完全对齐，以确保模型能够正确地识别和学习目标或区域
        # 过滤意味着检查数据集中的每个图像和标注文件，并移除或修正那些尺寸不匹配的文件
        # 过滤过程可能包括检查图像和标注文件的尺寸信息，比较它们是否一致，并删除或修改那些不符合要求的数据
        # 你需要过滤出这样的图像和标注文件，要么修正标注，要么从数据集中移除它们
        '''
        是的，像素级别的标注确实需要对图像中的每一个像素都要打标签,这种标注方式在计算机视觉中是一种非常精细和准确的处理方式，通常用于图像增强、噪声去除、边缘检测等任务
        具体来说，标注员需要在图像上使用工具，手动为每一个像素点打上标签，这些标签可以代表物体种类、物体分割区域等多种信息。通过这种方式，可以为许多视觉任务提供训练和测试数据的基础
        因此，像素级别的标注确实是对图像中的每一个像素都要进行标签的标注
        '''
        self.filter_files()

        # 获取过滤后的图像数量，并保存为类的实例变量
        self.size = len(self.images)

        # 定义图像预处理流程：缩放、转换为张量、归一化
        # 创建一个包含多个预处理步骤的Compose对象，用于图像预处理
        self.img_transform = transforms.Compose([
            # 将图像缩放到指定大小，这里假设self.trainsize是预定义好的目标大小
            transforms.Resize((self.trainsize, self.trainsize)),
            # 将PIL Image或NumPy ndarray转换为torch.Tensor，并归一化到[0.0, 1.0]范围
            transforms.ToTensor(),
            # 对图像进行归一化，使用给定的均值和标准差
            # 这里的均值和标准差通常是基于整个训练集的统计值，用于标准化图像数据
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        # 定义标注预处理流程：缩放、转换为张量
        # 创建一个包含多个预处理步骤的Compose对象，用于标注预处理
        self.gt_transform = transforms.Compose([
            # 将标注缩放到与图像相同的大小，确保图像和标注在尺寸上匹配
            transforms.Resize((self.trainsize, self.trainsize)),
            # 将标注数据（通常是NumPy ndarray或类似格式）转换为torch.Tensor
            # 注意：这里的标注数据可能是像素级别的（如语义分割任务中的掩码），也可能是其他形式（如边界框坐标）
            transforms.ToTensor()])

        '''
        对图像进行归一化是深度学习和机器学习中常见的预处理步骤，它对于模型训练的稳定性和性能具有重要的影响。以下是为什么需要对图像进行归一化的几个主要原因：

        尺度不变性：归一化将图像的像素值缩放到一个统一的范围内（通常是[0, 1]或[-1, 1]），这样模型在处理不同尺度的输入时能够表现得更稳定。不同的图像可能由于拍摄条件、设备差异或后期处理等原因，具有不同的亮度、对比度或整体颜色分布。归一化可以消除这些差异，使模型更关注于图像的内容而不是这些无关紧要的尺度变化

        加速收敛：归一化后的图像数据具有更好的数值特性，使得模型的优化过程（如梯度下降）更加高效。在归一化后，数据的分布更加集中，梯度计算更加稳定，这有助于模型更快地收敛到最优解

        提升模型性能：归一化可以减少模型在训练过程中的内部协变量偏移（Internal Covariate Shift），这是一种在训练神经网络时由于每层输入的分布不断变化而导致的问题。归一化可以稳定每一层的输入分布，使得模型更容易学习数据的内在表示，从而提高模型的性能。

        提高泛化能力：通过归一化，模型能够更好地处理不同分布的数据，从而提高其泛化能力。在实际应用中，测试数据往往与训练数据的分布存在差异，归一化有助于模型更好地适应这些差异

        数值稳定性：在某些深度学习框架中，归一化后的数据可以减少数值计算中的误差累积，提高计算的稳定性

        综上所述，对图像进行归一化是深度学习模型训练中的一个重要步骤，它有助于提高模型的稳定性、收敛速度和性能，同时也有助于提高模型的泛化能力
        '''

    # 定义一个类的__getitem__方法，它使得这个类的实例能够像列表或数组一样被索引
    # 这个方法通常在PyTorch的DataLoader中用于获取单个数据样本
    '''
    __getitem__ 方法是 Python 中的一个特殊方法，它允许类的实例像列表或数组那样使用索引来访问元素,当你在一个类的实例上使用方括号（[]）进行索引时，__getitem__ 方法就会被自动调用
    在深度学习的上下文中，__getitem__ 方法经常用于定义数据集类的行为，特别是在 PyTorch 的 DataLoader 中。DataLoader 负责从数据集中批量地抽取数据，并在每次迭代时调用 __getitem__ 方法来获取单个样本
    但是，__getitem__ 方法本身并不会自动运行,你需要显式地调用它，或者当你使用像 DataLoader 这样的工具时，它会在内部为你调用
    '''

    def __getitem__(self, index):

        # 使用rgb_loader方法加载指定索引的图像文件，并转换为RGB格式
        image = self.rgb_loader(self.images[index])

        # 使用binary_loader方法加载指定索引的标注文件，并转换为灰度图（L模式）
        gt = self.binary_loader(self.gts[index])

        # edge = self.binary_loader(self.edges[index])

        # edge = cv2.imread(self.edges[index], cv2.IMREAD_GRAYSCALE)
        #
        # edge = cv2.dilate(edge, self.kernel, iterations=1)
        #
        # edge = Image.fromarray(edge)

        # 随机对图像和标注进行水平翻转
        image, gt = cv_random_flip(image, gt)

        # 对图像和标注进行随机裁剪
        image, gt = randomCrop(image, gt)

        # 对图像和标注进行随机旋转
        image, gt = randomRotation(image, gt)

        # 对图像进行颜色增强，可能包括对比度、亮度等调整
        image = colorEnhance(image)

        # 注释掉的代码，表示未使用的功能，对标注应用随机高斯噪声  一般使用椒盐噪声比较好
        # gt=randomGaussian(gt)

        # 对标注应用椒盐噪声
        gt = randomPepper(gt)

        # 对图像应用之前定义的预处理流程，包括缩放、转换为张量、归一化
        image = self.img_transform(image)

        # 对标注应用之前定义的预处理流程，包括缩放和转换为张量
        gt = self.gt_transform(gt)

        # edge = self.gt_transform(edge)

        # 返回处理后的图像和标注
        return image, gt

    # 定义一个filter_files方法，用于过滤图像和标注文件，确保它们的尺寸匹配
    def filter_files(self):
        # 断言确保图像和标注文件的数量相同
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)

        # 初始化空的列表，用于存储过滤后的图像和标注文件路径
        images = []
        gts = []
        # edges = []

        # 遍历图像和标注文件路径的列表
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            # edge = Image.open(edge_path)

            # 如果图像和标注的尺寸相同，则将它们的路径添加到相应的列表中
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
                # edges.append(edge_path)

        # 将过滤后的文件路径重新赋值给类的属性
        self.images = images
        self.gts = gts
        # self.edges = edges

    # 定义一个名为rgb_loader的方法，它接收一个参数path，代表图片文件的路径
    def rgb_loader(self, path):
        # 使用'rb'模式（二进制读取模式）打开指定的图片文件
        with open(path, 'rb') as f:
            # 使用PIL库（或Pillow库）的Image模块打开文件对象f，读取其中的图片内容
            img = Image.open(f)
            # 将图片转换为RGB模式，确保图片由红、绿、蓝三个通道组成
            return img.convert('RGB')

    # 定义一个名为binary_loader的方法，它同样接收一个参数path，代表图片文件的路径
    def binary_loader(self, path):
        # 使用'rb'模式（二进制读取模式）打开指定的图片文件
        with open(path, 'rb') as f:
            # 使用PIL库（或Pillow库）的Image模块打开文件对象f，读取其中的图片内容
            img = Image.open(f)
            # 将图片转换为L模式，这是一个灰度模式，意味着图片将只包含一个通道，且每个像素的值都在0到255之间
            return img.convert('L')

    # 定义一个resize方法，该方法接收两个参数：img（图像）和gt（可能是图像对应的标签或真实值）
    def resize(self, img, gt):
        # 断言，确保传入的图像和真实值大小相同
        assert img.size == gt.size

        # 获取图像的宽度和高度
        w, h = img.size

        # 检查图像的高度或宽度是否小于预设的训练大小
        if h < self.trainsize or w < self.trainsize:

            # 如果小于训练大小，则将高度和宽度至少设置为训练大小
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)

            # 使用双线性插值法（Imgs.BILINEAR）对图像进行缩放;同样的，使用最近邻插值法（Imgs.NEAREST）对真实值进行缩放
            # 返回两个经过缩放的对象
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)

        # 如果图像大小满足或超过训练大小，则不进行缩放，直接返回原图像和真实值
        else:
            return img, gt

    # 定义一个特殊方法__len__，该方法返回数据集的样本数量
    def __len__(self):
        # 返回数据集的样本总数，这个值通常在类的初始化方法中设置
        return self.size

    '''
    特殊方法 __len__ 不会自动调用，但它会在你尝试获取对象长度时被自动调用。在 Python 中，当你使用内置的 len() 函数时，Python 解释器会查找并调用对象的 __len__ 方法来获取对象的长度
    例如，如果你有一个名为 dataset 的对象，它定义了一个 __len__ 方法，那么当你执行 len(dataset) 时，Python 会自动调用 dataset 的 __len__ 方法并返回其结果
    这在处理像列表、元组、字符串这样的内置可迭代对象时是很常见的，也适用于自定义的数据结构，如自定义的数据集类。在深度学习和机器学习的上下文中，__len__ 方法通常用于告诉框架数据集中有多少个样本
    例如，在 PyTorch 的 DataLoader 中，__len__ 方法用于确定应该执行多少个批次（batch）来遍历整个数据集。当你迭代 DataLoader 对象时，它会使用 __len__ 方法来确定迭代的总次数
    '''


# 定义一个函数get_loader，用于获取训练数据加载器（data loader）
def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=8, pin_memory=True):
    # 创建一个SalObjDataset实例，该实例将用于加载图像和对应的真实值（ground truth）
    dataset = SalObjDataset(image_root, gt_root, trainsize)

    # 使用PyTorch的data.DataLoader创建一个数据加载器实例
    # 这个加载器会负责从数据集中批量提取数据，并在训练时执行必要的操作，如打乱数据顺序
    data_loader = data.DataLoader(dataset=dataset,

                                  # 设置每个批次的大小
                                  batch_size=batchsize,

                                  # 是否在每次迭代时打乱数据顺序
                                  shuffle=shuffle,

                                  # 设置用于数据加载的子进程数量
                                  # 可以提高数据加载速度，特别是在多核CPU上
                                  num_workers=num_workers,

                                  # 是否将数据加载到固定内存地址，这可以加速数据到GPU的传输速度
                                  # 但仅当设备支持固定内存时有效
                                  pin_memory=pin_memory)
    # 返回创建好的数据加载器
    return data_loader


# test dataset and loader
# 定义一个名为test_dataset的类，用于处理测试数据集
class test_dataset:

    # 初始化函数，设置数据集的相关属性和变量
    def __init__(self, image_root, gt_root, testsize):
        # 设置测试时图像的大小
        self.testsize = testsize

        # 获取image_root目录下所有以.jpg或.png结尾的文件，并构建完整的文件路径列表
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]

        # 获取gt_root目录下所有以.jpg或.png结尾的文件，并构建完整的文件路径列表
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]

        # self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.jpg') or f.endswith('.png')]
        # 对图像和真实值列表进行排序
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        # self.edges = sorted(self.edges)

        # 定义图像变换，包括调整大小、转换为张量、归一化
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),  # 调整图像大小为testsize
            transforms.ToTensor(),  # 将图像转换为PyTorch张量
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # 对图像进行归一化

        # 定义真实值变换，这里仅转换为张量
        self.gt_transform = transforms.ToTensor()

        # 设置数据集的大小为图像列表的长度
        self.size = len(self.images)

        # 设置当前索引为0，用于在load_data方法中迭代数据集
        self.index = 0

    # 加载数据的方法，返回一张图像、对应的真实值、文件名和用于后续处理的图像
    def load_data(self):
        # 使用rgb_loader方法加载图像
        image = self.rgb_loader(self.images[self.index])

        # 对图像应用定义的变换，并增加一个维度以匹配模型输入的batch维度
        image = self.transform(image).unsqueeze(0)

        # 使用binary_loader方法加载真实值
        gt = self.binary_loader(self.gts[self.index])

        # 获取文件名
        name = self.gts[self.index].split('/')[-1]

        # 再次加载图像，但这次是为了调整其大小以匹配真实值的大小，用于后续处理
        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)

        # 处理文件名，如果是以.jpg结尾的，则去掉后缀并重新加上
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.jpg'

        # 更新索引，如果到达数据集末尾，则循环回到开始
        self.index += 1
        self.index = self.index % self.size

        # 返回处理后的图像、真实值、文件名和用于后续处理的图像
        return image, gt, name, np.array(image_for_post)

    # 加载RGB图像的方法
    def rgb_loader(self, path):
        # 打开文件并读取图像
        with open(path, 'rb') as f:
            img = Image.open(f)

            # 将图像转换为RGB模式并返回
            return img.convert('RGB')

    # 加载二值图像（真实值）的方法
    def binary_loader(self, path):
        # 打开文件并读取图像
        with open(path, 'rb') as f:
            img = Image.open(f)

            # 将图像转换为灰度模式（二值）并返回
            return img.convert('L')

    # 实现特殊方法__len__，返回数据集的大小
    def __len__(self):
        return self.size
