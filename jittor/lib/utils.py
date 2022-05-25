import jittor as jt
from jittor import nn

import re
import math
import collections
from functools import partial

GlobalParams = collections.namedtuple('GlobalParams',
                                      ['width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate',
                                       'num_classes', 'batch_norm_momentum', 'batch_norm_epsilon', 'drop_connect_rate',
                                       'depth_divisor', 'min_depth', 'include_top'])
BlockArgs = collections.namedtuple('BlockArgs', ['num_repeat', 'kernel_size', 'stride', 'expand_ratio', 'input_filters',
                                                 'output_filters', 'se_ratio', 'id_skip'])
GlobalParams.__new__.__defaults__ = ((None,) * len(GlobalParams._fields))
BlockArgs.__new__.__defaults__ = ((None,) * len(BlockArgs._fields))
if hasattr(nn, 'SiLU'):
    Swish = nn.SiLU
else:

    class Swish(nn.Module):

        def execute(self, x):
            return (x * jt.sigmoid(x))


class SwishImplementation(jt.Function):

    def execute(self, i):
        self.ctx = i
        result = (i * jt.sigmoid(i))
        return result

    def backward(self, grad_output):
        i = self.ctx
        sigmoid_i = jt.sigmoid(i)
        return (grad_output * (sigmoid_i * (1 + (i * (1 - sigmoid_i)))))


class MemoryEfficientSwish(nn.Module):

    def execute(self, x):
        return SwishImplementation.apply(x)


def round_filters(filters, global_params):
    'Calculate and round number of filters based on width multiplier.\n       Use width_coefficient, depth_divisor and min_depth of global_params.\n\n    Args:\n        filters (int): Filters number to be calculated.\n        global_params (namedtuple): Global params of the model.\n\n    Returns:\n        new_filters: New filters number after calculating.\n    '
    multiplier = global_params.width_coefficient
    if (not multiplier):
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = (min_depth or divisor)
    new_filters = max(min_depth, ((int((filters + (divisor / 2))) // divisor) * divisor))
    if (new_filters < (0.9 * filters)):
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    "Calculate module's repeat number of a block based on depth multiplier.\n       Use depth_coefficient of global_params.\n\n    Args:\n        repeats (int): num_repeat to be calculated.\n        global_params (namedtuple): Global params of the model.\n\n    Returns:\n        new repeat: New repeat number after calculating.\n    "
    multiplier = global_params.depth_coefficient
    if (not multiplier):
        return repeats
    return int(math.ceil((multiplier * repeats)))


def drop_connect(inputs, p, training):
    'Drop connect.\n\n    Args:\n        input (tensor: BCWH): Input of this structure.\n        p (float: 0.0~1.0): Probability of drop connection.\n        training (bool): The running mode.\n\n    Returns:\n        output: Output after drop connection.\n    '
    assert (0 <= p <= 1), 'p must be in range of [0,1]'
    if (not training):
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = (1 - p)
    random_tensor = keep_prob
    random_tensor += jt.rand([batch_size, 1, 1, 1], dtype=inputs.dtype)
    binary_tensor = jt.floor(random_tensor)
    output = ((inputs / keep_prob) * binary_tensor)
    return output


def get_width_and_height_from_size(x):
    'Obtain height and width from x.\n\n    Args:\n        x (int, tuple or list): Data size.\n\n    Returns:\n        size: A tuple or list (H,W).\n    '
    if isinstance(x, int):
        return (x, x)
    if (isinstance(x, list) or isinstance(x, tuple)):
        return x
    else:
        raise TypeError()


def calculate_output_image_size(input_image_size, stride):
    "Calculates the output image size when using Conv2dSamePadding with a stride.\n       Necessary for static padding. Thanks to mannatsingh for pointing this out.\n\n    Args:\n        input_image_size (int, tuple or list): Size of input image.\n        stride (int, tuple or list): Conv2d operation's stride.\n\n    Returns:\n        output_image_size: A list [H,W].\n    "
    if (input_image_size is None):
        return None
    (image_height, image_width) = get_width_and_height_from_size(input_image_size)
    stride = (stride if isinstance(stride, int) else stride[0])
    image_height = int(math.ceil((image_height / stride)))
    image_width = int(math.ceil((image_width / stride)))
    return [image_height, image_width]


def get_same_padding_conv2d(image_size=None):
    'Chooses static padding if you have specified an image size, and dynamic padding otherwise.\n       Static padding is necessary for ONNX exporting of models.\n\n    Args:\n        image_size (int or tuple): Size of the image.\n\n    Returns:\n        Conv2dDynamicSamePadding or Conv2dStaticSamePadding.\n    '
    if (image_size is None):
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class Conv2dDynamicSamePadding(nn.Conv):
    '2D Convolutions like TensorFlow, for a dynamic image size.\n       The padding is operated in forward function by calculating dynamically.\n    '

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = (self.stride if (len(self.stride) == 2) else ([self.stride[0]] * 2))

    def execute(self, x):
        (ih, iw) = x.shape[(- 2):]
        (kh, kw) = self.weight.shape[(- 2):]
        (sh, sw) = self.stride
        (oh, ow) = (math.ceil((ih / sh)), math.ceil((iw / sw)))
        pad_h = max((((((oh - 1) * self.stride[0]) + ((kh - 1) * self.dilation[0])) + 1) - ih), 0)
        pad_w = max((((((ow - 1) * self.stride[1]) + ((kw - 1) * self.dilation[1])) + 1) - iw), 0)
        if ((pad_h > 0) or (pad_w > 0)):
            x = nn.pad(x, [(pad_w // 2), (pad_w - (pad_w // 2)), (pad_h // 2), (pad_h - (pad_h // 2))])
        return nn.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dStaticSamePadding(nn.Conv):
    "2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.\n       The padding mudule is calculated in construction function, then used in forward.\n    "

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)

        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        assert (image_size is not None)
        (ih, iw) = ((image_size, image_size) if isinstance(image_size, int) else image_size)
        (kh, kw) = self.weight.shape[(- 2):]
        (sh, sw) = self.stride
        sh = sh[0] if isinstance(sh, list) else sh
        sw = sw[0] if isinstance(sw, list) else sw
        self.stride = (sh, sw)

        (oh, ow) = (math.ceil((int(ih) / sh)), math.ceil((iw / sw)))

        pad_h = max((((((oh - 1) * self.stride[0]) + ((kh - 1) * self.dilation[0])) + 1) - ih), 0)
        pad_w = max((((((ow - 1) * self.stride[1]) + ((kw - 1) * self.dilation[1])) + 1) - iw), 0)
        if ((pad_h > 0) or (pad_w > 0)):
            self.static_padding = nn.ZeroPad2d(
                ((pad_w // 2), (pad_w - (pad_w // 2)), (pad_h // 2), (pad_h - (pad_h // 2))))
        else:
            self.static_padding = nn.Identity()

    def execute(self, x):
        x = self.static_padding(x)
        x = nn.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


def get_same_padding_maxPool2d(image_size=None):
    'Chooses static padding if you have specified an image size, and dynamic padding otherwise.\n       Static padding is necessary for ONNX exporting of models.\n\n    Args:\n        image_size (int or tuple): Size of the image.\n\n    Returns:\n        MaxPool2dDynamicSamePadding or MaxPool2dStaticSamePadding.\n    '
    if (image_size is None):
        return MaxPool2dDynamicSamePadding
    else:
        return partial(MaxPool2dStaticSamePadding, image_size=image_size)


class MaxPool2dDynamicSamePadding(nn.Pool):
    "2D MaxPooling like TensorFlow's 'SAME' mode, with a dynamic image size.\n       The padding is operated in forward function by calculating dynamically.\n    "

    def __init__(self, kernel_size, stride, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.stride = (([self.stride] * 2) if isinstance(self.stride, int) else self.stride)
        self.kernel_size = (([self.kernel_size] * 2) if isinstance(self.kernel_size, int) else self.kernel_size)
        self.dilation = (([self.dilation] * 2) if isinstance(self.dilation, int) else self.dilation)

    def execute(self, x):
        (ih, iw) = x.shape[(- 2):]
        (kh, kw) = self.kernel_size
        (sh, sw) = self.stride
        (oh, ow) = (math.ceil((ih / sh)), math.ceil((iw / sw)))
        pad_h = max((((((oh - 1) * self.stride[0]) + ((kh - 1) * self.dilation[0])) + 1) - ih), 0)
        pad_w = max((((((ow - 1) * self.stride[1]) + ((kw - 1) * self.dilation[1])) + 1) - iw), 0)
        if ((pad_h > 0) or (pad_w > 0)):
            x = nn.pad(x, [(pad_w // 2), (pad_w - (pad_w // 2)), (pad_h // 2), (pad_h - (pad_h // 2))])
        return nn.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)


class MaxPool2dStaticSamePadding(nn.Pool):
    "2D MaxPooling like TensorFlow's 'SAME' mode, with the given input image size.\n       The padding mudule is calculated in construction function, then used in forward.\n    "

    def __init__(self, kernel_size, stride, image_size=None, **kwargs):
        super().__init__(kernel_size, stride, **kwargs)
        self.stride = (([self.stride] * 2) if isinstance(self.stride, int) else self.stride)
        self.kernel_size = (([self.kernel_size] * 2) if isinstance(self.kernel_size, int) else self.kernel_size)
        self.dilation = (([self.dilation] * 2) if isinstance(self.dilation, int) else self.dilation)
        assert (image_size is not None)
        (ih, iw) = ((image_size, image_size) if isinstance(image_size, int) else image_size)
        (kh, kw) = self.kernel_size
        (sh, sw) = self.stride
        sh = sh[0] if isinstance(sh, list) else sh
        sw = sw[0] if isinstance(sw, list) else sw

        (oh, ow) = (math.ceil((ih / sh)), math.ceil((iw / sw)))
        pad_h = max((((((oh - 1) * self.stride[0]) + ((kh - 1) * self.dilation[0])) + 1) - ih), 0)
        pad_w = max((((((ow - 1) * self.stride[1]) + ((kw - 1) * self.dilation[1])) + 1) - iw), 0)
        if ((pad_h > 0) or (pad_w > 0)):
            self.static_padding = nn.ZeroPad2d(
                ((pad_w // 2), (pad_w - (pad_w // 2)), (pad_h // 2), (pad_h - (pad_h // 2))))
        else:
            self.static_padding = nn.Identity()

    def execute(self, x):
        x = self.static_padding(x)
        x = nn.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode,
                         self.return_indices)
        return x


class BlockDecoder(object):
    'Block Decoder for readability,\n       straight from the official TensorFlow repository.\n    '

    @staticmethod
    def _decode_block_string(block_string):
        "Get a block through a string notation of arguments.\n\n        Args:\n            block_string (str): A string notation of arguments.\n                                Examples: 'r1_k3_s11_e1_i32_o16_se0.25_noskip'.\n\n        Returns:\n            BlockArgs: The namedtuple defined at the top of this file.\n        "
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split('(\\d.*)', op)
            if (len(splits) >= 2):
                (key, value) = splits[:2]
                options[key] = value
        assert ((('s' in options) and (len(options['s']) == 1)) or (
                (len(options['s']) == 2) and (options['s'][0] == options['s'][1])))
        return BlockArgs(num_repeat=int(options['r']), kernel_size=int(options['k']), stride=[int(options['s'][0])],
                         expand_ratio=int(options['e']), input_filters=int(options['i']),
                         output_filters=int(options['o']),
                         se_ratio=(float(options['se']) if ('se' in options) else None),
                         id_skip=('noskip' not in block_string))

    @staticmethod
    def _encode_block_string(block):
        'Encode a block to a string.\n\n        Args:\n            block (namedtuple): A BlockArgs type argument.\n\n        Returns:\n            block_string: A String form of BlockArgs.\n        '
        args = [('r%d' % block.num_repeat), ('k%d' % block.kernel_size),
                ('s%d%d' % (block.strides[0], block.strides[1])), ('e%s' % block.expand_ratio),
                ('i%d' % block.input_filters), ('o%d' % block.output_filters)]
        if (0 < block.se_ratio <= 1):
            args.append(('se%s' % block.se_ratio))
        if (block.id_skip is False):
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        'Decode a list of string notations to specify blocks inside the network.\n\n        Args:\n            string_list (list[str]): A list of strings, each string is a notation of block.\n\n        Returns:\n            blocks_args: A list of BlockArgs namedtuples of block args.\n        '
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        'Encode a list of BlockArgs to a list of strings.\n\n        Args:\n            blocks_args (list[namedtuples]): A list of BlockArgs namedtuples of block args.\n\n        Returns:\n            block_strings: A list of strings, each string is a notation of block.\n        '
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def efficientnet_params(model_name):
    'Map EfficientNet model name to parameter coefficients.\n\n    Args:\n        model_name (str): Model name to be queried.\n\n    Returns:\n        params_dict[model_name]: A (width,depth,res,dropout) tuple.\n    '
    params_dict = {'efficientnet-b0': (1.0, 1.0, 224, 0.2), 'efficientnet-b1': (1.0, 1.1, 240, 0.2),
                   'efficientnet-b2': (1.1, 1.2, 260, 0.3), 'efficientnet-b3': (1.2, 1.4, 300, 0.3),
                   'efficientnet-b4': (1.4, 1.8, 380, 0.4), 'efficientnet-b5': (1.6, 2.2, 456, 0.4),
                   'efficientnet-b6': (1.8, 2.6, 528, 0.5), 'efficientnet-b7': (2.0, 3.1, 600, 0.5),
                   'efficientnet-b8': (2.2, 3.6, 672, 0.5), 'efficientnet-l2': (4.3, 5.3, 800, 0.5)}
    return params_dict[model_name]


def efficientnet(width_coefficient=None, depth_coefficient=None, image_size=None, dropout_rate=0.2,
                 drop_connect_rate=0.2, num_classes=1000, include_top=True):
    'Create BlockArgs and GlobalParams for efficientnet model.\n\n    Args:\n        width_coefficient (float)\n        depth_coefficient (float)\n        image_size (int)\n        dropout_rate (float)\n        drop_connect_rate (float)\n        num_classes (int)\n\n        Meaning as the name suggests.\n\n    Returns:\n        blocks_args, global_params.\n    '
    blocks_args = ['r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25', 'r2_k5_s22_e6_i24_o40_se0.25',
                   'r3_k3_s22_e6_i40_o80_se0.25', 'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
                   'r1_k3_s11_e6_i192_o320_se0.25']
    blocks_args = BlockDecoder.decode(blocks_args)
    global_params = GlobalParams(width_coefficient=width_coefficient, depth_coefficient=depth_coefficient,
                                 image_size=image_size, dropout_rate=dropout_rate, num_classes=num_classes,
                                 batch_norm_momentum=0.99, batch_norm_epsilon=0.001,
                                 drop_connect_rate=drop_connect_rate, depth_divisor=8, min_depth=None,
                                 include_top=include_top)
    return (blocks_args, global_params)


def get_model_params(model_name, override_params):
    "Get the block args and global params for a given model name.\n\n    Args:\n        model_name (str): Model's name.\n        override_params (dict): A dict to modify global_params.\n\n    Returns:\n        blocks_args, global_params\n    "
    if model_name.startswith('efficientnet'):
        (w, d, s, p) = efficientnet_params(model_name)
        (blocks_args, global_params) = efficientnet(width_coefficient=w, depth_coefficient=d, dropout_rate=p,
                                                    image_size=s)
    else:
        raise NotImplementedError('model name is not pre-defined: {}'.format(model_name))
    if override_params:
        global_params = global_params._replace(**override_params)
    return (blocks_args, global_params)


url_map = {
    'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth',
    'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth',
    'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth',
    'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth'}
url_map_advprop = {
    'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b0-b64d5a18.pth',
    'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b1-0f3ce85a.pth',
    'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b2-6e9d97e5.pth',
    'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b3-cdd7c0f4.pth',
    'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b4-44fb3a87.pth',
    'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b5-86493f6b.pth',
    'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b6-ac80338e.pth',
    'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b7-4652b6dd.pth',
    'efficientnet-b8': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b8-22a8fe65.pth'}


def load_pretrained_weights(model, model_name, weights_path=None, load_fc=True, advprop=False, verbose=True):
    'Loads pretrained weights from weights path or download using url.\n\n    Args:\n        model (Module): The whole model of efficientnet.\n        model_name (str): Model name of efficientnet.\n        weights_path (None or str):\n            str: path to pretrained weights file on the local disk.\n            None: use pretrained weights downloaded from the Internet.\n        load_fc (bool): Whether to load pretrained weights for fc layer at the end of the model.\n        advprop (bool): Whether to load pretrained weights\n                        trained with advprop (valid when weights_path is None).\n    '
    if isinstance(weights_path, str):
        state_dict = jt.load(weights_path)
    else:
        url_map_ = (url_map_advprop if advprop else url_map)
        state_dict = jt.load(url_map_[model_name])
    if load_fc:
        ret = model.load_parameters(state_dict, strict=False)
        assert (not ret.missing_keys), 'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
    else:
        state_dict.pop('_fc.weight')
        state_dict.pop('_fc.bias')
        ret = model.load_parameters(state_dict, strict=False)
        assert (set(ret.missing_keys) == set(
            ['_fc.weight', '_fc.bias'])), 'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
    assert (not ret.unexpected_keys), 'Missing keys when loading pretrained weights: {}'.format(ret.unexpected_keys)
    if verbose:
        print('Loaded pretrained weights for {}'.format(model_name))
