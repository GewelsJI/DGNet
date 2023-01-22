import jittor as jt
from jittor import nn

from lib_jittor.lib.utils import round_filters, round_repeats, drop_connect, get_same_padding_conv2d, get_model_params, \
    efficientnet_params, load_pretrained_weights, Swish, MemoryEfficientSwish, calculate_output_image_size

VALID_MODELS = (
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5',
    'efficientnet-b6', 'efficientnet-b7', 'efficientnet-b8', 'efficientnet-l2')


class MBConvBlock(nn.Module):
    'Mobile Inverted Residual Bottleneck Block.\n\n    Args:\n        block_args (namedtuple): BlockArgs, defined in utils.py.\n        global_params (namedtuple): GlobalParam, defined in utils.py.\n        image_size (tuple or list): [image_height, image_width].\n\n    References:\n        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)\n        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)\n        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)\n    '

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = (1 - global_params.batch_norm_momentum)
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = ((self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1))
        self.id_skip = block_args.id_skip
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        inp = self._block_args.input_filters
        oup = (self._block_args.input_filters * self._block_args.expand_ratio)
        if (self._block_args.expand_ratio != 1):
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(inp, oup, 1, bias=False)
            self._bn0 = nn.BatchNorm(oup, momentum=self._bn_mom, eps=self._bn_eps)
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(oup, oup, k, groups=oup, stride=s, bias=False)
        self._bn1 = nn.BatchNorm(oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int((self._block_args.input_filters * self._block_args.se_ratio)))
            self._se_reduce = Conv2d(oup, num_squeezed_channels, 1)
            self._se_expand = Conv2d(num_squeezed_channels, oup, 1)

        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(oup, final_oup, 1, bias=False)
        self._bn2 = nn.BatchNorm(final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def execute(self, inputs, drop_connect_rate=None):
        "MBConvBlock's forward function.\n\n        Args:\n            inputs (tensor): Input tensor.\n            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).\n\n        Returns:\n            Output of this block after processing.\n        "
        x = inputs
        if (self._block_args.expand_ratio != 1):
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)
        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)
        if self.has_se:
            x_squeezed = self._avg_pooling(x)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = (jt.sigmoid(x_squeezed) * x)
        x = self._project_conv(x)
        x = self._bn2(x)
        (input_filters, output_filters) = (self._block_args.input_filters, self._block_args.output_filters)
        if (self.id_skip and (self._block_args.stride == 1) and (input_filters == output_filters)):
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.is_training())
            x = (x + inputs)
        return x

    def set_swish(self, memory_efficient=True):
        'Sets swish function as memory efficient (for training) or standard (for export).\n\n        Args:\n            memory_efficient (bool): Whether to use memory-efficient version of swish.\n        '
        self._swish = (MemoryEfficientSwish() if memory_efficient else Swish())


class EfficientNet(nn.Module):
    "EfficientNet model.\n       Most easily loaded with the .from_name or .from_pretrained methods.\n\n    Args:\n        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.\n        global_params (namedtuple): A set of GlobalParams shared between blocks.\n\n    References:\n        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)\n\n    Example:\n        >>> import torch\n        >>> from efficientnet.model1 import EfficientNet\n        >>> inputs = torch.rand(1, 3, 224, 224)\n        >>> model = EfficientNet.from_pretrained('efficientnet-b0')\n        >>> model.eval()\n        >>> outputs = model(inputs)\n    "

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert (len(blocks_args) > 0), 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        bn_mom = (1 - self._global_params.batch_norm_momentum)
        bn_eps = self._global_params.batch_norm_epsilon
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        in_channels = 3
        out_channels = round_filters(32, self._global_params)
        self._conv_stem = Conv2d(in_channels, out_channels, 3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm(out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, 2)
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:
            block_args = block_args._replace(input_filters=round_filters(block_args.input_filters, self._global_params),
                                             output_filters=round_filters(block_args.output_filters,
                                                                          self._global_params),
                                             num_repeat=round_repeats(block_args.num_repeat, self._global_params))
            self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if (block_args.num_repeat > 1):
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range((block_args.num_repeat - 1)):
                self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
        in_channels = block_args.output_filters
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        self._conv_head = Conv2d(in_channels, out_channels, 1, bias=False)
        self._bn1 = nn.BatchNorm(out_channels, momentum=bn_mom, eps=bn_eps)

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        if self._global_params.include_top:
            self._dropout = nn.Dropout(self._global_params.dropout_rate)
            self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        'Sets swish function as memory efficient (for training) or standard (for export).\n\n        Args:\n            memory_efficient (bool): Whether to use memory-efficient version of swish.\n        '
        self._swish = (MemoryEfficientSwish() if memory_efficient else Swish())
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_endpoints(self, inputs):
        "Use convolution layer to extract features\n        from reduction levels i in [1, 2, 3, 4, 5].\n\n        Args:\n            inputs (tensor): Input tensor.\n\n        Returns:\n            Dictionary of last intermediate features\n            with reduction levels i in [1, 2, 3, 4, 5].\n            Example:\n                >>> import torch\n                >>> from efficientnet.model1 import EfficientNet\n                >>> inputs = torch.rand(1, 3, 224, 224)\n                >>> model = EfficientNet.from_pretrained('efficientnet-b0')\n                >>> endpoints = model.extract_endpoints(inputs)\n                >>> print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])\n                >>> print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])\n                >>> print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])\n                >>> print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])\n                >>> print(endpoints['reduction_5'].shape)  # torch.Size([1, 320, 7, 7])\n                >>> print(endpoints['reduction_6'].shape)  # torch.Size([1, 1280, 7, 7])\n        "
        endpoints = dict()
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x
        for (idx, block) in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= (float(idx) / len(self._blocks))
            x = block(x, drop_connect_rate=drop_connect_rate)
            if (prev_x.shape[2] > x.shape[2]):
                endpoints['reduction_{}'.format((len(endpoints) + 1))] = prev_x
            elif (idx == (len(self._blocks) - 1)):
                endpoints['reduction_{}'.format((len(endpoints) + 1))] = x
            prev_x = x
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints['reduction_{}'.format((len(endpoints) + 1))] = x
        return endpoints

    def extract_endpoints_dual(self, inputs, grad_feats):
        "Use convolution layer to extract features\n        from reduction levels i in [1, 2, 3, 4, 5].\n\n        Args:\n            inputs (tensor): Input tensor.\n\n        Returns:\n            Dictionary of last intermediate features\n            with reduction levels i in [1, 2, 3, 4, 5].\n            Example:\n                >>> import torch\n                >>> from efficientnet.model1 import EfficientNet\n                >>> inputs = torch.rand(1, 3, 224, 224)\n                >>> model = EfficientNet.from_pretrained('efficientnet-b0')\n                >>> endpoints = model.extract_endpoints(inputs)\n                >>> print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])\n                >>> print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])\n                >>> print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])\n                >>> print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])\n                >>> print(endpoints['reduction_5'].shape)  # torch.Size([1, 320, 7, 7])\n                >>> print(endpoints['reduction_6'].shape)  # torch.Size([1, 1280, 7, 7])\n        "
        endpoints = dict()
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x
        for (idx, block) in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= (float(idx) / len(self._blocks))
            if (idx < 2):
                x = (x + grad_feats[idx])
            x = block(x, drop_connect_rate=drop_connect_rate)
            if (prev_x.shape[2] > x.shape[2]):
                endpoints['reduction_{}'.format((len(endpoints) + 1))] = prev_x
            elif (idx == (len(self._blocks) - 1)):
                endpoints['reduction_{}'.format((len(endpoints) + 1))] = x
            prev_x = x
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints['reduction_{}'.format((len(endpoints) + 1))] = x
        return endpoints

    def extract_features(self, inputs):
        'use convolution layer to extract feature .\n\n        Args:\n            inputs (tensor): Input tensor.\n\n        Returns:\n            Output of the final convolution\n            layer in the efficientnet model.\n        '
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        for (idx, block) in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= (float(idx) / len(self._blocks))
            x = block(x, drop_connect_rate=drop_connect_rate)
        x = self._swish(self._bn1(self._conv_head(x)))
        return x

    def execute(self, inputs):
        "EfficientNet's forward function.\n           Calls extract_features to extract features, applies final linear layer, and returns logits.\n\n        Args:\n            inputs (tensor): Input tensor.\n\n        Returns:\n            Output of this model after processing.\n        "
        x = self.extract_features(inputs)
        x = self._avg_pooling(x)
        if self._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self._dropout(x)
            x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, in_channels=3, **override_params):
        "Create an efficientnet model according to name.\n\n        Args:\n            model_name (str): Name for efficientnet.\n            in_channels (int): Input data's channel number.\n            override_params (other key word params):\n                Params to override model's global_params.\n                Optional key:\n                    'width_coefficient', 'depth_coefficient',\n                    'image_size', 'dropout_rate',\n                    'num_classes', 'batch_norm_momentum',\n                    'batch_norm_epsilon', 'drop_connect_rate',\n                    'depth_divisor', 'min_depth'\n\n        Returns:\n            An efficientnet model.\n        "
        cls._check_model_name_is_valid(model_name)
        (blocks_args, global_params) = get_model_params(model_name, override_params)
        model = cls(blocks_args, global_params)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def from_pretrained(cls, model_name, weights_path=None, advprop=False, in_channels=3, num_classes=1000,
                        **override_params):
        "Create an efficientnet model according to name.\n\n        Args:\n            model_name (str): Name for efficientnet.\n            weights_path (None or str):\n                str: path to pretrained weights file on the local disk.\n                None: use pretrained weights downloaded from the Internet.\n            advprop (bool):\n                Whether to load pretrained weights\n                trained with advprop (valid when weights_path is None).\n            in_channels (int): Input data's channel number.\n            num_classes (int):\n                Number of categories for classification.\n                It controls the output size for final linear layer.\n            override_params (other key word params):\n                Params to override model's global_params.\n                Optional key:\n                    'width_coefficient', 'depth_coefficient',\n                    'image_size', 'dropout_rate',\n                    'batch_norm_momentum',\n                    'batch_norm_epsilon', 'drop_connect_rate',\n                    'depth_divisor', 'min_depth'\n\n        Returns:\n            A pretrained efficientnet model.\n        "
        model = cls.from_name(model_name, num_classes=num_classes, **override_params)
        load_pretrained_weights(model, model_name, weights_path=weights_path, load_fc=(num_classes == 1000),
                                advprop=advprop)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        'Get the input image size for a given efficientnet model.\n\n        Args:\n            model_name (str): Name for efficientnet.\n\n        Returns:\n            Input image size (resolution).\n        '
        cls._check_model_name_is_valid(model_name)
        (_, _, res, _) = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        'Validates model name.\n\n        Args:\n            model_name (str): Name for efficientnet.\n\n        Returns:\n            bool: Is a valid name or not.\n        '
        if (model_name not in VALID_MODELS):
            raise ValueError(('model_name should be one of: ' + ', '.join(VALID_MODELS)))

    def _change_in_channels(self, in_channels):
        "Adjust model's first convolution layer to in_channels, if in_channels not equals 3.\n\n        Args:\n            in_channels (int): Input data's channel number.\n        "
        if (in_channels != 3):
            Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv2d(in_channels, out_channels, 3, stride=2, bias=False)