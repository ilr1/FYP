# -*- coding: utf-8 -*-
#
# author: Ozan Caglayan


from collections import OrderedDict

import torch
from torchvision import models
from torchvision.models.vgg import cfgs as vgg_cfg


class Flatten(torch.nn.Module):
    """A flatten module to squeeze single dimensions."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def get_vgg_names(config, batch_norm=False):
    """Returns meaningful layer names for VGG variants."""
    names = []

    # Counters for layer naming
    n_block, n_conv = 1, 1

    for v in config:
        if v == 'M':
            names.append('pool%d' % n_block)
            n_block += 1
            n_conv = 1
        else:
            conv_name = 'conv%d_%d' % (n_block, n_conv)
            names.append(conv_name)
            if batch_norm:
                names.append('%s+bn' % conv_name)
                names.append('%s+bn+relu' % conv_name)
            else:
                names.append('%s+relu' % conv_name)

            n_conv += 1

    return names


# Mapping from torchvision's internal layer names to our naming scheme
resnet_layers = {
    'conv1': 'conv1',
    'bn1': 'bn1',
    'relu': 'relu',
    'maxpool': 'maxpool',
    'layer1': 'res2c_relu',     # only differences are here
    'layer2': 'res3d_relu',     # only differences are here
    'layer3': 'res4f_relu',     # only differences are here
    'layer4': 'res5c_relu',     # only differences are here
    'avgpool': 'avgpool',
    'fc': 'fc',  # You'll never want to extract features from 'fc'!
}


class ImageEncoder(object):
    CFG_MAP = {
        # ResNet variants
        'resnet18': resnet_layers,
        'resnet34': resnet_layers,
        'resnet50': resnet_layers,
        'resnet101': resnet_layers,
        'resnet152': resnet_layers,
        # Plain VGGs
        'vgg11': get_vgg_names(vgg_cfg['A']),
        'vgg13': get_vgg_names(vgg_cfg['B']),
        'vgg16': get_vgg_names(vgg_cfg['D']),
        'vgg19': get_vgg_names(vgg_cfg['E']),
        # Batchnorm VGGs
        'vgg11_bn': get_vgg_names(vgg_cfg['A'], batch_norm=True),
        'vgg13_bn': get_vgg_names(vgg_cfg['B'], batch_norm=True),
        'vgg16_bn': get_vgg_names(vgg_cfg['D'], batch_norm=True),
        'vgg19_bn': get_vgg_names(vgg_cfg['E'], batch_norm=True),
    }

    def __init__(self, cnn_type, pretrained=True):
        self.pretrained = pretrained
        self.cnn_type = cnn_type
        self.cnn = None

        assert self.cnn_type in self.CFG_MAP, \
            "{} not supported by ImageEncoder".format(self.cnn_type)

        # Load vanilla CNN instance
        self._base_cnn = getattr(models, self.cnn_type)(pretrained=pretrained)

    def get_base_layers(self):
        """Returns possible extraction points for the requested CNN."""
        layers = self.CFG_MAP[self.cnn_type]
        if isinstance(layers, list):
            return layers
        elif isinstance(layers, dict):
            return list(layers.values())

    def setup(self, layer):
        """Truncates the requested CNN until `layer`, `layer` included. The
        final instance is stored under `self.cnn` and can be obtained with
        the `.get()` method. The instance will have `requires_grad=False`
        for all parameters by default. You can use `set_requires_grad()`
        to selectively or completely enable `requires_grad` at layer-level.

        If layer == 'penultimate' and CNN type is VGG, whole CNN except
        the last classification layer will be returned.

        Arguments:
            layer(str): A layer name for VGG/ResNet. Possible truncation
                points can be seen using the method `get_base_layers()`.
        """

        layers = OrderedDict()
        self.layer_map = self.CFG_MAP[self.cnn_type]

        if self.cnn_type.startswith('vgg'):
            assert len(self._base_cnn.features) == len(self.layer_map)

            # There's no named modules inside VGG, all integers
            for module, params in zip(self.layer_map, self._base_cnn.features):
                layers[module] = params
                # 'penultimate' takes all conv layers by default
                if layer != 'penultimate' and module == layer:
                    break

            if layer == 'penultimate':
                # Add flatten layer
                layers['flatten'] = Flatten()

                # Exclude final classification layer
                for i in range(len(self._base_cnn.classifier) - 1):
                    mod = self._base_cnn.classifier[i]
                    name = "{}{}".format(mod.__class__.__name__, i)
                    layers[name] = mod

        elif self.cnn_type.startswith('resnet'):
            assert layer in self.layer_map.values(), \
                "The given layer {} is not known.".format(layer)
            for module, params in self._base_cnn.named_children():
                # Add the layer with our naming scheme
                layers[self.layer_map[module]] = params
                # If we've hit the extraction point, break the loop
                if self.layer_map[module] == layer:
                    break

        self.cnn = torch.nn.Sequential(layers)

        # Disable requires_grad by default
        self.set_requires_grad(False)

    def set_requires_grad(self, value=False, layers='all'):
        """Sets requires_grad for the given layer(s).

        Arguments:
            layers(str): A string or comma separated list of strings or
                a range i.e. 'layer_from:layer_to'
                for which the requires_grad attribute will be set according
                to `value`. If `all`, all layers will be affected.

        Examples:
            # Requires grad only for res4f_relu
            set_requires_grad(val, 'res4f_relu')
            # Requires grad only for res4f_relu and res5c_relu
            set_requires_grad(val, 'res4f_relu,res5c_relu')
            # Requires grad for all layers between [res2c_relu, res5c_relu]
            set_requires_grad(val, 'res2c_relu:res5c_relu')
        """
        assert self.cnn is not None, "ImageEncoder.setup() is not called"
        assert value in (True, False), "value should be a boolean."

        if layers == 'all':
            for name, param in self.cnn.named_parameters():
                param.requires_grad = value
        else:
            named_children = list(self.cnn.named_children())
            in_range = None
            if ':' in layers:
                layer_begin, layer_end = layers.split(':')
                in_range = False
                if not layer_begin:
                    # from beginning upto layer_end
                    layer_begin = named_children[0][0]
                elif not layer_end:
                    # from layer_begin upto end
                    layer_end = named_children[-1][0]

            for name, module in named_children:
                if in_range is not None:
                    # range given
                    in_range = in_range or name == layer_begin
                    if in_range:
                        for param in module.parameters():
                            param.requires_grad = value
                        in_range = (name != layer_end)
                else:
                    # list of layer names given
                    if name in layers.split(','):
                        for param in module.parameters():
                            param.requires_grad = value

    def get(self):
        """Returns the configured CNN instance."""
        return self.cnn