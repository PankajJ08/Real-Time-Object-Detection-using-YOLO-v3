import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utility import *


def parse_cfg(cfgfile):
    """
    Take a configuration file as a input and return a dictionary containing
    all the blocks as an output.
    """
    with open(cfgfile, "r") as file:
        lines = [x.rstrip().lstrip() for x in file.read().split("\n")
                 if len(x) > 0 and x[0] != "#"]

    blocks = []             # storing block in a this list
    temp = {}               # storing informations of a block in this dict

    for line in lines:
        if line.startswith("["):            # block starting
                if len(temp) != 0:
                    blocks.append(temp)
                    temp = {}
                temp["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            temp[key.rstrip()] = value.lstrip()
    blocks.append(temp)

    return blocks


def create_modules(blocks):
    network_info = blocks[0]
    module_list = nn.ModuleList()
    inp_filter = 3
    output_filter = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        if x["type"] == "convolutional":
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            padding = int(x["pad"])
            activation = x["activation"]

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            conv = nn.Conv2d(inp_filter, filters, kernel_size,
                             stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            if batch_normalize:
                module.add_module("batch_norm_{0}".format(index),
                                  nn.BatchNorm2d(filters))

            if activation == "leaky":
                module.add_module("leaky_{0}".format(index),
                                  nn.LeakyReLU(0.1, True))

        elif x["type"] == "upsample":
            stride = int(x["stride"])
            module.add_module("upsample_{0}".format(index),
                              nn.Upsample(scale_factor=2, mode="nearest"))

        elif x["type"] == "route":
            x["layers"] = x["layers"].split(",")
            start = int(x["layers"][0])
            try:
                end = int(x["layers"][1])
            except:
                end = 0

            if start > 0:
                start -= index
            if end > 0:
                end -= index
            if end < 0:
                filters = output_filter[index + start] + output_filter[index + end]
            else:
                filters = output_filter[index + start]

            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)


        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{0}".format(index), shortcut)

        elif x["type"] == "yolo":
            mask = [int(x) for x in x["mask"].split(",")]
            anchors = [int(x) for x in x["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]           # Only use those anchors that define in mask

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{0}".format(index), detection)

        module_list.append(module)
        inp_filter = filters
        output_filter.append(filters)

    return network_info, module_list


class Darknet(nn.Module):
    """Class to define our Network"""

    def __init__(self, cfg_file):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfg_file)
        self.network_info, self.module_list = create_modules(self.blocks)

    def forward(self, x):
        modules = self.blocks[1:]
        outputs = {}

        flag = 0
        for index, module in enumerate(modules):
            module_type = module["type"]

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[index](x)

            elif module_type == "route":
                layers = [int(i) for i in module["layers"]]

                if (layers[0]) > 0:
                    layers[0] -= index

                if len(layers) == 1:
                    x = outputs[index + layers[0]]

                else:
                    if (layers[1]) > 0:
                        layers[1] -= index

                    layer1 = outputs[index + layers[0]]
                    layer2 = outputs[index + layers[1]]
                    x = torch.cat((layer1, layer2), 1)

            elif module_type == "shortcut":
                prev = int(module["from"])
                x = outputs[index - 1] + outputs[index + prev]

            elif module_type == "yolo":
                anchors = self.module_list[index][0].anchors
                input_dim = int(self.network_info["height"])
                classes = int(module["classes"])

                x = transform_tensor(x.data, input_dim, anchors, classes)

                if not flag:
                    detections = x
                    flag = 1

                else:
                    detections = torch.cat((detections, x), 1)

            outputs[index] = x

        return detections


    def load_weights(self, weightfile):
        #Open the weights file
        fp = open(weightfile, "rb")

        # First 5 values is header
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype = np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if batch_normalize:
                    bn = model[1]

                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    #Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    #Number of biases
                    num_biases = conv.bias.numel()

                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
