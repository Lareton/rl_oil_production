
import torch.nn as nn
import torch
from torch import optim


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size=3, stride=1, padding=1):
        super(conv2DBatchNormRelu, self).__init__()

        self.unit = nn.Sequential(
            nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, padding=padding, stride=stride),
            nn.BatchNorm2d(int(n_filters)),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        return self.unit(inputs)


class UNet2(nn.Module):
    def __init__(self, conv_layers_chanels=[8, 64, 128, 256, 512],
                 deconv_layers_channels=[512, 256, 128, 64, 1],
                 output_channel=1):
        super().__init__()

        self.conv_down_layers = []
        self.dec_conv_layers = []
        self.pool_layers = []
        self.upsample_layers = []

        for i in range(len(conv_layers_chanels) - 1):
            inp, out = conv_layers_chanels[i], conv_layers_chanels[i + 1]
            conv_layer = nn.Sequential(
                conv2DBatchNormRelu(inp, out),
                conv2DBatchNormRelu(out, out)
            )
            pool_layer = nn.Conv2d(out, out, kernel_size=3, stride=2, padding=1)

            self.conv_down_layers.append(conv_layer)
            self.pool_layers.append(pool_layer)


        # bottleneck
        self.bottle_neck = nn.Sequential(
            conv2DBatchNormRelu(512, 512, 1, 1, 0),
        )

        for i in range(len(deconv_layers_channels) - 1):
            inp, out = deconv_layers_channels[i], deconv_layers_channels[i + 1]

            self.upsample_layers.append(nn.ConvTranspose2d(inp, inp, kernel_size=3, stride=2, padding=1))

            self.dec_conv_layers.append(nn.Sequential(
                conv2DBatchNormRelu(inp*2, out),
                conv2DBatchNormRelu(out, out),
            ))

        out = deconv_layers_channels[-1]
        self.last_block = nn.Sequential(
            nn.Conv2d(out, out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out),
        )


    def forward(self, x):
        results_conv = []
        results_pool = []

        for i in range(len(self.conv_down_layers)):
            input_conv = x
            if i > 0:
                input_conv = results_pool[-1]

            results_conv.append(self.conv_down_layers[i](input_conv))
            results_pool.append(self.pool_layers[i](results_conv[-1]))

        # bottleneck
        b = self.bottle_neck(results_pool[-1])
        results_deconv = []

        for i in range(len(self.dec_conv_layers)):
            inp_deconv = b
            if i > 0:
                inp_deconv = results_deconv[-1]

            skip_data = results_conv[len(results_conv) - i - 1]
            output_size = skip_data.size()

            t = self.dec_conv_layers[i](torch.cat([
                self.upsample_layers[i](inp_deconv, output_size=output_size), skip_data], 1))
            results_deconv.append(t)

        res = self.last_block(results_deconv[-1])
        return res

    def get_all_weights(self):
        return [*self.conv_down_layers, *self.dec_conv_layers,
                       *self.pool_layers, *self.upsample_layers]

    def to_cuda(self):
        self.cuda()

        for tensor in self.get_all_weights():
            tensor = tensor.to("cuda")

    def to_cpu(self):
        self.cpu()

        for tensor in self.get_all_weights():
            tensor = tensor.to("cpu")
