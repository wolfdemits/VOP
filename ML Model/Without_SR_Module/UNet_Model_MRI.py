#%% 

#####################
## UNet Model Code ##
#####################

## This script defines the blocks and classes used to build-up the CNN model

""" General Structure of Class Definition Blocks 

class NAME(nn.Module):

    def __init__(self, ...):    # Responsible for initialisation 
                                # Automatically called when an instance of the class is created
    
    def forward(self, ...):     # Defines the forward pass of the model (prediction step)
                                # It dictates how the input data is passed through the layers and
                                  and transformed to an output, i.e., describes the operations
    
        return
"""


# Import necessary packages

import torch # type: ignore
import torch.nn as nn # type: ignore


# Function to determine the total amount of parameters in the neural net

def count_model_parameters(model):

    total_params = 0
    
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params+=param

    return total_params


#### CONVOLUTIONAL BLOCKS ####################################################

class ConvBlock(nn.Module):
    
    """ Convolutional blocks can include: conv layers, dropout layers, batch
    normalization layers, and the activation function as last block """
    
    def __init__(self, dim, in_channel, out_channel, conv_kernel, activation):

        # dim = dimension --> '2d' or '3d'

        super().__init__()
        
        Conv = nn.Conv2d if dim == '2d' else nn.Conv3d
        Norm = nn.BatchNorm2d if dim == '2d' else nn.BatchNorm3d
        
        self.conv  = Conv(in_channel, out_channel, conv_kernel, stride=1, padding="same")  
        
        # Definition of Conv:
        #   in_channel  = number of input channels for the convolutional layer 
        #   out_channel = determines how many different feature maps the convolution will produce after applying the kernel 
        #   conv_kernel = size of the convolutional kernel (i.e., width and height of the filter used to convolve over the input)
        #   stride      = determines over how many pixels the kernel jumps across
        #                 e.g., stride = 1 --> Kernel moves one pixel at a time (!! This preserves spatial consistency)
        #                       stride = 2 --> Kernel jumps to the second next pixel 
        #                   Note 
        #   padding     = is used to ensure that the output feature map has the same spatial dimensions as the input, with stride =1  

        self.norm  = Norm(out_channel)
        self.activ = getattr(nn, activation)()  # Call activation function from nn module
        
    def forward(self, x):

        x = self.conv(x)
        x = self.norm(x)
        x = self.activ(x)
        
        return x


class DoubleConvBlock(nn.Module):
    
    """ The ConvBlock Class is applied twice """ 
    
    def __init__(self, dim, in_channel, mid_channel, out_channel, conv_kernel, activation):
        
        super().__init__()
        
        self.conv1 = ConvBlock(dim, in_channel, mid_channel, conv_kernel, activation)
        self.conv2 = ConvBlock(dim, mid_channel, out_channel, conv_kernel, activation)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    

#### SINGLE CONV BLOCKS ######################################################   
    
class Conv_1x1(nn.Module):

    """ Just one convolution is performed, but no batch
    normalisation nor activation function applied """
    
    def __init__(self, dim, in_channel, out_channel):
        
        super().__init__()
        Conv = nn.Conv2d if dim == '2d' else nn.Conv3d
        self.conv = Conv(in_channel, out_channel, kernel_size=1, stride=1, padding="same", dilation=1)
        
    def forward(self, x):
        x = self.conv(x)
        return x
    

#### ENCODER #################################################################

""" Role of the Encoder: 

The idea is to encode the input into a more compact, informative representation that contains the essential 
information needed for the task. The encoder processes the input data by applying a series of operations that 
aim to progressively extract more complex/abstract features from the data using downsampling and convolutional 
layers. 

Key Functions of the Encoder:

- Downsampling       : reducing the spatial dimensions of the input to help focus on the most important patterns and features
- Convolution blocks : using convolutional layers to extract key features from the input data.

"""

## Pooling

class DownBlock_Pool(nn.Module):
    """ Class to perform the encoding via pooling 
        -> reduce the spatial size without learning additional parameters (reduced memory usage) 
    """

    def __init__(self, dim, in_channel, out_channel, conv_kernel, pool_mode, activation):
        
        super().__init__()
        
        # Choose pooling method
        if pool_mode == 'maxpool':
            pool_operation = nn.MaxPool2d if dim == '2d' else nn.MaxPool3d
        
        elif pool_mode == 'meanpool':
            pool_operation = nn.AvgPool2d if dim == '2d' else nn.AvgPool3d
        
        # Use double convolutional block
        double_conv = DoubleConvBlock(dim, in_channel, out_channel, out_channel, conv_kernel, activation)

        # Define the "down" (= encoding) path in the neural net
        #   nn.Sequential --> Operational layers will be added in the order they are passed in the constructor
        self.Down = nn.Sequential(pool_operation(2), double_conv)   

    def forward(self, x):
        return self.Down(x)


## Strided Convolutions 

class DownBlock_ConvStride2(nn.Module):
    """ Class to perform the encoding via convolutions with a stride of 2. Strided convolutions also reduce the 
    spatial dimensions of the input by applying filters that "skip" over the input """

    def __init__(self, dim, in_channel, out_channel, conv_kernel, activation):
        
        super().__init__()

        Conv = nn.Conv2d if dim == '2d' else nn.Conv3d
        Norm = nn.BatchNorm2d if dim == '2d' else nn.BatchNorm3d
        
        self.Down = nn.Sequential(
                        Conv(in_channel, in_channel, kernel_size=3, stride=2),  # Conv with stride = 2 --> reduction of spatial dimension
                        Norm(in_channel), 
                        getattr(nn, activation)())
        
        self.DoubleConv = DoubleConvBlock(dim, in_channel, out_channel, out_channel, conv_kernel, activation)
       

    def forward(self, x):

        x = self.Down(x)
        x = self.DoubleConv(x)

        return x



#### HELPER FUNCTIONS FOR RESIZING & CONCATENATION

def Resize_Padding(dim, x_to_resize, x_reference, PadValue):
    
    diff_w = x_reference.shape[-1] - x_to_resize.shape[-1]
    diff_h = x_reference.shape[-2] - x_to_resize.shape[-2]
        
    if dim == '2d':            
        x_padded = torch.nn.functional.pad(x_to_resize, (0, diff_w, 0, diff_h), value=PadValue)
    
    elif dim == '3d':
        diff_z = x_reference.shape[-3] - x_to_resize.shape[-3]  
        x_padded = torch.nn.functional.pad(x_to_resize, (0, diff_w, 0, diff_h, 0, diff_z), value=PadValue)

    return x_padded


def Concat_SkipConnection(dim, x_out_skip, x_decode, PadValue):

    if (x_out_skip.shape[2:] != x_decode.shape[2:]):
        x_decode = Resize_Padding(dim, x_to_resize=x_decode, x_reference=x_out_skip, PadValue=PadValue) 

    x = torch.cat([x_decode, x_out_skip], dim=1)

    return x


#### DECODER #################################################################

""" 
The role of a decoder is to take a compressed representation of an image (produced by an encoder) 
and transform it back into a high-resolution image. The decoder includes upsampling layers (to 
increase image size) and convolutional layers (to refine details). 

Two cases:

1. Without Skip Connections
   - The decoder gradually upsamples (enlarges) the compressed data using upsampling techniques like interpolation or transposed convolutions.

2. With Skip Connections
   - The decoder receives extra information from the encoder, which helps restore important details lost during compression.

"""

class UpSample(nn.Module):
    
    def __init__(self, dim, num_main_channel, num_skip_channel, num_channel_out, conv_kernel, activation):
        
        """
        - num_main_channel = amount of input channels before the upsampling
        - num_skip_channel = amount of input channels before the skip path
        - num_channels_out = amount of channels after the double convolution
        """
        
        super().__init__()

        self.dim = dim

        mode = 'bilinear' if dim=='2d' else 'trilinear'

        self.Up = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)

        # Initialise the number of input channels before the double convolution 
        # (Note: no skip --> in_channels = num_main_channel)
        in_channel = num_main_channel + num_skip_channel 
        

        # Initialise amount of channels for the double convolution 
        
        if num_skip_channel != 0: # with skip connections
            self.skip_connection = True
            mid_channel = num_main_channel
            out_channel = num_channel_out
            
        elif num_skip_channel == 0: # no skip connections
            self.skip_connection = False
            mid_channel = num_channel_out
            out_channel = num_channel_out
        
        # Initialise convolutional path
        self.DoubleConv = DoubleConvBlock(dim, in_channel, mid_channel, out_channel, conv_kernel, activation)
    
    
    def forward(self, x, x_encode, PadValue):

        x = self.Up(x)

        if (self.skip_connection): # if skip connection, then concatenate
            x = Concat_SkipConnection(self.dim, x_out_skip=x_encode, x_decode=x, PadValue=PadValue)

        x = self.DoubleConv(x)
        
        return x


class UpConv(nn.Module):
    
    def __init__(self, dim, num_main_channel, num_skip_channel, num_channel_out, conv_kernel, activation):
    
        super().__init__()
 
        self.dim = dim
        ConvTranspose = nn.ConvTranspose2d if dim=='2d' else nn.ConvTranspose3d
        self.Up = ConvTranspose(num_main_channel, num_main_channel, kernel_size=2, stride=2, padding=0)
        
        # Initialise the number of input channels before the double convolution 
        # (Note: no skip --> in_channels = num_main_channel)
        in_channel = num_main_channel + num_skip_channel


        # Initialise amount of channels for the double convolution 
        
        if num_skip_channel != 0: # with skip connections
            self.skip_connection = True
            mid_channel = num_main_channel
            out_channel = num_channel_out
            
        elif num_skip_channel == 0: # no skip connections
            self.skip_connection = False
            mid_channel = num_channel_out
            out_channel = num_channel_out

        # Initialise convolutional path
        self.DoubleConv = DoubleConvBlock(dim, in_channel, mid_channel, out_channel, conv_kernel, activation)
 
    
    def forward(self, x, x_encode, PadValue):

        x = self.Up(x)

        if (self.skip_connection): # if skip connection, then concatenate
            x = Concat_SkipConnection(self.dim, x_out_skip=x_encode, x_decode=x, PadValue=PadValue)

        x = self.DoubleConv(x)
 
        return x


#### UNET #####################################################################

class UNet(nn.Module):

    def __init__(self, dim, num_in_channels, features_main, features_skip, conv_kernel_size, 
            down_mode, up_mode, activation, residual_connection, PadValue=-1):
        """
        Initialisation of the U-Net Class
        
        Inputs:
        - dim: dimension of the data ('2d' or '3d')
        - num_in_channels
        - features_main
        - features_skip
        - conv_kernel_size
        - down_mode
        - up_mode
        - activation
        - residual_connection
        - PadValue """

        super().__init__()
        
        self.dim = dim
        self.depth = len(features_skip)
        self.num_in_channels = num_in_channels
        self.residual_connection = residual_connection
        self.PadValue = PadValue
        
        
        # INCOME, FIRST LAYERS

        self.income = DoubleConvBlock( dim, num_in_channels, features_main[0], features_main[0], conv_kernel_size, activation)
        

        # DOWN PART

        if (down_mode == 'maxpool') or (down_mode == 'meanpool'):
            self.Downs = nn.ModuleList([
                                DownBlock_Pool(dim, 
                                        features_main[i],
                                        features_main[i+1],
                                        conv_kernel_size,
                                        down_mode,
                                        activation)
                                for i in range(self.depth)])

        elif (down_mode == 'convStrided'):
            self.Downs = nn.ModuleList([
                                DownBlock_ConvStride2(dim, 
                                        features_main[i],
                                        features_main[i+1],
                                        conv_kernel_size,
                                        activation)
                                for i in range(self.depth)])
        
        
        # UP PART

        if (up_mode == 'upsample'):
            self.Ups = nn.ModuleList([
                            UpSample(dim, 
                                     features_main[i+1],
                                     features_skip[i],
                                     features_main[i],
                                     conv_kernel_size,
                                     activation)
                            for i in reversed(range(self.depth))])
        
        elif (up_mode == 'upconv'):
            self.Ups = nn.ModuleList([
                            UpConv(dim, 
                                   features_main[i+1],
                                   features_skip[i],
                                   features_main[i],
                                   conv_kernel_size,
                                   activation)
                            for i in reversed(range(self.depth))])
        

        # OUT: Conv1x1 && Add ReLU (enforce non-negativity)
        self.out_Conv1x1 = Conv_1x1(dim, features_main[0], 1)
        self.out_ReLU = getattr(nn, 'ReLU')()
    
    
    def forward(self, x_input):
        
        x = self.income(x_input)
        
        save_skip = []

        for encode_block in self.Downs:
            save_skip.append(x)
            x = encode_block(x)
            
        for decode_block in self.Ups:
            x = decode_block(x, save_skip.pop(), self.PadValue)
            
        x = self.out_Conv1x1(x)
        
        x = Resize_Padding(self.dim, x_to_resize=x, x_reference=x_input, PadValue=self.PadValue)
    
        
        # Include Residual Connection: Network predicts y_diff = x_out - x_input
        
        if self.residual_connection:
            y_diff = x
            residualChannel = self.num_in_channels // 2   
            
            if (self.dim == '2d'):
                x_input_oneChannel = x_input[:, residualChannel:residualChannel+1, :, :]
            
            elif (self.dim == '3d'):
                x_input_oneChannel = x_input[:, residualChannel:residualChannel+1, :, :, :]
            
            x_out = y_diff + x_input_oneChannel

        else: 
            x_out = x
        
        x = self.out_ReLU(x)
        
        return x_out
    

#### TEST #####################################################################

if __name__ == '__main__':
    
    random2Dtensor = torch.rand(1, 3, 123, 76)

    model = UNet(
        dim = '2d', 
        num_in_channels = 3, 
        features_main = [64, 128, 256, 512, 1024], 
        features_skip = [64, 128, 256, 512], 
        conv_kernel_size = 3, 
        down_mode = 'maxpool',
        up_mode = 'upconv', 
        activation = 'PReLU', 
        residual_connection = True, 
        PadValue = 0)
    
    output = model(random2Dtensor)
    print(output.shape)


# %%
