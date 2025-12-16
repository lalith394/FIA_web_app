
import torch
import torch.nn as nn
from utils import model_summary

class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv_block(x)
        #print("Conv block: \t\t", x.shape)
        return x

class Encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convolution = Conv_block(in_channels, out_channels)
        self.max_pooling = nn.MaxPool2d(kernel_size=2)
    
    def forward(self, x):
        x = self.convolution(x)
        p = self.max_pooling(x)
        #print("Encoder block x: \t", x.shape, ", p: ", p.shape)
        return x, p

class Encoder_block_Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convolution = Conv_block(in_channels, out_channels)
        self.max_pooling = nn.MaxPool2d(kernel_size=1)
    
    def forward(self, x):
        x = self.convolution(x)
        p = self.max_pooling(x)
        #print("Encoder block x: \t", x.shape, ", p: ", p.shape)
        return x, p

class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.dim)

class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        self.decoder = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(2,2), stride=2)
        self.relu = nn.ReLU(inplace=True)
        if skip_channels is not None:
            self.conv_block = Conv_block(in_channels=out_channels + skip_channels, out_channels=out_channels)
            self.concat = Concat(dim=1)
    
    def forward(self, x, sk):
        if sk is None:
            x = self.decoder(x)
            x = self.relu(x)
            #print("Decoder block x: \t", x.shape)
            return x
        
        x = self.decoder(x)
        #print("ConvT x: \t\t", x.shape)
        x = self.concat(x, sk)
        #print("Concat x: \t\t", x.shape)
        x = self.conv_block(x)
        #print("Decoder block x: \t", x.shape)
        return x

class decoder_block_Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        self.decoder = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1), stride=1)
        self.relu = nn.ReLU(inplace=True)
        if skip_channels is not None:
            self.conv_block = Conv_block(in_channels=out_channels + skip_channels, out_channels=out_channels)
            self.concat = Concat(dim=1)
    
    def forward(self, x, sk):
        if sk is None:
            x = self.decoder(x)
            x = self.relu(x)
            #print("Decoder block x: \t", x.shape)
            return x
        
        x = self.decoder(x)
        #print("ConvT x: \t\t", x.shape)
        x = self.concat(x, sk)
        #print("Concat x: \t\t", x.shape)
        x = self.conv_block(x)
        #print("Decoder block x: \t", x.shape)
        return x

class Output(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0),
            nn.Sigmoid())
    
    def forward(self, x):
        x = self.output(x)
        #print("Output: \t\t", x.shape)
        return x	

class Output_MultiClass(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0))
    
    def forward(self, x):
        x = self.output(x)
        #print("Output: \t\t", x.shape)
        return x	   
class UNet(nn.Module):
    def __init__(self, num_outputs = 1, num_channels = [64, 128, 256, 512, 1024, 512, 256, 128, 64]):
        super().__init__()
        self.e1 = Encoder_block(in_channels=3, out_channels=num_channels[0])
        self.e2 = Encoder_block(in_channels=num_channels[0], out_channels=num_channels[1])
        self.e3 = Encoder_block(in_channels=num_channels[1], out_channels=num_channels[2])
        self.e4 = Encoder_block(in_channels=num_channels[2], out_channels=num_channels[3])
        self.bottle_neck = Conv_block(in_channels=num_channels[3], out_channels=num_channels[4])
        self.decoder_1 = decoder_block(in_channels=num_channels[4], out_channels=num_channels[5], skip_channels=num_channels[3])
        self.decoder_2 = decoder_block(in_channels=num_channels[5], out_channels=num_channels[6], skip_channels=num_channels[2])
        self.decoder_3 = decoder_block(in_channels=num_channels[6], out_channels=num_channels[7], skip_channels=num_channels[1])
        self.decoder_4 = decoder_block(in_channels=num_channels[7], out_channels=num_channels[8], skip_channels=num_channels[0])
        self.output = Output(in_channels=num_channels[8], out_channels=num_outputs)
    
    def forward(self, x, return_features = False):
        s1, p1 = self.e1(x) #[2, 64, 384, 576], [2, 64, 192, 288]
        s2, p2 = self.e2(p1) #[2, 128, 192, 288], [2, 128, 96, 144]
        s3, p3 = self.e3(p2) #[2, 256, 96, 144], [2, 256, 48, 72]
        s4, p4 = self.e4(p3) #[2, 512, 48, 72], [2, 512, 24, 36]
        b1 = self.bottle_neck(p4) #[2, 1024, 24, 36]
        #print("bottle_neck: \t\t", b1.shape)
        d1 = self.decoder_1(b1, s4)
        d2 = self.decoder_2(d1, s3)
        d3 = self.decoder_3(d2, s2)
        d4 = self.decoder_4(d3, s1)
        out = self.output(d4)
        if return_features is True:
            return out, d4
        else:
            return out

class AutoEncoder_RFMiD(nn.Module):
    def __init__(self, num_outputs = 3, num_channels = [64, 128, 256, 512, 1024, 512, 256, 128, 64]):
        super().__init__()
        self.e1 = Encoder_block(in_channels=3, out_channels=num_channels[0])
        self.e2 = Encoder_block(in_channels=num_channels[0], out_channels=num_channels[1])
        self.e3 = Encoder_block(in_channels=num_channels[1], out_channels=num_channels[2])
        self.e4 = Encoder_block(in_channels=num_channels[2], out_channels=num_channels[3])
        self.e5 = Encoder_block_Intermediate(in_channels=num_channels[3], out_channels=num_channels[3])
        self.e6 = Encoder_block_Intermediate(in_channels=num_channels[3], out_channels=num_channels[3])
        
        
        self.bottle_neck = Conv_block(in_channels=num_channels[3], out_channels=num_channels[4])

        self.d1 = decoder_block_Intermediate(in_channels=num_channels[4], out_channels=num_channels[5], skip_channels=num_channels[3])
        self.d2 = decoder_block_Intermediate(in_channels=num_channels[5], out_channels=num_channels[5], skip_channels=num_channels[3])
        self.d3 = decoder_block(in_channels=num_channels[5], out_channels=num_channels[5], skip_channels=num_channels[3])
        self.d4 = decoder_block(in_channels=num_channels[5], out_channels=num_channels[6], skip_channels=num_channels[2])
        self.d5 = decoder_block(in_channels=num_channels[6], out_channels=num_channels[7], skip_channels=num_channels[1])
        self.d6 = decoder_block(in_channels=num_channels[7], out_channels=num_channels[8], skip_channels=num_channels[0])
        self.out = Output(in_channels=num_channels[8], out_channels=num_outputs)
    
    def forward(self, x):
        #                                [1, 64, 128, 288]
        s1, p1 = self.e1(x)             #[2, 64, 192, 288]
        _, p2 = self.e2(p1)             #[2, 128, 96, 144]
        _, p3 = self.e3(p2)             #[2, 256, 48, 72]
        _, p4 = self.e4(p3)             #[2, 512, 24, 36] 
        _, p5 = self.e5(p4)           #[2, 512, 12, 18] 
        _, p6 = self.e6(p5)           #[2, 512, 6, 9]
        b1 = self.bottle_neck(p6)       #[2, 1024, 6, 9]
        d1 = self.d1(b1, None)   #[2, 512, 12, 18]
        d2 = self.d2(d1, None)  #[2, 512, 24, 36]
        d3 = self.d3(d2, None)  #[2, 512, 48, 72]
        d4 = self.d4(d3, None)   #[2, 256, 96, 144]
        d5 = self.d5(d4, None)   #[2, 128, 192, 288]
        d6 = self.d6(d5, s1)   #[2, 64, 384, 576]
        out = self.out(d6)           #[2, 1, 384, 576]
        return out
  
