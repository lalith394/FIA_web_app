
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

class AutoEncoder(nn.Module):
    def __init__(self, num_outputs = 3, num_channels = [64, 128, 256, 512, 1024, 512, 256, 128, 64]):
        super().__init__()
        self.e1 = Encoder_block(in_channels=3, out_channels=num_channels[0])
        self.e2 = Encoder_block(in_channels=num_channels[0], out_channels=num_channels[1])
        self.e3 = Encoder_block(in_channels=num_channels[1], out_channels=num_channels[2])
        self.e4 = Encoder_block(in_channels=num_channels[2], out_channels=num_channels[3])
        self.e512 = Encoder_block_Intermediate(in_channels=num_channels[3], out_channels=num_channels[3])

        self.bottle_neck = Conv_block(in_channels=num_channels[3], out_channels=num_channels[4])
        self.decoder_1 = decoder_block(in_channels=num_channels[4], out_channels=num_channels[5], skip_channels=None)
        self.decoder512 = decoder_block_Intermediate(in_channels=num_channels[5], out_channels=num_channels[5], skip_channels=None)
        self.decoder_2 = decoder_block(in_channels=num_channels[5], out_channels=num_channels[6], skip_channels=None)
        self.decoder_3 = decoder_block(in_channels=num_channels[6], out_channels=num_channels[7], skip_channels=None)
        self.decoder_4 = decoder_block(in_channels=num_channels[7], out_channels=num_channels[8], skip_channels=num_channels[0])
        self.output = Output_MultiClass(in_channels=num_channels[8], out_channels=num_outputs)
    
    def forward(self, x, return_features = False):
        #                                [1, 64, 192, 288]
        s1, p1 = self.e1(x)             #[2, 64, 192, 288]
        _, p2 = self.e2(p1)             #[2, 128, 96, 144]
        _, p3 = self.e3(p2)             #[2, 256, 48, 72]
        _, p4 = self.e4(p3)             #[2, 512, 24, 36] 
        _, p4 = self.e512(p4)           #[2, 512, 12, 18] 
        _, p4 = self.e512(p4)           #[2, 512, 6, 9]
        b1 = self.bottle_neck(p4)       #[2, 1024, 6, 9]
        d1 = self.decoder_1(b1, None)   #[2, 512, 12, 18]
        d1 = self.decoder512(d1, None)  #[2, 512, 24, 36]
        d1 = self.decoder512(d1, None)  #[2, 512, 48, 72]
        d2 = self.decoder_2(d1, None)   #[2, 256, 96, 144]
        d3 = self.decoder_3(d2, None)   #[2, 128, 192, 288]
        d4 = self.decoder_4(d3, s1)   #[2, 64, 384, 576]
        out = self.output(d4)           #[2, 1, 384, 576]
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
        _, p1 = self.e1(x)             #[2, 64, 192, 288]
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
        d6 = self.d6(d5, None)   #[2, 64, 384, 576]
        out = self.out(d6)           #[2, 1, 384, 576]
        return out

class VAE(AutoEncoder_RFMiD):
    def __init__(self, num_outputs=3, num_channels=[64, 128, 256, 512, 1024, 512, 256, 128, 64]):
        super().__init__(num_outputs, num_channels)
        self.LATENT_VAR1 = 20
        self.LATENT_VAR2 = 20

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.relu = nn.ReLU()
        """ z1 (skip connection at d5) """
        self.s2_dense = nn.Linear(in_features=128*64*96, out_features=self.LATENT_VAR1)
        self.mu1 = nn.Linear(in_features=self.LATENT_VAR1, out_features=self.LATENT_VAR1)
        self.logvar1 = nn.Linear(in_features=self.LATENT_VAR1, out_features=self.LATENT_VAR1)
        self.unpack_z1 = nn.Linear(in_features=self.LATENT_VAR1, out_features=128*64*96)
        self.unflatten_z1 = nn.Unflatten(dim=1, unflattened_size=(128, 64, 96))


        """ z2 (bottleneck) """
        self.bottle_neck_dense = nn.Linear(in_features=1024*8*12, out_features=self.LATENT_VAR2)
        self.mu2 = nn.Linear(in_features=self.LATENT_VAR2, out_features=self.LATENT_VAR2)
        self.logvar2 = nn.Linear(in_features=self.LATENT_VAR2, out_features=self.LATENT_VAR2)
        self.unpack = nn.Linear(in_features=self.LATENT_VAR2, out_features=1024*8*12)
        self.unflattenBottleNeck = nn.Unflatten(dim=1, unflattened_size=(1024, 8, 12))

    
    def reparameterize(self, mu, log_var):
        # Compute the standard deviation from the log variance
        std = torch.exp(0.5 * log_var)
        # Generate random noise using the same shape as std
        eps = torch.randn_like(std)
        # Return the reparameterized sample
        return mu + eps * std

    def sample(self, num_images):
        with torch.no_grad():
            z1 = torch.randn(num_images, self.LATENT_VAR1)
            z2 = torch.randn(num_images, self.LATENT_VAR2)
            z1 = self.unpack_z1(z1)
            z1 = self.relu(z1)
            z1 = self.unflatten_z1(z1)
            z2 = self.unpack(z2)
            z2 = self.relu(z2)
            z2 = self.unflattenBottleNeck(z2)

            d1 = self.d1(z2, None)   #[2, 512, 12, 18]
            d2 = self.d2(d1, None)  #[2, 512, 24, 36]
            d3 = self.d3(d2, None)  #[2, 512, 48, 72]
            d4 = self.d4(d3, None)   #[2, 256, 96, 144]
            d5 = self.d5(d4, z1)   #[2, 128, 192, 288]
            d6 = self.d6(d5, None)   #[2, 64, 384, 576]
            out = self.out(d6)           #[2, 1, 384, 576]
            return out


    def forward(self, x):
        #                                [1, 64, 128, 288]
        _, p1 = self.e1(x)             #[2, 64, 192, 288]
        s2, p2 = self.e2(p1)             #[2, 128, 96, 144]

        """ Latent variable 1 at s2 """
        fs2 = self.flatten(s2)
        fc_s2 = self.s2_dense(fs2)
        fc_s2 = self.relu(fc_s2)
        mu_1, logvar_1 = self.mu1(fc_s2), self.logvar1(fc_s2)
        z1 = self.reparameterize(mu_1, logvar_1)
        z1 = self.unpack_z1(z1)
        z1 = self.relu(z1)
        z1 = self.unflatten_z1(z1)

        _, p3 = self.e3(p2)             #[2, 256, 48, 72]
        _, p4 = self.e4(p3)             #[2, 512, 24, 36] 
        _, p5 = self.e5(p4)           #[2, 512, 12, 18] 
        _, p6 = self.e6(p5)           #[2, 512, 6, 9]
        b1 = self.bottle_neck(p6)       #[2, 1024, 6, 9]

        """ Latent variable 2 at bottle neck """
        fb = self.flatten(b1)
        bd = self.bottle_neck_dense(fb)
        bd = self.relu(bd)
        mu_2, logvar_2 = self.mu2(bd), self.logvar2(bd)
        z2 = self.reparameterize(mu_2, logvar_2)
        z2 = self.unpack(z2)
        z2 = self.relu(z2)
        z2 = self.unflattenBottleNeck(z2)

        d1 = self.d1(z2, None)   #[2, 512, 12, 18]
        d2 = self.d2(d1, None)  #[2, 512, 24, 36]
        d3 = self.d3(d2, None)  #[2, 512, 48, 72]
        d4 = self.d4(d3, None)   #[2, 256, 96, 144]
        d5 = self.d5(d4, z1)   #[2, 128, 192, 288]
        d6 = self.d6(d5, None)   #[2, 64, 384, 576]
        out = self.out(d6)           #[2, 1, 384, 576]
        return out, mu_1, logvar_1, mu_2, logvar_2


""" class LadderVAE1(AutoEncoder_RFMiD):
    def __init__(self):
        super().__init__()
        # Add mu and log_var layers for reparameterization
        self.mu1, self.log_var1 = nn.Linear(in_features=1024, out_features=1024), nn.Linear(in_features=1024, out_features=1024)
        self.fc_z1_to_spatial = nn.Linear(1024, 1024 * 8 * 12)
        self.mu2, self.log_var2 = nn.Linear(in_features=64, out_features=64), nn.Linear(in_features=64, out_features=64)
        self.fc_z2_to_spatial = nn.Linear(64, 64 * 128 * 192)

    def reparameterize(self, mu, log_var):
        # Compute the standard deviation from the log variance
        std = torch.exp(0.5 * log_var)
        # Generate random noise using the same shape as std
        eps = torch.randn_like(std)
        # Return the reparameterized sample
        return mu + eps * std

    def forward(self, x):
        #                                [1, 64, 192, 288]
        s1, p1 = self.e1(x)             #[2, 64, 192, 288]
        _, p2 = self.e2(p1)             #[2, 128, 96, 144]
        _, p3 = self.e3(p2)             #[2, 256, 48, 72]
        _, p4 = self.e4(p3)             #[2, 512, 24, 36] 
        _, p4 = self.e512(p4)           #[2, 512, 12, 18] 
        _, p4 = self.e512(p4)           #[2, 512, 6, 9]
        b1 = self.bottle_neck(p4)       #[2, 1024, 6, 9]

        mu1, logvar1 = self.mu1(b1.mean([2,3])), self.log_var1(b1.mean([2,3]))
        mu2, logvar2 = self.mu2(s1.mean([2,3])), self.log_var2(s1.mean([2,3]))

        z1 = self.reparameterize(mu1, logvar1)
        z1_projected = self.fc_z1_to_spatial(z1).view(-1, 1024, 8, 12)
        z2 = self.reparameterize(mu2, logvar2)
        z2_projected = self.fc_z2_to_spatial(z2).view(-1, 64, 128, 192)

        d1 = self.decoder_1(z1_projected, None)   #[2, 512, 12, 18]
        d1 = self.decoder512(d1, None)  #[2, 512, 24, 36]
        d1 = self.decoder512(d1, None)  #[2, 512, 48, 72]
        d2 = self.decoder_2(d1, None)   #[2, 256, 96, 144]
        d3 = self.decoder_3(d2, None)   #[2, 128, 192, 288]
        d4 = self.decoder_4(d3, z2_projected)   #[2, 64, 384, 576]
        out = self.output(d4)           #[2, 1, 384, 576]
        return out, mu1, logvar1, mu2, logvar2

    def sample(self, num_samples):
        with torch.no_grad():
            # Generate random noise
            z1 = torch.randn(num_samples, 1)
            z2 = torch.randn(num_samples, 1)
            z1_projected = self.fc_z1_to_spatial(z1).view(-1, 1024, 8, 12)
            z2_projected = self.fc_z2_to_spatial(z2).view(-1, 64, 128, 192)
            # Pass the noise through the decoder to generate samples
            d1 = self.decoder_1(z1_projected, None)   #[2, 512, 12, 18]
            d1 = self.decoder512(d1, None)  #[2, 512, 24, 36]
            d1 = self.decoder512(d1, None)  #[2, 512, 48, 72]
            d2 = self.decoder_2(d1, None)   #[2, 256, 96, 144]
            d3 = self.decoder_3(d2, None)   #[2, 128, 192, 288]
            d4 = self.decoder_4(d3, z2_projected)   #[2, 64, 384, 576]
            samples = self.output(d4)
        # Return the generated samples
        return samples


class LadderVAE(AutoEncoder_RFMiD):
    def __init__(self):
        super().__init__()
        B1_DIM = 1024
        Z2_CHANNELS = 1024

        S1_DIM = 64
        Z1_CHANNELS = 64

        self.mu1, self.log_var1 = nn.Conv2d(in_channels=S1_DIM, out_channels=Z1_CHANNELS, kernel_size=1), nn.Conv2d(in_channels=S1_DIM, out_channels=Z1_CHANNELS, kernel_size=1)
        self.decode_z1 = nn.Conv2d(in_channels=Z1_CHANNELS, out_channels=S1_DIM, kernel_size=1)


        #bottle_neck_feat = 1024*8*12
        # Add mu and log_var layers for reparameterization
        #self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        #self.unflatten_z1 = nn.Unflatten(dim=1, unflattened_size=(64, 128, 192))
        #self.unflatten_z2 = nn.Unflatten(dim=1, unflattened_size=(1024, 8, 12))
        #self.mu1, self.log_var1 = nn.Linear(in_features=64*128*192, out_features=64*128*192), nn.Linear(in_features=164*128*192, out_features=64*128*192)
        self.mu2, self.log_var2 = nn.Conv2d(in_channels=B1_DIM, out_channels=Z2_CHANNELS, kernel_size=1), nn.Conv2d(in_channels=B1_DIM, out_channels=Z2_CHANNELS, kernel_size=1)
        self.decode_z2 = nn.Conv2d(in_channels=Z2_CHANNELS, out_channels=B1_DIM, kernel_size=1)

    def reparameterize(self, mu, log_var):
        # Compute the standard deviation from the log variance
        std = torch.exp(0.5 * log_var)
        # Generate random noise using the same shape as std
        eps = torch.randn_like(std)
        # Return the reparameterized sample
        return mu + eps * std

    def forward(self, x):
        #                                [1, 64, 192, 288]
        s1, p1 = self.e1(x)             #[2, 64, 192, 288]    
        #print("E1:\t",p1.shape, s1.shape)
        mu_1, logvar1 = self.mu1(s1), self.log_var1(s1)
        #print(f"LS \t mu1: {mu_1.shape} log(var): {logvar1.shape}")
        z1 = self.reparameterize(mu_1, logvar1)
        #print("Z1:\t",z1.shape)
        z1 = self.decode_z1(z1)
        #print("Z1_D:\t",z1.shape)    


        _, p2 = self.e2(p1)             #[2, 128, 96, 144]
        #print("E2:\t",p2.shape)  
        _, p3 = self.e3(p2)             #[2, 256, 48, 72]
        #print("E3:\t",p3.shape)
        _, p4 = self.e4(p3)             #[2, 512, 24, 36] 
        #print("E4:\t",p4.shape)
        _, p4 = self.e512(p4)           #[2, 512, 12, 18] 
        #print("E5:\t",p4.shape)
        _, p4 = self.e512(p4)           #[2, 512, 6, 9]
        #print("E6:\t",p4.shape)
        b1 = self.bottle_neck(p4)       #[2, 1024, 6, 9]
        #print("B1:\t",b1.shape)

        mu_2, logvar2 = self.mu2(b1), self.log_var2(b1)
        #print(f"LS \t mu2: {mu_2.shape} log(var): {logvar2.shape}")
        z2 = self.reparameterize(mu_2, logvar2)
        #print("Z2:\t",z2.shape)
        z2 = self.decode_z2(z2)
        #print("Z2_D:\t",z2.shape)


        d1 = self.decoder_1(z2, None)   #[2, 512, 12, 18]
        #print("D1:\t",d1.shape)
        d1 = self.decoder512(d1, None)  #[2, 512, 24, 36]
        #print("D2:\t",d1.shape)
        d1 = self.decoder512(d1, None)  #[2, 512, 48, 72]
        #print("D3:\t",d1.shape)
        d2 = self.decoder_2(d1, None)   #[2, 256, 96, 144]
        #print("D2:\t",d2.shape)
        d3 = self.decoder_3(d2, None)   #[2, 128, 192, 288]
        #print("D3:\t",d3.shape,"Skip: \t", z1.shape)
        d4 = self.decoder_4(d3, z1)   #[2, 64, 384, 576]
        #print("D4:\t",d4.shape)
        out = self.output(d4)           #[2, 1, 384, 576]
        #print("Output:\t",out.shape)
        return out, mu_1, logvar1, mu_2, logvar2
    
    def sample(self, num_samples):
        with torch.no_grad():
            # Generate random noise
            z1 = torch.randn(num_samples, 64, 128, 192)
            z2 = torch.randn(num_samples, 1024, 8, 12)
            # Pass the noise through the decoder to generate samples
            d1 = self.decoder_1(z2, None)   #[2, 512, 12, 18]
            d1 = self.decoder512(d1, None)  #[2, 512, 24, 36]
            d1 = self.decoder512(d1, None)  #[2, 512, 48, 72]
            d2 = self.decoder_2(d1, None)   #[2, 256, 96, 144]
            d3 = self.decoder_3(d2, None)   #[2, 128, 192, 288]
            d4 = self.decoder_4(d3, z1)   #[2, 64, 384, 576]
            samples = self.output(d4)
        # Return the generated samples
        return samples
 """

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_model = VAE()
    samples = unet_model.sample(16).cpu()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flatten()):
        img = samples[i].permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    

    #model_path = 'models/ae_e6_d6_w128_h128_rfm_sk[d6_d5]'
    model_summary(unet_model, f'test_sm_2.txt')
    
