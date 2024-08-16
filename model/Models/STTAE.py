import torch
import torch.nn as nn
import torchvision.models as models

from Models import VRAE


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class Encoder(nn.Module):
    def __init__(self, base_model='VGG19', frame_diff=None, args=None):
        super(Encoder, self).__init__()        
        
        self.cnn_encoder = models.vgg19(pretrained=True)
        self.cnn_encoder.classifier = nn.Sequential(*list(self.cnn_encoder.classifier.children())[:-1], nn.Linear(in_features=4096, out_features=args.dim, bias=True))
        
    def forward(self, x):
        # x: (N, T, C, H, W)
        feature_sequence = {}
        
        # encode each frame
        for i, clip in enumerate(x):
            feature = self.cnn_encoder(clip)
            feature_sequence[i] = feature.unsqueeze(0)

        batch_feature_sequence = torch.zeros_like(feature_sequence[i])

        for key, values in feature_sequence.items():
            batch_feature_sequence = torch.cat( (batch_feature_sequence, values), dim=0 )    

        return batch_feature_sequence[1:]
    
    
class VariationalRAE(nn.Module):
    def __init__(self, config, batch_size, device, args=None):
        super(VariationalRAE, self).__init__()
        
        self.seq_encoder = VRAE.Encoder(args.dim, config.hidden_size, config.hidden_layer_depth, config.latent_length, config.dropout)
        self.lmbd = VRAE.Lambda(config.hidden_size, config.latent_length)
        self.seq_decoder = VRAE.Decoder(config.sequence_length, batch_size, config.hidden_size, config.hidden_layer_depth, config.latent_length, args.dim, device)

    def forward(self, x):
        # x : (T, N, dim)
        cell_output = self.seq_encoder(x)
        latent = self.lmbd(cell_output)
        x_decoded = self.seq_decoder(latent)
        
        return x_decoded.permute(1, 0, 2), latent
    
    def kl_loss(self):
        """
        Compute the loss given output x decoded, input x and the specified loss function
        :param x_decoded: output of the decoder
        :param x: input to the encoder
        :param loss_fn: loss function specified
        :return: joint loss, reconstruction loss and kl-divergence loss
        """
        latent_mean, latent_logvar = self.lmbd.latent_mean, self.lmbd.latent_logvar

        kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())

        return kl_loss
    
    
# class STTAE(nn.Module):
#     def __init__(self, config, batch_size, base_model='VGG19', args=None):
#         super(STTAE, self).__init__()
        
#         self.encoder = Encoder(base_model=base_model, frame_diff=args.fd)
#         self.vrae = VariationalRAE(config, batch_size)
        
#     def forward(self, x):
#         # x: torch.tensor (N, C, T, H, W)  ->  (N, T, C, H, W)
#         x = x.permute(0,2,1,3,4)
        
#         batch_feature_sequence = self.encoder(x)  # batch_feature_sequence: (N, T, dim)        
#         x_decoded = self.vrae(batch_feature_sequence.permute(1, 0, 2))
        
#         return batch_feature_sequence, x_decoded  # batch_feature_sequence : x (original features)      
#                                                   # x_decoded : reconstructed features
    
#     def kl_loss(self):
#         """
#         Compute the loss given output x decoded, input x and the specified loss function
#         :param x_decoded: output of the decoder
#         :param x: input to the encoder
#         :param loss_fn: loss function specified
#         :return: joint loss, reconstruction loss and kl-divergence loss
#         """
#         latent_mean, latent_logvar = self.vrae.lmbd.latent_mean, self.vrae.lmbd.latent_logvar

#         kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())

#         return kl_loss
    