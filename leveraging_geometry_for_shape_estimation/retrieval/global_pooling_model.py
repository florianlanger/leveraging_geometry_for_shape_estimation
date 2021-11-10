import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from os.path import join, exists, isfile, realpath, dirname

class Global_Pooling_Model(nn.Module):
    def __init__(self,pretrained_model,config,exp_path):
        super(Global_Pooling_Model, self).__init__()
        self.embedding_dimensions = config["model"]["embedding_dim"]
        self.config = config
        # Encoder
        # drop max pool and Relu
        layers = list(pretrained_model.features.children())[:-2]
        # maybe not dont train
        # if using pretrained then only train conv5_1, conv5_2, and conv5_3
        if config["training"]["encoder_weights_fixed"] == "True":
            for l in layers[:-5]: 
                for p in l.parameters():
                    p.requires_grad = False

        if config["training"]["compute_clusters"] == "True":
            layers.append(L2Norm())

        encoder = nn.Sequential(*layers)
        model = nn.Module() 
        model.add_module('encoder', encoder)


        if config["model"]["pooling"] == 'max':
            global_pool = nn.AdaptiveMaxPool2d((1,1))
            model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
        
        elif config["model"]["pooling"] == 'avg':
            global_pool = nn.AdaptiveAvgPool2d((1,1))
            model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))


        self.model_real = model
        if self.config["model"]["number_encoders"] == 2:
            self.model_syn = copy.deepcopy(model)


    def forward(self,x,kind,feature_masks,visualise=False):
        if self.config["model"]["number_encoders"] == 2:
            if kind == 'real':
                image_encoding = self.model_real.encoder(x)
                if self.config["model"]["filter_features"] == "True":
                    # expect feature_masks shape (bs,16,16)
                    feature_masks = feature_masks.unsqueeze(1).repeat(1,512,1,1)
                    image_encoding = image_encoding * feature_masks
                pooled = self.model_real.pool(image_encoding)

            elif kind == 'syn':
                image_encoding = self.model_syn.encoder(x)
                pooled = self.model_syn.pool(image_encoding)
            
        elif self.config["model"]["number_encoders"] == 1:
            if kind == 'real':
                image_encoding = self.model_real.encoder(x)
                if self.config["model"]["filter_features"] == "True":
                    # expect feature_masks shape (bs,16,16)
                    feature_masks = feature_masks.unsqueeze(1).repeat(1,512,1,1)
                    image_encoding = image_encoding * feature_masks
                pooled = self.model_real.pool(image_encoding)

            elif kind == 'syn':
                # note here it is model real, still need to check though whether real or syn for filtering
                image_encoding = self.model_real.encoder(x)
                pooled = self.model_real.pool(image_encoding)

        return pooled,image_encoding


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)
    