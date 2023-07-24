import torch
import torch.optim as optim
import torch
import torch.optim as optim
import torch.nn as nn
import clip
import numpy as np

from configs import parse_option
from pae import Ortho
from distributions import Distribution

# The arguments
args = parse_option()

# CLIP model
class CustomCLIP(nn.Module):
    def __init__(self):
        super(CustomCLIP, self).__init__()
        self.clip_model, self.preprocess = clip.load("RN50", device=args.device)
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self._ortho = Ortho()
        self.text_distribution = Distribution()

        for param in self.parameters():
            param.requires_grad = True

    def forward(self, images):
        torch.autograd.set_detect_anomaly(True)
        # The two prompts
        text1 = "Good photo."
        text2 = "Bad photo."
        text3 = "Beautiful photo."
        text4 = "Ugly photo."
        text5 = "Bright photo."
        text6 = "Dim photo."
        text7 = "poor"
        text8 = "fair"
        text9 = "perfect"
        text10 = "horrible"

        # Encode images
        image_embed = self.clip_model.encode_image(images.to(args.device))

        # Encode text
        text1_token = clip.tokenize(text1).to(args.device)
        text2_token = clip.tokenize(text2).to(args.device)
        text3_token = clip.tokenize(text3).to(args.device)
        text4_token = clip.tokenize(text4).to(args.device)
        text5_token = clip.tokenize(text5).to(args.device)
        text6_token = clip.tokenize(text6).to(args.device)
        text7_token = clip.tokenize(text7).to(args.device)
        text8_token = clip.tokenize(text8).to(args.device)
        text9_token = clip.tokenize(text9).to(args.device)
        text10_token = clip.tokenize(text10).to(args.device)
        text1_embed = self.clip_model.encode_text(text1_token)
        text2_embed = self.clip_model.encode_text(text2_token)
        text3_embed = self.clip_model.encode_text(text3_token)
        text4_embed = self.clip_model.encode_text(text4_token)
        text5_embed = self.clip_model.encode_text(text5_token)
        text6_embed = self.clip_model.encode_text(text6_token)
        text7_embed = self.clip_model.encode_text(text7_token)
        text8_embed = self.clip_model.encode_text(text8_token)
        text9_embed = self.clip_model.encode_text(text9_token)
        text10_embed = self.clip_model.encode_text(text10_token)

        ortho_vect = self._ortho.getOrtho(torch.stack([text1_embed[0], text2_embed[0], text3_embed[0], text4_embed[0], text5_embed[0], text6_embed[0], text7_embed[0], text8_embed[0], text9_embed[0], text10_embed[0]]))
        image_embed_projection = torch.matmul(image_embed, ortho_vect.T)   # dimension: batch_size by vector space size (number of text prompts)
        image_embed_projection = image_embed_projection.unsqueeze(1)  # Make it broadcastable by adding a new dimension (batch_size x 1 x 10)
        text_embed_projection_good = torch.matmul(torch.cat((text1_embed,text3_embed,text5_embed,text7_embed,text9_embed)), ortho_vect.T) # projected good text embeddings
        text_embed_projection_bad = torch.matmul(torch.cat((text2_embed,text4_embed,text6_embed,text8_embed,text10_embed)), ortho_vect.T) # projected bad text embeddings

        tg = self.text_distribution.generateSamples(good=True) # random samples of the good subspace
        tb = self.text_distribution.generateSamples(good=False) # random samples of the bad subspace

        # Calculate cosine similarities
        s1 = self.cosine_similarity(image_embed_projection, tg)
        s2 = self.cosine_similarity(image_embed_projection, tb)

        s1_scaled = torch.tensor(args.tau, device=args.device) * s1
        s2_scaled = torch.tensor(args.tau, device=args.device) * s2
   
        # Apply softmax function
        softmax_sim = nn.Softmax(dim=1)(torch.stack([s1_scaled, s2_scaled], dim=1))
        mos = (softmax_sim[:, 0] * 100).to(torch.float32)
        mos = torch.mean(mos, dim=1)
        
        return mos, image_embed_projection, tg, tb

# CLIP IQA model
class CLIP_IQA(nn.Module):
    def __init__(self):
        super(CLIP_IQA, self).__init__()
        self.clip_model, self.preprocess = clip.load("RN50", device=args.device)
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, images):
        # The two prompts
        text1 = "Good photo."
        text2 = "Bad photo."

        batch_size = images.size(0)

        # Encode images
        image_embed = self.clip_model.encode_image(images.to(args.device))

        # Encode text
        text1_token = clip.tokenize(text1).to(args.device)
        text2_token = clip.tokenize(text2).to(args.device)
        text1_embed = self.clip_model.encode_text(text1_token)
        text2_embed = self.clip_model.encode_text(text2_token)

        # Calculate cosine similarities
        s1 = self.cosine_similarity(image_embed, text1_embed)
        s2 = self.cosine_similarity(image_embed, text2_embed)

        s1_scaled = torch.tensor(args.tau, device=args.device) * s1
        s2_scaled = torch.tensor(args.tau, device=args.device) * s2
    
        # Apply softmax function
        softmax_sim = nn.Softmax(dim=1)(torch.stack([s1_scaled, s2_scaled], dim=1))

        return softmax_sim[0][0].tolist()
