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

        # The text prompts
        self.text1 = "Good photo."
        self.text2 = "Bad photo."
        self.text3 = "Beautiful photo."
        self.text4 = "Ugly photo."
        self.text5 = "Bright photo."
        self.text6 = "Dim photo."
        self.text7 = "poor"
        self.text8 = "fair"
        self.text9 = "perfect"
        self.text10 = "horrible"

        # Encode text
        self.text1_token = clip.tokenize(self.text1).to(args.device)
        self.text2_token = clip.tokenize(self.text2).to(args.device)
        self.text3_token = clip.tokenize(self.text3).to(args.device)
        self.text4_token = clip.tokenize(self.text4).to(args.device)
        self.text5_token = clip.tokenize(self.text5).to(args.device)
        self.text6_token = clip.tokenize(self.text6).to(args.device)
        self.text7_token = clip.tokenize(self.text7).to(args.device)
        self.text8_token = clip.tokenize(self.text8).to(args.device)
        self.text9_token = clip.tokenize(self.text9).to(args.device)
        self.text10_token = clip.tokenize(self.text10).to(args.device)
        self.text1_embed = self.clip_model.encode_text(self.text1_token)
        self.text2_embed = self.clip_model.encode_text(self.text2_token)
        self.text3_embed = self.clip_model.encode_text(self.text3_token)
        self.text4_embed = self.clip_model.encode_text(self.text4_token)
        self.text5_embed = self.clip_model.encode_text(self.text5_token)
        self.text6_embed = self.clip_model.encode_text(self.text6_token)
        self.text7_embed = self.clip_model.encode_text(self.text7_token)
        self.text8_embed = self.clip_model.encode_text(self.text8_token)
        self.text9_embed = self.clip_model.encode_text(self.text9_token)
        self.text10_embed = self.clip_model.encode_text(self.text10_token)

        self.ortho_vect = self._ortho.getOrtho(torch.stack([self.text1_embed[0], self.text2_embed[0], self.text3_embed[0], self.text4_embed[0], self.text5_embed[0], self.text6_embed[0], self.text7_embed[0], self.text8_embed[0], self.text9_embed[0], self.text10_embed[0]]))
        
        self.text_embed_projection_good = torch.matmul(torch.cat((self.text1_embed,self.text3_embed,self.text5_embed,self.text7_embed,self.text9_embed)).to(torch.float32), self.ortho_vect.T) # projected good text embeddings
        self.text_embed_projection_bad = torch.matmul(torch.cat((self.text2_embed,self.text4_embed,self.text6_embed,self.text8_embed,self.text10_embed)).to(torch.float32), self.ortho_vect.T) # projected bad text embeddings

        self.text_distribution = Distribution(self.text_embed_projection_good, self.text_embed_projection_bad)

        for param in self.parameters():
            param.requires_grad = True

    def forward(self, images):

        # Encode images
        image_embed = self.clip_model.encode_image(images.to(args.device))

        image_embed_projection = torch.matmul(image_embed, self.ortho_vect.T)   # dimension: batch_size by vector space size (number of text prompts)
        image_embed_projection = image_embed_projection.unsqueeze(1)  # Make it broadcastable by adding a new dimension (batch_size x 1 x 10)

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
