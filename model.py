import torch
import torch.optim as optim
import torch.nn as nn
import clip
import numpy as np

from configs import parse_option

# The arguments
args = parse_option()

# CLIP model
class CustomCLIP(nn.Module):
    def __init__(self):
        super(CustomCLIP, self).__init__()
        self.clip_model, self.preprocess = clip.load("RN50", device=args.device)
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

        for param in self.parameters():
            param.requires_grad = True

    def forward(self, images):
        # The two prompts
        text1 = "Good photo."
        text2 = "Bad photo."

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
        mos = (softmax_sim[:, 0] * 100).to(torch.float32)

        return mos, image_embed, text1_embed, text2_embed

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
