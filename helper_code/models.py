from transformers import ViTModel, ViTImageProcessor
import torch.nn as nn
import torch


torch.manual_seed(1234)

#Class to make the encoder. It has the ViT architecture, just removes the classification head.
class ViTEmbeddingNet(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model
        
    def forward(self, pixel_values: torch.FloatTensor,labels: torch.LongTensor = None):
        outputs = self.vit(pixel_values)
        # Use [CLS] token (first token in the sequence) as embedding
        return outputs.last_hidden_state[:, 0]


#Classification head for model
class ClassificationHead(nn.Module):

    def __init__(self, input_dim = 768, num_classes = 4, hidden_dim=128):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.norm(x)
        return self.head(x)


#Puts encoder and classification head together
class FullModel(nn.Module):
    def __init__(self, encoder, classification_head):
        self.encoder = encoder
        self.head = classification_head


    def forward(self, pixel_values: torch.FloatTensor,labels: torch.LongTensor = None):
        embeddings = self.encoder(pixel_values, labels)
        return self.head(embeddings)


#Creates encoder
def create_encoder(model_name = "google/vit-base-patch16-224"):
    model_name = "google/vit-base-patch16-224"
    vit = ViTModel.from_pretrained(model_name, dtype=torch.float32)
    return ViTEmbeddingNet(vit)