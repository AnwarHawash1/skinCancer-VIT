from transformers import ViTForImageClassification, ViTImageProcessor
import torch.nn as nn

class CustomViTModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomViTModel, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")
        self.feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.vit.config.hidden_size, self.num_classes)

    def forward(self, inputs):
        outputs = self.vit(inputs)
        return outputs.logits
