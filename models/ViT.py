import torch
import torch.nn as nn
from transformers import ViTForImageClassification


def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False

class ViTModel(nn.Module):
    def __init__(self, model_name, label_dict, freeze_backbone = True):
        super(ViTModel, self).__init__()
        self.model_name = model_name
        self.labels = label_dict
        self.num_classes = len(label_dict)
        self.model = ViTForImageClassification.from_pretrained(
            self.model_name,
            id2label={str(i): c for i, c in enumerate(self.labels)},
            label2id={c: str(i) for i, c in enumerate(self.labels)}
        )

        if freeze_backbone:
            set_parameter_requires_grad(self.model)
        
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier.in_features, 1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, self.num_classes)
        )


        params_to_update = []
        name_params_to_update = []

        for name,param in self.model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                name_params_to_update.append(name)

        self.params_to_update = params_to_update
        self.name_params_to_update = name_params_to_update
    
    def forward(self, x):
        x = self.model(x)
        return x
    

    def print_params(self):
        print("Parameters will update !")
        print(self.name_params_to_update)