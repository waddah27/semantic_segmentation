import torch
from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights, deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from torchvision.transforms.functional import to_pil_image
from augmentation import transform
from unet_model import Unet

img = read_image("dataset/images/000000561795.jpg")

# Step 1: Initialize model with the best available weights
weights = DeepLabV3_ResNet101_Weights.DEFAULT #FCN_ResNet50_Weights.DEFAULT
model = deeplabv3_resnet101(pretrained=True)#(weights=weights) #fcn_resnet50(weights=weights)
model2 = Unet()
print(model)
# model.load_state_dict(torch.load("weights/model_fold_1.pth"))
model.eval()
model2.eval()

# Step 2: Initialize the inference transforms
preprocess = transform #weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(to_pil_image(img)).unsqueeze(0)

# Step 4: Use the model and visualize the prediction
prediction = model2(batch)#["out"]
normalized_masks = prediction.softmax(dim=1)
class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
mask = normalized_masks[0, 1]#class_to_idx["person"]]

to_pil_image(mask).show()