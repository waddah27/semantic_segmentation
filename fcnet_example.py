import torch
import segmentation_models_pytorch as smp
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
model3 = smp.Unet(
    encoder_name="resnet101",        # choose encoder, e.g. mobilenet_v2
    encoder_weights="imagenet",     # use `imagenet` pretreined weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
)
print(model3)
# model.load_state_dict(torch.load("weights/model_fold_1.pth"))
model.eval()
model2.eval()
model3.eval()

# Step 2: Initialize the inference transforms
preprocess = transform #weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(to_pil_image(img)).unsqueeze(0)

# Step 4: Use the model and visualize the prediction
prediction = model3(batch)#["out"]
normalized_masks = prediction.softmax(dim=1)
class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
# mask = normalized_masks[0, 1]

to_pil_image(normalized_masks.squeeze(0).squeeze(0)).show()