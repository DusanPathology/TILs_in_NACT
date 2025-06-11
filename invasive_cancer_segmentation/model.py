import segmentation_models_pytorch as smp

NUM_CLASSES = 5
DEVICE = "cuda" #Change as necessary

model1 = smp.MAnet(
    encoder_name="efficientnet-b0",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES
)

checkpoint = torch.load("path/to/model/weights", map_location=torch.device(DEVICE))
model.load_state_dict(checkpoint['model_state_dict'])
