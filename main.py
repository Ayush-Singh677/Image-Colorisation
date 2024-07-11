import torch
import torch.nn as nn
import numpy as np
import cv2
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d(2),
        )

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 2, kernel_size=3, padding=1),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Tanh(),
        )

    def forward(self, x):
        ds = self.downsample(x)
        output = self.upsample(ds)
        return output

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = image_lab[:, :, 0]
    l_channel_normalized = l_channel / 255.0
    return l_channel_normalized, image_lab

def scale_ab_channels(ab_channels):
    return (ab_channels + 1) * 128  # Correct scaling to [0, 255] range

def combine_channels(l_channel, ab_channels):
    l_channel = np.expand_dims(l_channel, axis=-1) * 255.0
    lab_image = np.concatenate((l_channel, ab_channels), axis=-1)
    return lab_image

def visualize(image_path, model):
    l_channel, image_lab = preprocess_image(image_path)
    l_channel_input = np.expand_dims(l_channel, axis=0)
    l_channel_input = np.expand_dims(l_channel_input, axis=1)
    l_channel_tensor = torch.from_numpy(l_channel_input).float().to(device)

    with torch.no_grad():
        predicted_ab_channels = model(l_channel_tensor).squeeze(0).cpu().numpy()

    predicted_ab_channels = predicted_ab_channels.transpose(1, 2, 0)
    scaled_ab_channels = scale_ab_channels(predicted_ab_channels)
    predicted_image_lab = combine_channels(l_channel, scaled_ab_channels)
    predicted_image_rgb = cv2.cvtColor(predicted_image_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    return predicted_image_rgb

def load_model(model_path):
    model = Unet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def main(input_path, output_path, model_path):
    image = cv2.imread(input_path)
    height, width, channels = image.shape
    model = load_model(model_path)
    colorized_image = visualize(input_path, model)
    colorized_image = cv2.resize(colorized_image, (width,height), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(output_path, cv2.cvtColor(colorized_image, cv2.COLOR_RGB2BGR))
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Automatic Image Colorization")
    parser.add_argument('--input', type=str, required=True, help="Path to the input grayscale image")
    parser.add_argument('--output', type=str, required=True, help="Path to save the colorized image")
    parser.add_argument('--model', type=str, default='model.pth', help="Path to the model parameters")
    args = parser.parse_args()

    main(args.input, args.output, args.model)