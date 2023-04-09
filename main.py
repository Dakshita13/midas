import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os
from pathlib import Path

from video_reader import read_mp4

BASE_DIR = os.getcwd()


model = torch.hub.load('intel-isl/MiDaS', 'MiDaS')
model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model.eval()

def predict_depth(img):
    # Load image
    # img = cv2.imread(image_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = Image.fromarray(img)

    # Apply transform
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img_tensor = transform(img)
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    # Move to device
    img_tensor = img_tensor.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # Predict depth
    with torch.no_grad():
        prediction = model(img_tensor)
        prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1),
                                                      size=(img.height, img.width),
                                                      mode="bicubic",
                                                      align_corners=False)
        prediction = prediction.squeeze().cpu().numpy()
    return prediction


def plot_depth(pil_image):
    # depth_map = predict_depth(r"C:\Users\user\Desktop\padhai\pytho\midas\Images\pothole_1.jpeg")
    depth_map = predict_depth(pil_image)
    # cv2.imshow("Depth Map", depth_map)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    import matplotlib.pyplot as plt
    import numpy as np
    # convert the depth map to a numpy array
    # depth_map_np = depth_map.squeeze().cpu().detach().numpy()
    depth_map_np = depth_map
    # normalize the depth map values to [0, 1] range
    depth_map_np = (depth_map_np - np.min(depth_map_np)) / (np.max(depth_map_np) - np.min(depth_map_np))

    # display the depth map
    plt.imshow(depth_map_np, cmap='jet')
    plt.show()

if __name__ == "__main__":
    video_directory = os.path.join(BASE_DIR, "Videos", "vid.mp4")
    image_list = read_mp4(video_directory)
    for image in image_list:
        plot_depth(image)
