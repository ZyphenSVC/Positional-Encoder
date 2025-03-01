import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image # Used to load and manipulate images

def positional_encoding(x, L=10):
	#frequencies: Creates a tensor of increasing frequency values (powers of 2 times π).
	# ex. l=4 -- [ 1.0 * π, 2.0 * π, 4.0 * π, 8.0 * π ]
    frequencies = 2.**torch.linspace(0, L -1, steps=L, dtype=torch.float32)
    #x_expanded: Expands x by multiplying each input coordinate by all frequency values.
    x_expanded = x[..., None] * frequencies[None, :]
    #torch.cat([...], dim=-1): Concatenates sine and cosine transformations along the last dimension.
    encoding = torch.cat([np.sin(x_expanded), np.cos(x_expanded)], dim=-1)
    #view(x.shape[0], -1): Reshapes the output to maintain batch size while flattening the feature dimension.
    return encoding.view(x.shape[0], -1)

class NeRFModel(nn.Module):
    def __init__(self, input_dim=3, L=10, hidden_dim=256):
        super(NeRFModel, self).__init__()
        self.L = L
        self.input_dim = input_dim * (2 * L)

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 4)

        self.softplus = nn.Softplus()
        self.relu = nn.ReLU6()
        self.relu = nn.Softsign()


    def forward(self, x):
        x = positional_encoding(x, self.L)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softplus(self.fc2(x))
        x = self.softplus(self.fc3(x))
        return x

def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    return transform(img).permute(1, 2, 0)  # Convert to HWC format

def generate_rays(image):
    H, W, _ = image.shape
    i, j = torch.tensor(np.meshgrid(torch.tensor(np.linspace(-1, 1, W,dtype=np.float32),
                                         dtype=torch.float32),
                          torch.tensor(np.linspace(-1, 1, H,dtype=np.float32), dtype=torch.float32),
                          indexing='ij'))

    rays = torch.stack([i, j, torch.tensor(np.ones_like(i))], dim=-1).view(-1, 3)
    return rays

def train_nerf(model, image, epochs=1000, lr=1e-3):
    rays = generate_rays(image)
    target_colors = image.view(-1, 3)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(rays)
        predicted_colors = output[:, 1:]
        loss = loss_fn(predicted_colors, target_colors)
        loss.backward()
        optimizer.step()


        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

            rendered_image = render_nerf(model, image)

            plt.figure(figsize=(5, 4))

            # Display Rendered Image
            plt.imshow(rendered_image)
            plt.title(f"Epoch {epoch}, Loss: {loss.item()}")
            # plt.axis('off')
            plt.show()

def render_nerf(model, image):
    rays = generate_rays(image)
    output = model(rays)
    colors = output[:, 1:].view(image.shape)
    return colors.detach().numpy()

#temp data for now
image = load_image("images.jpeg")
model = NeRFModel(L=128)
train_nerf(model, image)
