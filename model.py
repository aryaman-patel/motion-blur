import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import os
import scipy.io as sio

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class DeBlurImages(nn.Module):
    """
    A Fully Convolution network
    """
    def __init__(self):
        super(DeBlurImages, self).__init__()

        self.D = 128  # Define D before using it

        # Encoding Layers.
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels= 96, kernel_size= 7)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels= 96, out_channels= 256, kernel_size= 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels= 256, out_channels= 512, kernel_size= 3)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode= False)
        self.conv4 = nn.Conv2d(in_channels= 512, out_channels= 512, kernel_size= 3)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(in_channels= 512, out_channels= 512, kernel_size= 3)

        # Decoding Layers.
        self.uconv1 = nn.ConvTranspose2d(in_channels= 512, out_channels= 256, kernel_size= 8, stride= 2, padding= 0)
        self.uconv2 = nn.ConvTranspose2d(in_channels= 256, out_channels= self.D, kernel_size= 5, stride= 2, padding= 0)
        self.uconv3 = nn.ConvTranspose2d(in_channels= self.D, out_channels= self.D, kernel_size= 20, stride= 4, padding= 0)

        # 1D convolutions
        self.conv6 = nn.Conv2d(in_channels= 512, out_channels= 256, kernel_size = 1)
        self.conv7 = nn.Conv2d(in_channels= 256, out_channels= self.D, kernel_size = 1)
                
    def forward(self, x):
        """
        Forward pass of the network
        """
        x = self.conv1(x)
        x = nn.ReLU()(x)   
        x = self.pool1(x)

        x = self.conv2(x)
        x = nn.ReLU()(x)   
        x = self.pool2(x)
        residual2 = self.conv7(x)

        x = self.conv3(x)
        x = nn.ReLU()(x)   
        x = self.pool3(x)
        residual1 = self.conv6(x)

        
        x = self.conv4(x)
        x = nn.ReLU()(x)   
        x = self.pool4(x)

        x = self.conv5(x)
        x = nn.ReLU()(x)   

        
        x = self.uconv1(x)
        x = nn.ReLU()(x)    
        x = self.uconv2(residual1 + x)
        x = nn.ReLU()(x)   
        x = self.uconv3(residual2 + x)
        return x
    
    def compute_loss_rmse(self, prediction, u_target, v_target):
        """
        Computes the RMSE loss function for the optical flow estimation
        Args:
            prediction: The predicted flow field of size BxDxHxW
            u_target: The target u component of the flow field of size BxHxW
            v_target: The target v component of the flow field of size BxHxW
        """

        # Slice the predictions into u and v components as half of 256
        u_prediction = prediction[:, 0:64, :, :]
        v_prediction = prediction[:, 64:128, :, :]

        # Compute the squared differences
        u_diff = (u_prediction - u_target)**2
        v_diff = (v_prediction - v_target)**2

        # Compute the mean of the squared differences
        u_mean_squared_error = torch.mean(u_diff)
        v_mean_squared_error = torch.mean(v_diff)

        # Compute the RMSE
        rmse_u = torch.sqrt(u_mean_squared_error)
        rmse_v = torch.sqrt(v_mean_squared_error)

        # Compute the total loss
        loss = rmse_u + rmse_v
        return loss

    def compute_loss(self, prediction, u_target, v_target):
        """
        Computes the loss function for the optical flow estimation using Softmax layers.
        Args:
            prediction: The predicted flow field of size BxDxHxW
            u_target: The target u component of the flow field of size BxHxW
            v_target: The target v component of the flow field of size BxHxW
        """

        # Slice the predictions into u and v components as half of 256
        u_prediction = prediction[:, 0:64, :, :]
        v_prediction = prediction[:, 64:128, :, :]

        # Compute the softmax of the u_prediction and v_prediction
        u_softmax = torch.nn.functional.softmax(u_prediction, dim=1)
        v_softmax = torch.nn.functional.softmax(v_prediction, dim=1)

        # Reshape the tensors to BxCx(H*W)
        B, C, H, W = u_softmax.shape
        u_prediction = u_prediction.view(B, C, H * W)
        v_prediction = v_prediction.view(B, C, H * W)
        u_target = u_target.view(B, 1, H * W)
        v_target = v_target.view(B, 1, H * W)

        # Compute the indicator function for the u and v components using a tolerance of 1e-50.
        #u_indicator = torch.isclose(u_prediction, u_target.float(), atol=1e-50)
        #v_indicator = torch.isclose(v_prediction, v_target.float(), atol=1e-50)
        
        u_indicator = torch.where(u_prediction.int() == u_target.int(), torch.tensor(1), torch.tensor(0))
        v_indicator = torch.where(v_prediction.int() == v_target.int(), torch.tensor(1), torch.tensor(0)) 

        # Reshape the tensors back to BxCxHxW
        u_indicator = u_indicator.view(B, C, H, W)
        v_indicator = v_indicator.view(B, C, H, W)

        # Compute the loss for the u and v components
        loss_u = - torch.sum(u_indicator * torch.log(u_softmax))
        loss_v = - torch.sum(v_indicator * torch.log(v_softmax))

        # Compute the total loss
        epsilon = 1e-5
        loss = loss_u + loss_v #+ epsilon
        return loss
 



class BlurDataLoader(Dataset):
    """
    Dataloader for loading the images and the u, v mats. 
    """

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        if not file_name.endswith("_blurryimg.png"):
            return self.__getitem__((idx + 1) % len(self.file_list))

        img_path = os.path.join(self.root_dir, file_name)
        image = Image.open(img_path)
        image = self.transform(image)
        image = image.float()

        mat_path = os.path.join(self.root_dir, file_name.split('blurryimg.png')[0] + 'mfmap.mat')
        mat_data = sio.loadmat(mat_path)
        mfmap = mat_data['mfmap']

        # Create zeros of size mfmap [:, :, 0] and mfmap [:, :, 1]
        u_values = np.zeros((1, 300, 460))
        v_values = np.zeros((1, 300, 460))
        
        # Copy the values from mfmap to u_values and v_values keeping in mind the shape of the arrays
        u_values[:, :min(mfmap.shape[0], 300), :min(mfmap.shape[1], 460)] = mfmap[:min(mfmap.shape[0], 300), :min(mfmap.shape[1], 460), 0]
        v_values[:, :min(mfmap.shape[0], 300), :min(mfmap.shape[1], 460)] = mfmap[:min(mfmap.shape[0], 300), :min(mfmap.shape[1], 460), 1]
        u_values = torch.from_numpy(u_values)
        v_values = torch.from_numpy(v_values)

        return image.to(device), u_values.to(device), v_values.to(device)

    # Function to perform inference on a single image
def inference(image_path):
    """
    Function to perform inference on a single image 
    Args:
        image_path: The path of the image
    """
    print("Performing evaluation on a single image ...")
    image = Image.open(image_path)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5])(image)
    image = image.float()
    image = image.unsqueeze(0)
    model = DeBlurImages()
    model.load_state_dict(torch.load('DeBlur/weights.pth'))
    with torch.no_grad():
        prediction = model(image)
        prediction = prediction.squeeze(0)
        prediction = prediction.cpu().detach().numpy()
        # Save the file as a mat file
        sio.savemat('DeBlur/result' + '_flowfield.mat', {'mfmap': prediction})
    return prediction

def main() :
    
    dataset = BlurDataLoader("DeBlur/dataset_old/data_syn/train")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for batch in dataloader:
        images, u_values, v_values = batch
        print("Batch size:", images.size(0))
        print("Image shape:", images.size())
        print("u values shape:", u_values.size())
        print("v values shape:", v_values.size())
        print("- - - - - - - - - - ")
        break

    model = DeBlurImages()
    learning_rate = 1e-6
    momentum = 0.9
    step_size = 30
    gamma = 0.1  # decay factor
    num_epochs = 200    
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    model.load_state_dict(torch.load('/home/patel.aryam/DeBlur/weights.pth'))

    for epoch in range(num_epochs) :
        for images , u_values, v_values in tqdm(dataloader) :
            prediction = model(images)
            loss = model.compute_loss_rmse(prediction, u_values, v_values)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()  # decay the learning rate
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]}")
        
    # Save the best weights of the model.
    torch.save(model.state_dict(), 'DeBlur/weights.pth')

    inference(image_path= "DeBlur/dataset_old/data_syn/train/8143_006_blurryimg.png")

    

if __name__ == "__main__":
    main()


