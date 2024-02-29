import torch
import numpy as np
import torchvision.transforms as trans
from PIL import Image

#Defined a pretrained pytorch model. In this example Nvidia's
#End-to-End Deep Learning for Self-Driving Cars architecture is used.
#https://developer.nvidia.com/blog/deep-learning-self-driving-cars/

class ADModel(torch.nn.Module):
    def __init__(self):
        super(ADModel, self).__init__()

        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 24, 5, stride=2),
            torch.nn.ELU(),
            torch.nn.Conv2d(24, 36, 5, stride=2),
            torch.nn.ELU(),
            torch.nn.Conv2d(36, 48, 5, stride=2),
            torch.nn.ELU(),
            torch.nn.Conv2d(48, 64, 3),
            torch.nn.ELU(),
            torch.nn.Conv2d(64, 64, 3),
            torch.nn.Dropout(0.5)
        )
        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=31680, out_features=100),
            torch.nn.ELU(),
            torch.nn.Linear(in_features=100, out_features=50),
            torch.nn.ELU(),
            torch.nn.Linear(in_features=50, out_features=10),
            torch.nn.Linear(in_features=10, out_features=3)
        )

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        self.transforms_numpy = trans.Compose([
                    trans.ToTensor(),
                    trans.Resize(size=(320, 180))
                ])
        self.use_pil = False

    def forward(self, input):
        input = torch.reshape(input, (input.size(0), 3, 320, 180))
        output = self.conv_layers(input)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output

    def load(self):
        self.load_state_dict(torch.load("..\\ad_checkpoints\\AutonomousDriving\\enhanced-8.pth")) #specify the trained weights that will be loaded


    def test_single(self, sample):
        """
            Get the most recent filename in a directory matching a specified prefix and extension.

            Parameters
            ----------
            sample : np.array
                An unpreprocessed frame as a uint8 numpy array with the same dimension as the ego vehicle RGB camera.

            Returns
            -------
            dict
                A dictionary that contains steering,throttle and brake values of the ego vehicle controls.

            Description
            -----------
           This function takes a single input (an image from the ego vehicle camera) and predicts steering,throttle and brake of the vehicle.
           If only one or two of the values is predicted from the model then a constant float can be set for the others.If the model expects the
           frame on a specific values-scale/size then it should be further processed inside this function.

            Examples
            --------
            result = {"steering": result[0][0], "throttle": 1.0 , "brake": 0.0 }
            """

        if self.use_pil:
            im = Image.fromarray(sample)
            im = im.resize((320, 180))
            im = np.array(im)
            im = (im / 255.0).astype(np.float32)
            sample = torch.from_numpy(im).permute(0, 2, 1).to("cuda:0")
        else:
            sample = sample / 255.0
            sample = sample.astype(np.float32)
            sample = self.transforms_numpy(np.ascontiguousarray(sample)).permute(0, 2, 1).to("cuda:0")
            print(sample.shape)

        sample = sample[None, :, :, :]

        result = self(sample)
        result = result.detach().cpu().numpy()
        result = {"steering": result[0][0], "throttle": result[0][1], "brake": result[0][2]}

        return result