import torch
import numpy as np
from torchvision import models
import os
from epe.autonomous_driving.rl_buffer import ReplayBuffer
import torchvision.transforms as transforms
import cv2

#A sample of a simple DQN reinforcement learning algorithm and how you can integrate such models

#Define a Pytorch model
class DQNCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(DQNCNN, self).__init__()

        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50 = torch.nn.Sequential(*(list(self.resnet50.children())[:-2]))
        for param in self.resnet50.parameters():
            param.requires_grad = True


        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=100352, out_features=512),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=512, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=num_classes),
            #torch.nn.Tanh()
        )

        #Define a pretrained Variational Autoencoder or any other feature extraction method if needed


    def forward(self, x):
        print(x.shape)
        x = self.resnet50(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class RLModel:
    def __init__(self,action_dim, learning_rate= 0.001, gamma=0.99, epsilon=0.9, epsilon_min=0.01, epsilon_decay=0.9995,batch_size=32,target_update=100,replay_buffer_size=10000):
        self.action_dim = action_dim
        self.policy_net = DQNCNN(action_dim).to("cuda:0")
        self.target_net = DQNCNN(action_dim).to("cuda:0")
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate, weight_decay=0.0001)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lossfn = torch.nn.MSELoss()
        self.target_update = target_update
        self.tick_counter = 0
        self.min_buffer_size = batch_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        self.transforms_numpy = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(400),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    #Get latent space from a DL autoencoder. The function recieves the frame as a numpy array
    #and returns the latent space also as a vector in numpy array.
    def encode(self,x):
        return x.numpy()

    #Add to replay buffer
    def add_observation(self,state, action, next_state, reward, done):
        self.replay_buffer.add(state, action, next_state, reward, done)

    #Get a uint8 numpy array frame and preprocess it based on how the model
    #or the feature extraction method expects the input (size,value-scaling etc). The returned array
    #should also be a numpy array.
    def preprocess_camera_frame(self,frame):
        #rgb_frame = cv2.resize(frame , (160, 80),interpolation=cv2.INTER_AREA)
        rgb_frame = self.transforms_numpy(np.array(frame)).permute(0, 2, 1).cpu().detach().numpy().astype(np.float32)
        #rgb_frame = self.encode(rgb_frame)
        return rgb_frame

    #Select action based on a given state. If random is true then exploration &
    #exploitation will be used.
    def select_action(self, state,random):
        self.policy_net.eval()
        if np.random.rand() < self.epsilon and random:
            return np.random.choice(self.action_dim)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to("cuda:0")
            q_values = self.policy_net(state_tensor)
            best_action, best_action_index = torch.max(q_values, 1)
            action = best_action_index.item()
            return action

    #save the models
    def save(self,model_name,episode_num):
        torch.save(self.policy_net.state_dict(), "../ad_checkpoints/policy-"+model_name+"-"+str(episode_num)+".pth")
        torch.save(self.target_net.state_dict(), "../ad_checkpoints/target-"+model_name+"-"+str(episode_num)+".pth")

    #set evaluation mode on both models
    def eval_mode(self):
        self.policy_net.eval()
        self.target_net.eval()

    #load models given a model name and its episode number. Both parameters can be set
    #from the carla_config.yaml file.
    def load(self,model_name,episode_num):
        self.policy_net.load_state_dict(torch.load("../ad_checkpoints/policy-"+model_name+"-"+str(episode_num)+".pth"))
        self.target_net.load_state_dict(torch.load("../ad_checkpoints/target-"+model_name+"-"+str(episode_num)+".pth"))

    #DQN train algorithm
    def train(self):
        self.policy_net.train()
        if len(self.replay_buffer.buffer) < self.min_buffer_size:
            return 0,0
        print("update policy...")

        state, action, next_state, reward, done = self.replay_buffer.sample(self.batch_size)

        state_batch = torch.FloatTensor(state).to("cuda:0")
        action_batch = torch.FloatTensor(action).to("cuda:0")
        next_state_batch = torch.FloatTensor(next_state).to("cuda:0")
        reward_batch = torch.FloatTensor(reward).to("cuda:0")
        done_batch = torch.FloatTensor(done).to("cuda:0")

        q_values = self.policy_net(state_batch).squeeze(1).gather(0, action_batch.unsqueeze(1).type(torch.int64)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).squeeze(1)
            next_q_values,_ = torch.max(next_q_values,1)
        expected_q_values = (next_q_values * self.gamma) * (1 - done_batch) + reward_batch

        loss = self.lossfn(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.tick_counter += 1
        if self.tick_counter % self.target_update == 0:
            self.update_target()

        return loss.item(),0

    #copy policy network parameters to target network
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    #set train mode on policy network. Target networks is always set to evaluation mode.
    def set_mode_train(self):
        self.policy_net.train()