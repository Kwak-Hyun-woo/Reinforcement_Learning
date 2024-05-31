import torch

class DQNet(torch.nn.Module):
    def __init__(self, emb_size, hidden_size, out_size):
        super(DQNet, self).__init__()
        self.fc1 = torch.nn.Linear(emb_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, hidden_size // 2)      
        self.fc4 = torch.nn.Linear(hidden_size // 2, out_size)
        # self.fc2 = torch.nn.Linear(hidden_size , out_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
      output = self.relu(self.fc1(x.squeeze()))
      output = self.relu(self.fc2(output))
      output = self.relu(self.fc3(output))
      output = self.relu(self.fc4(output))
      return output

