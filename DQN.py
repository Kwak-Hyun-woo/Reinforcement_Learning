import torch

class DQNet(torch.nn.Module):
    def __init__(self, emb_size, hidden_size, out_size):
        super(DQNet, self).__init__()
        self.fc1 = torch.nn.Linear(emb_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, out_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc2(self.relu(self.fc1(x.squeeze()))))

