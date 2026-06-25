from torch.utils.data import Dataset

class ReplayBuffer(Dataset):
    def __init__(self):
        self.states      = []
        self.next_states = []
        self.rewards     = []
        self.actions     = []

    def __len__(self):
        # Return number of samples
        return len(self.actions)

    def __getitem__(self, idx):
        # Return one sample
        return self.states[idx], self.next_states[idx], self.rewards[idx], self.actions[idx]

    def additem(self, state, next_state, reward, action):
        self.states.append(state)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.actions.append(action)