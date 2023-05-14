import torch
from models import Autoencoder

class AutoencoderExperiment: 
    
    def __init__(self, opt):
        # Utils
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # Setup model
        self.model = Autoencoder(opt['window_size'], opt['latent_size'])
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        # Setup optimization procedure
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.criterion = [...]

    def save_checkpoint(self, path, iteration, total_train_loss):
        checkpoint = {}

        checkpoint['iteration'] = iteration
        checkpoint['total_train_loss'] = total_train_loss
        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        iteration = checkpoint['iteration']
        total_train_loss = checkpoint['total_train_loss']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return iteration, total_train_loss

    def train_iteration(self, data):
        # remember using to(self.device) method for input


        




        logits = [...]
        loss = [...]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def validate(self, loader, threshold):
        self.model.eval()
        accuracy = 0
        loss = 0
        with torch.no_grad():
            for x, y in loader:
                # remember using to(self.device) method for both input and output









                logits = [...]
                loss += [...]
                
                pred = [...]
                accuracy += (pred == y).sum().item()

        self.model.train()
        return accuracy, loss