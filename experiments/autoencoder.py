import numpy as np
import torch
from models import Autoencoder
from sklearn.metrics import recall_score, f1_score, precision_recall_curve, PrecisionRecallDisplay, average_precision_score
import matplotlib.pyplot as plt

class AEExperiment: 
    
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
        
        self.train_loss = torch.nn.MSELoss()
        self.test_loss = torch.nn.MSELoss(reduction="none")

    def save_checkpoint(self, path, iteration, bestAP, threshold):
        checkpoint = {}

        checkpoint['iteration'] = iteration
        checkpoint['bestAP'] = bestAP
        checkpoint['threshold'] = threshold
        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        iteration = checkpoint['iteration']
        bestAP = checkpoint['bestAP']
        self.threshold = checkpoint['threshold']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return iteration, bestAP

    def draw_plot(self, precision, recall, ap):
        # np.save('precision', precision)
        # np.save('recall', recall)
        
        disp = PrecisionRecallDisplay(precision, recall, estimator_name='Autoencoder', average_precision=ap)
        disp.plot()
        plt.show()
    
    def train_iteration(self, data):
        x, _ = data
        x = x.to(self.device)

        logits = self.model(x)
        loss = self.train_loss(logits, x)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def validate(self, loader, threshold=None):
        self.model.eval()
        target = list()
        loss_scores = list()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss = self.test_loss(logits, x).view(x.shape[0], -1).mean(1)
                loss_scores.append(loss)
                target.append(y)

        target_labels = torch.cat(target, dim=0).cpu()
        loss_scores = torch.cat(loss_scores, dim=0).cpu()

        # Normalize scores in [0, 1]
        # loss_scores = (loss_scores - torch.min(loss_scores)) / (torch.max(loss_scores) - torch.min(loss_scores))
    
        self.model.train()

        if threshold is None:
            precision, recall, thresholds = precision_recall_curve(target_labels, loss_scores)

            f1 = 2 * (precision * recall) / (precision + recall)
            ap = average_precision_score(target_labels, loss_scores)
            optimal_threshold = thresholds[np.where(f1 == max(f1))][0]

            # Plot precision-recall curve
            # self.draw_plot(precision, recall, ap)

            return ap, optimal_threshold
        else:
            predicted = (loss_scores >= threshold)
            return f1_score(target_labels, predicted)


    def evaluate(self, loader, threshold):
        self.model.eval()
        target = list()
        predicted = list()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss = self.test_loss(logits, x).view(x.shape[0], -1).mean(1)
                predicted.append(loss >= threshold)
                target.append(y)

        target_labels = torch.cat(target, dim=0).cpu()
        predicted = torch.cat(predicted, dim=0).cpu()
        self.model.train()
        return recall_score(target_labels, predicted)