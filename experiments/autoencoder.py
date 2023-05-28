import numpy
import torch
from models import Autoencoder
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, PrecisionRecallDisplay, average_precision_score
import matplotlib.pyplot as plt

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
        
        self.train_loss = torch.nn.MSELoss()
        self.test_loss = torch.nn.MSELoss(reduction="none")

    def save_checkpoint(self, path, iteration, bestAP):
        checkpoint = {}

        checkpoint['iteration'] = iteration
        checkpoint['bestAP'] = bestAP
        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        iteration = checkpoint['iteration']
        bestAP = checkpoint['bestAP']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return iteration, bestAP

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

        loss_scores = (loss_scores - torch.min(loss_scores)) / (torch.max(loss_scores) - torch.min(loss_scores))
    
        precision, recall, thresholds = precision_recall_curve(target_labels, loss_scores)

        # Plot precision-recall curve
        # disp = PrecisionRecallDisplay(precision, recall)
        # disp.plot()
        # plt.show()

        self.model.train()

        # if threshold is None:
        #     precision = []
        #     recall = []
        #     thresholds = []
        #     ap = 0
        #     max_f1 = 0
        #     optimal_threshold = 0

        #     for i in torch.sort(loss_scores, descending=True)[0]:
        #         thresholds.append(i)
        #         labels = (loss_scores >= i)

        #         p = precision_score(target_labels, labels)
        #         r = recall_score(target_labels, labels)
        #         ap += (r - recall[-1])*p if len(recall)!= 0 else r*p

        #         f = f1_score(target_labels, labels)
        #         if f > max_f1:
        #             optimal_threshold = float(i)
        #             max_f1 = f
        #         precision.append(p)
        #         recall.append(r)  

        #     return ap, float(optimal_threshold)
        # else:
        #     predicted = (loss_scores >= threshold)
        #     return f1_score(target_labels, predicted)



        if threshold is None:
            f1 = 2 * (precision * recall) / (precision + recall)
            ap = average_precision_score(target_labels, loss_scores)
            optimal_threshold = thresholds[numpy.where(f1 == max(f1))]
            return ap, float(optimal_threshold)
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