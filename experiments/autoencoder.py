import torch
from models import Autoencoder
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, RocCurveDisplay

from parse_args import parse_arguments

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
        # loss function of autoencoder
        self.criterion = self.model.L1Loss(reduction="none")

    def save_checkpoint(self, path, iteration):
        checkpoint = {}

        checkpoint['iteration'] = iteration
        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        iteration = checkpoint['iteration']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return iteration

    def train_iteration(self, data):
        x, _ = data
        x = x.to(self.device)

        logits = self.model(x)
        loss = self.criterion(logits, x)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def validate(self, loader):
        # The threshold is saved in the self.opt['threshold'] variable
        self.model.eval()

        accuracy = 0

        predicted = list()
        target = list()
        loss_scores = list()

        dict_lebel_score = dict()

        with torch.no_grad():
            for x, y in loader:
                # remember using to(self.device) method for both input and output
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)

                loss = torch.mean(self.criterion(x, logits), dim=2)

                predicted_y = loss > self.opt['threshold'].type(torch.int32).ravel()

                loss_scores.append(loss.ravel())
                predicted.append(predicted_y)
                target.append(y)

                # accuracy += (pred == y).sum().item()

        self.model.train()

        target = torch.cat(target, dim=0)
        predicted = torch.cat(predicted, dim=0)
        loss_scores = torch.cat(loss_scores, dim=0)

        dict_lebel_score[0] = loss_scores [loss_scores >= self.opt['threshold']]
        dict_lebel_score[1] = loss_scores [loss_scores < self.opt['threshold']]

        RocCurveDisplay.from_predictions(target, loss_scores, name= "ROC curve", color="darkorange")

        cm = confusion_matrix(target, predicted)
        recall = recall_score(target, predicted)
        precision = precision_score(target, predicted)
        f1 = f1_score(target, predicted)

        return accuracy, f1

    def find_optimal_for_f1(self, scores, groundTruth):
        list_f1_th = list()
        thresholds = set(scores)
        for th in thresholds:
            predicted_label = (scores >= th).type(torch.int32).ravel()
            f1 = f1_score(groundTruth, predicted_label)
            list_f1_th.append((f1, th))

        return max(list_f1_th, key=lambda x: x[0])[1]

    def evaluate(self, loader):
        self.model.eval()
        accuracy = 0
        predicted = list()
        target = list()
        loss_scores = list()
        dict_lebel_score = dict()

        with torch.no_grad():
            for x, y in loader:
                # remember using to(self.device) method for both input and output
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)

                loss = torch.mean(self.criterion(x, logits), dim=2)

                predicted_y = loss > self.opt['threshold'].type(torch.int32).ravel()

                loss_scores.append(loss.ravel())
                predicted.append(predicted_y)
                target.append(y)

                # accuracy += (pred == y).sum().item()

        self.model.train()

        target = torch.cat(target, dim=0)
        predicted = torch.cat(predicted, dim=0)
        loss_scores = torch.cat(loss_scores, dim=0)

        dict_lebel_score[0] = loss_scores[loss_scores >= self.opt['threshold']]
        dict_lebel_score[1] = loss_scores[loss_scores < self.opt['threshold']]

        RocCurveDisplay.from_predictions(target, loss_scores, name="ROC curve", color="darkorange")

        cm = confusion_matrix(target, predicted)
        recall = recall_score(target, predicted)
        precision = precision_score(target, predicted)
        f1 = f1_score(target, predicted)

        return accuracy, f1

