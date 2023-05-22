import numpy
import torch
from models import Autoencoder
from sklearn.metrics import recall_score, f1_score, precision_recall_curve, PrecisionRecallDisplay, average_precision_score
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
        
        self.train_loss = torch.nn.L1Loss()
        self.test_loss = torch.nn.L1Loss(reduction="none")

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
        loss = self.train_loss(logits, x)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def validate(self, loader, threshold=None):
        # notes:
        # - done -input parameter to be added: threshold with the default value equal to None.
        # - done - look at these links:
        #   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.PrecisionRecallDisplay.html#sklearn.metrics.PrecisionRecallDisplay.from_predictions
        #   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score
        #   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve
        # - done - return variables: if threshold==None -> (AP, optimal threshold)
        #                     else -> F1 score
        # - plot the precision recall curve (commented)
        # - use self.test_loss as loss function
        # - I don't think it makes any change but for the loss function the first parameter should be the output and the second
        #   parameter should be the expected output (opposite to what you did)
        # - label anomaly as 1 and normal data as 0 (opposite to what you did)
        # - done - move self.model.train() to the end (before return)

        self.model.eval()

        # if threshold is not None:
        #     predicted = list()
        target = list()
        loss_scores = list()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                # loss = torch.mean(self.test_loss(logits, x), dim=2)
                # loss = self.test_loss(logits, x).view(x.shape[0], -1).mean(1)
                loss = self.test_loss(logits, x).mean(2).view(target.shape[0], -1)
                loss_scores.append(loss)
                # target.append(y.ravel())
                target.append(y)

        target_labels = torch.cat(target, dim=0).cpu()
        loss_scores = torch.cat(loss_scores, dim=0).cpu()
        precision, recall, thresholds = precision_recall_curve(target_labels, loss_scores)
        
        # Plot precision-recall curve
        # disp = PrecisionRecallDisplay(precision, recall)
        # disp.plot()
        # plt.show()

        self.model.train()

        if threshold is None:
            f1 = 2 * (precision * recall) / (precision + recall)
            # average_precision = average_precision_score(target_labels, loss_scores.ravel())
            ap = average_precision_score(target_labels, loss_scores)
            optimal_threshold = thresholds[numpy.where( f1 == max(f1, key=lambda x:x) )]
            return ap, optimal_threshold
        else:
            predicted = (loss_scores.numpy() >= optimal_threshold)
            return f1_score(target_labels, predicted)


    def evaluate(self, loader, threshold):
        # notes:
        # - done - input parameter to be added: threshold
        # - done -return variable: recall
        # - done - use self.test_loss as loss function

        self.model.eval()

        predicted = list()
        target = list()
        loss_scores = list()

        with torch.no_grad():
            for x, y in loader:
                # remember using to(self.device) method for both input and output
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)

                loss = torch.mean(self.test_loss(logits, x), dim=2)
                predicted_y = (loss > threshold ).type(torch.int32).ravel()

                loss_scores.append(loss.ravel())
                predicted.append(predicted_y)
                target.append(y.ravel())

        target_labels = torch.cat(target, dim=0)
        predicted = torch.cat(predicted, dim=0)
        # loss_scores = torch.cat(loss_scores, dim=0)

        # dict_lebel_score[0] = loss_scores [loss_scores >= self.opt['threshold']]
        # dict_lebel_score[1] = loss_scores [loss_scores < self.opt['threshold']]
        #
        # fpr, tpr, thresholds = metrics.roc_curve(target, loss_scores.ravel(), pos_label=1)
        # roc_auc = metrics.auc(fpr, tpr)
        #
        # recall = recall_score(target, predicted)
        # precision = precision_score(target, predicted)
        # f1 = f1_score(target, predicted)

        #  New
        # precision, recall, thresholds = precision_recall_curve(target_labels, loss_scores.ravel())
        # f1 = 2 * (precision * recall) / (precision + recall)
        # average_precision = numpy.mean(precision)
        # optimal_threshold = thresholds[numpy.where(f1 == max(f1, key=lambda x: x))]

        # average_precision = average_precision_score(target_labels, loss_scores.ravel())
        self.model.train()
        return recall_score(target_labels, predicted)