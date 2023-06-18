import torch
from models import AdversarialAutoEncoder
import numpy as np
from sklearn.metrics import recall_score, f1_score, precision_recall_curve, PrecisionRecallDisplay, average_precision_score
import matplotlib.pyplot as plt

class AAEExperiment: 
    
    def __init__(self, opt):
        # Utils
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # Setup model
        self.model = AdversarialAutoEncoder(opt['window_size'], opt['latent_size'])
        self.model.train()
        self.model.to(self.device)

        # Setup optimization procedure
        self.gen_optimizer = torch.optim.Adam(self.model.autoencoder.parameters(), lr=opt['lr'])
        self.disc_optimizer = torch.optim.Adam(self.model.discriminator.parameters(), lr=opt['lr'])
        
        self.generator_loss = torch.nn.MSELoss()
        self.discriminator_loss = torch.nn.BCELoss()
        self.test_loss = torch.nn.MSELoss(reduction="none")

    def save_checkpoint(self, path, iteration, bestAP, threshold):
        checkpoint = {}

        checkpoint['iteration'] = iteration
        checkpoint['bestAP'] = bestAP
        checkpoint['threshold'] = threshold
        checkpoint['model'] = self.model.state_dict()
        checkpoint['gen_optimizer'] = self.gen_optimizer.state_dict()
        checkpoint['disc_optimizer'] = self.disc_optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        iteration = checkpoint['iteration']
        bestAP = checkpoint['bestAP']
        self.threshold = checkpoint['threshold']
        self.model.load_state_dict(checkpoint['model'])
        self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        self.disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])

        return iteration, bestAP

    def draw_plot(self, precision, recall, ap):
        # precision2 = np.load('precision.npy')
        # recall2 = np.load('recall.npy')

        disp = PrecisionRecallDisplay(precision, recall, estimator_name='AAE', average_precision=ap)
        # disp2 = PrecisionRecallDisplay(precision2, recall2, estimator_name='AE', average_precision=0.7652)

        fig, ax = plt.subplots()

        # disp2.plot(ax=ax)
        disp.plot(ax=ax)

        ax.set_title('W = 100, L = 30, \u03BB = 0.01')
        ax.legend(loc='upper right')
        plt.show()
    
    def train_iteration(self, data):
        x, _ = data
        x = x.to(self.device)
        # Freeze generator weights
        for param in self.model.autoencoder.parameters():
            param.requires_grad = False
        for param in self.model.discriminator.parameters():
            param.requires_grad = True
        # Autoencoder output
        logits = self.model.autoencoder(x)
        # Discriminator outputs
        real_pred = self.model.discriminator(x) # discriminator output for real input
        fake_pred = self.model.discriminator(logits) # discriminator output for fake input
        # Update discriminator weights
        disc_loss1 = self.discriminator_loss(real_pred, torch.ones((real_pred.shape)).to(self.device))
        disc_loss2 = self.discriminator_loss(fake_pred, torch.zeros((fake_pred.shape)).to(self.device))
        disc_loss = (disc_loss1 + disc_loss2) / 2

        # R1 regularization
        x.requires_grad = True
        mean_outputs = torch.mean(self.model.discriminator(x)) # Feed the discriminator with real data
        gradients = torch.autograd.grad(outputs=mean_outputs, inputs=x)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = torch.mean(torch.sum(gradients**2, dim=1))
        disc_loss += 0.1 * gradient_penalty

        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()

        x.requires_grad = False
        # Freeze discriminator weights
        for param in self.model.autoencoder.parameters():
            param.requires_grad = True
        for param in self.model.discriminator.parameters():
            param.requires_grad = False
        # Autoencoder output
        logits = self.model.autoencoder(x)
        # New discriminator outputs
        real_pred = self.model.discriminator(x) # discriminator output for real input
        fake_pred = self.model.discriminator(logits) # discriminator output for fake input
        # Update generator weights
        gen_loss = self.generator_loss(logits, x)
        reg_term = self.generator_loss(real_pred, fake_pred)
        loss = gen_loss + self.opt['reg_weight'] * reg_term
        
        self.gen_optimizer.zero_grad()
        loss.backward()
        self.gen_optimizer.step()
        
        return loss.item()

    def validate(self, loader, threshold=None):

        self.model.eval()
        target = list()
        loss_scores = list()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model.autoencoder(x)
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

                logits = self.model.autoencoder(x)
                loss = self.test_loss(logits, x).view(x.shape[0], -1).mean(1)
                predicted.append(loss >= threshold)
                target.append(y)

        target_labels = torch.cat(target, dim=0).cpu()
        predicted = torch.cat(predicted, dim=0).cpu()
        self.model.train()
        return recall_score(target_labels, predicted)