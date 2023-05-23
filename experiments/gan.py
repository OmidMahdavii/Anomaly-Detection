import torch
from models import GAN
import numpy
from sklearn.metrics import recall_score, f1_score, precision_recall_curve, PrecisionRecallDisplay, average_precision_score
import matplotlib.pyplot as plt

class GANExperiment: 
    
    def __init__(self, opt):
        # Utils
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # Setup model
        self.model = GAN(opt['window_size'], opt['latent_size'])
        self.model.train()
        self.model.to(self.device)

        # Setup optimization procedure
        self.gen_optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.disc_optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        
        self.generator_loss = torch.nn.L1Loss()
        self.discriminator_loss = torch.nn.BCELoss()
        self.test_loss = torch.nn.L1Loss(reduction="none")

    def save_checkpoint(self, path, iteration, bestAP):
        checkpoint = {}

        checkpoint['iteration'] = iteration
        checkpoint['bestAP'] = bestAP
        checkpoint['model'] = self.model.state_dict()
        checkpoint['gen_optimizer'] = self.gen_optimizer.state_dict()
        checkpoint['disc_optimizer'] = self.disc_optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        iteration = checkpoint['iteration']
        bestAP = checkpoint['bestAP']
        self.model.load_state_dict(checkpoint['model'])
        self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        self.disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])

        return iteration, bestAP

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

        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()

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
                # loss = torch.mean(self.test_loss(logits, x), dim=2)
                loss = self.test_loss(logits, x).view(x.shape[0], -1).mean(1)
                # loss = self.test_loss(logits, x).mean(2).view(target.shape[0], -1)
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
            ap = average_precision_score(target_labels, loss_scores)
            optimal_threshold = thresholds[numpy.where(f1 == max(f1, key=lambda x: x))]
            return ap, float(optimal_threshold)
        else:
            predicted = (loss_scores >= threshold)
            return f1_score(target_labels, predicted)

        # if threshold is not None:
        #     predicted = list()
        # target = list()
        # loss_scores = list()
        #
        # with torch.no_grad():
        #     for x, y in loader:
        #         # remember using to(self.device) method for both input and output
        #         x = x.to(self.device)
        #         y = y.to(self.device)
        #
        #         logits = self.model(x)
        #
        #         loss = torch.mean(self.test_loss(logits, x), dim=2)
        #
        #         if threshold is not None:
        #             predicted_y = (loss > threshold).type(torch.int32).ravel()
        #
        #         loss_scores.append(loss.ravel())
        #         target.append(y.ravel())
        #         if threshold is not None:
        #             predicted.append(predicted_y)
        #
        # target_labels = torch.cat(target, dim=0)
        # loss_scores = torch.cat(loss_scores, dim=0)
        # if threshold is not None:
        #     predicted = torch.cat(predicted, dim=0)
        #
        # # dict_lebel_score[0] = loss_scores [loss_scores >= self.opt['threshold']]
        # # dict_lebel_score[1] = loss_scores [loss_scores < self.opt['threshold']]
        # #
        # # fpr, tpr, thresholds = metrics.roc_curve(target, loss_scores.ravel(), pos_label=1)
        # # roc_auc = metrics.auc(fpr, tpr)
        # #
        # # recall = recall_score(target, predicted)
        # # precision = precision_score(target, predicted)
        # # f1 = f1_score(target, predicted)
        #
        # precision, recall, thresholds = precision_recall_curve(target_labels, loss_scores.ravel())
        # f1 = 2 * (precision * recall) / (precision + recall)
        # average_precision = numpy.mean(precision)
        # # average_precision = average_precision_score(target_labels, loss_scores.ravel())
        # optimal_threshold = thresholds[numpy.where( f1 == max(f1, key=lambda x:x) )]
        #
        #
        # self.model.train()
        # if threshold is not None:
        #     return f1_score(target_labels, predicted)
        # else:
        #     return average_precision, optimal_threshold


    def evaluate(self, loader, threshold):

        self.model.eval()
        target = list()
        predicted = list()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model.autoencoder(x)
                # loss = torch.mean(self.test_loss(logits, x), dim=2)
                loss = self.test_loss(logits, x).view(x.shape[0], -1).mean(1)
                predicted.append(loss >= threshold)
                # loss = self.test_loss(logits, x).mean(2).view(target.shape[0], -1)
                # loss_scores.append(loss)
                # target.append(y.ravel())
                target.append(y)

        target_labels = torch.cat(target, dim=0).cpu()
        predicted = torch.cat(predicted, dim=0).cpu()
        self.model.train()
        return recall_score(target_labels, predicted)

        # self.model.eval()
        #
        # predicted = list()
        # target = list()
        # loss_scores = list()
        #
        # with torch.no_grad():
        #     for x, y in loader:
        #         # remember using to(self.device) method for both input and output
        #         x = x.to(self.device)
        #         y = y.to(self.device)
        #
        #         logits = self.model(x)
        #
        #         loss = torch.mean(self.test_loss(logits, x), dim=2)
        #         predicted_y = (loss > threshold).type(torch.int32).ravel()
        #
        #         loss_scores.append(loss.ravel())
        #         predicted.append(predicted_y)
        #         target.append(y.ravel())
        #
        # target_labels = torch.cat(target, dim=0)
        # predicted = torch.cat(predicted, dim=0)
        # # loss_scores = torch.cat(loss_scores, dim=0)
        # # average_precision = average_precision_score(target_labels, loss_scores.ravel())
        # self.model.train()
        # return recall_score(target_labels, predicted)