import torch
from models import GAN
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, RocCurveDisplay

class GANExperiment: 
    
    def __init__(self, opt):
        # Utils
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # Setup model
        self.model = GAN(opt['window_size'], opt['latent_size'])
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        # Setup optimization procedure
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        # loss function of GAN
        self.criterion = self.model.BCEWithLogitsLoss()

    def save_checkpoint(self, path, iteration, train_loss):
        checkpoint = {}

        checkpoint['iteration'] = iteration
        checkpoint['train_loss'] = train_loss
        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        iteration = checkpoint['iteration']
        train_loss = checkpoint['train_loss']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return iteration, train_loss

    def train_iteration(self, data):
        # The regularization weight is saved in the self.opt['reg_weight'] variable
        # remember using to(self.device) method for input


        




        logits = [...]
        loss = [...]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def validate(self, loader):
        # The threshold is saved in the self.opt['threshold'] variable
        self.model.eval()

        accuracy = 0
        loss = 0

        predicted_ae = list()
        target = list()
        loss_scores = list()
        predicted_discr = list()


        with torch.no_grad():
            for x, y in loader:
                # remember using to(self.device) method for both input and output

                x = x.to(self.device)
                y = y.to(self.device)

                logits,  discriminator_y = self.model(x)

                loss = torch.mean(self.criterion(x, logits), dim=2)

                predicted_y = loss > self.opt['threshold'].type(torch.int32).ravel()

                loss_scores.append(loss.ravel())
                predicted_ae.append(predicted_y)
                predicted_discr.append(discriminator_y.ravel())
                target.append(y)

                
                # pred = [...]
                # accuracy += (pred == y).sum().item()

        self.model.train()

        target = torch.cat(target, dim=0)
        predicted_ae = torch.cat(predicted_ae, dim=0)
        loss_scores = torch.cat(loss_scores, dim=0)
        predicted_discr = torch.cat(predicted_discr, dim=0)

        RocCurveDisplay.from_predictions(target, loss_scores, name="ROC curve", color="darkorange")

        confusion_matrix_ae = confusion_matrix(target, predicted_ae)
        confusion_matrix_discr = confusion_matrix(target, predicted_discr)

        recall_ae = recall_score(target, predicted_ae)
        precision_ae = precision_score(target, predicted_ae)
        f1_ae = f1_score(target, predicted_ae)

        recall_discr = recall_score(target, predicted_discr)
        precision_discr = precision_score(target, predicted_discr)
        f1_discr = f1_score(target, predicted_discr)

        return accuracy, loss

    def find_optimal_for_f1(self, scores, groundTruth):
        list_f1_th = list()
        thresholds = set(scores)
        for th in thresholds:
            predicted_label = (scores >= th).type(torch.int32).ravel()
            f1 = f1_score(groundTruth, predicted_label)
            list_f1_th.append((f1, th))

        return max(list_f1_th, key=lambda x: x[0])[1]

    def evaluate(self, loader):

        # The threshold is saved in the self.opt['threshold'] variable
        self.model.eval()

        accuracy = 0
        loss = 0

        predicted_ae = list()
        target = list()
        loss_scores = list()
        predicted_discr = list()

        with torch.no_grad():
            for x, y in loader:
                # remember using to(self.device) method for both input and output

                x = x.to(self.device)
                y = y.to(self.device)

                logits, discriminator_y = self.model(x)

                loss = torch.mean(self.criterion(x, logits), dim=2)

                predicted_y = loss > self.opt['threshold'].type(torch.int32).ravel()

                loss_scores.append(loss.ravel())
                predicted_ae.append(predicted_y)
                predicted_discr.append(discriminator_y.ravel())
                target.append(y)

                # pred = [...]
                # accuracy += (pred == y).sum().item()

        self.model.train()

        target = torch.cat(target, dim=0)
        predicted_ae = torch.cat(predicted_ae, dim=0)
        loss_scores = torch.cat(loss_scores, dim=0)
        predicted_discr = torch.cat(predicted_discr, dim=0)

        RocCurveDisplay.from_predictions(target, loss_scores, name="ROC curve", color="darkorange")

        confusion_matrix_ae = confusion_matrix(target, predicted_ae)
        confusion_matrix_discr = confusion_matrix(target, predicted_discr)

        recall_ae = recall_score(target, predicted_ae)
        precision_ae = precision_score(target, predicted_ae)
        f1_ae = f1_score(target, predicted_ae)

        recall_discr = recall_score(target, predicted_discr)
        precision_discr = precision_score(target, predicted_discr)
        f1_discr = f1_score(target, predicted_discr)

        return accuracy, loss