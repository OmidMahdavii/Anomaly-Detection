import torch
from models import GAN
from sklearn import metrics
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
        target = torch.zeros((x.shape[0])).to(self.device)
        x = x.to(self.device)

        logits, l = self.model(x)
        loss1 = self.criterion(logits, x)
        loss2 = self.criterion(l, target)
        loss = loss1 + self.opt['reg_weight'] * loss2

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
        predicted_discr_with_recinstructed_input = list()
        predicted_discr_with_real_input = list()



        with torch.no_grad():
            for x, y in loader:
                # remember using to(self.device) method for both input and output

                x = x.to(self.device)
                y = y.to(self.device)

                logits,  discriminator_y, discriminator_x= self.model(x)

                loss = torch.mean(self.criterion(x, logits), dim=2)

                predicted_y = loss > self.opt['threshold'].type(torch.int32).ravel()

                loss_scores.append(loss.ravel())
                predicted_ae.append(predicted_y)
                predicted_discr_with_recinstructed_input.append(discriminator_y.ravel())
                predicted_discr_with_real_input.append(discriminator_x.ravel())
                target.append(y)

                
                # pred = [...]
                # accuracy += (pred == y).sum().item()

        self.model.train()

        target = torch.cat(target, dim=0)
        predicted_ae = torch.cat(predicted_ae, dim=0)
        loss_scores = torch.cat(loss_scores, dim=0)
        predicted_discr_with_recinstructed_input = torch.cat(predicted_discr_with_recinstructed_input, dim=0)
        predicted_discr_with_real_input = torch.cat(predicted_discr_with_real_input, dim=0)

        # RocCurveDisplay.from_predictions(target, loss_scores, name="ROC curve", color="darkorange")

        fpr, tpr, thresholds = metrics.roc_curve(target, loss_scores.ravel(), pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)


        confusion_matrix_ae = confusion_matrix(target, predicted_ae)
        confusion_matrix_discr = confusion_matrix(target, predicted_discr_with_recinstructed_input)

        recall_ae = recall_score(target, predicted_ae)
        precision_ae = precision_score(target, predicted_ae)
        f1_ae = f1_score(target, predicted_ae)

        recall_discr_with_constructed_input = recall_score(target, predicted_discr_with_recinstructed_input)
        precision_discr_with_constructed_input = precision_score(target, predicted_discr_with_recinstructed_input)
        f1_discr_with_constructed_input = f1_score(target, predicted_discr_with_recinstructed_input)

        recall_discr = recall_score(target, predicted_discr_with_real_input)
        precision_discr = precision_score(target, predicted_discr_with_real_input)
        f1_discr = f1_score(target, predicted_discr_with_real_input)

        return accuracy, loss

    def evaluate(self, loader):

        # The threshold is saved in the self.opt['threshold'] variable
        self.model.eval()

        accuracy = 0
        loss = 0

        predicted_ae = list()
        target = list()
        loss_scores = list()
        predicted_discr_with_recinstructed_input = list()
        predicted_discr_with_real_input = list()

        with torch.no_grad():
            for x, y in loader:
                # remember using to(self.device) method for both input and output

                x = x.to(self.device)
                y = y.to(self.device)

                logits, discriminator_y, discriminator_x = self.model(x)

                loss = torch.mean(self.criterion(x, logits), dim=2)

                predicted_y = loss > self.opt['threshold'].type(torch.int32).ravel()

                loss_scores.append(loss.ravel())
                predicted_ae.append(predicted_y)
                predicted_discr_with_recinstructed_input.append(discriminator_y.ravel())
                predicted_discr_with_real_input.append(discriminator_x.ravel())
                target.append(y)

                # pred = [...]
                # accuracy += (pred == y).sum().item()

        self.model.train()

        target = torch.cat(target, dim=0)
        predicted_ae = torch.cat(predicted_ae, dim=0)
        loss_scores = torch.cat(loss_scores, dim=0)
        predicted_discr_with_recinstructed_input = torch.cat(predicted_discr_with_recinstructed_input, dim=0)
        predicted_discr_with_real_input = torch.cat(predicted_discr_with_real_input, dim=0)

        # RocCurveDisplay.from_predictions(target, loss_scores, name="ROC curve", color="darkorange")

        fpr, tpr, thresholds = metrics.roc_curve(target, loss_scores.ravel(), pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)

        confusion_matrix_ae = confusion_matrix(target, predicted_ae)
        confusion_matrix_discr = confusion_matrix(target, predicted_discr_with_recinstructed_input)

        recall_ae = recall_score(target, predicted_ae)
        precision_ae = precision_score(target, predicted_ae)
        f1_ae = f1_score(target, predicted_ae)

        recall_discr_with_constructed_input = recall_score(target, predicted_discr_with_recinstructed_input)
        precision_discr_with_constructed_input = precision_score(target, predicted_discr_with_recinstructed_input)
        f1_discr_with_constructed_input = f1_score(target, predicted_discr_with_recinstructed_input)

        recall_discr = recall_score(target, predicted_discr_with_real_input)
        precision_discr = precision_score(target, predicted_discr_with_real_input)
        f1_discr = f1_score(target, predicted_discr_with_real_input)

        return accuracy, loss

    def find_optimal_for_f1(self, scores, groundTruth):
        list_f1_th = list()
        thresholds = set(scores)
        for th in thresholds:
            predicted_label = (scores >= th).type(torch.int32).ravel()
            f1 = f1_score(groundTruth, predicted_label)
            list_f1_th.append((f1, th))

        return max(list_f1_th, key=lambda x: x[0])[1]

    def find_thr(self, fpr, tpr, thr):

        GOAT = (0, 1)
        distance = list()

        for pair in zip(tpr, fpr):
            distance.append(math.sqrt((pair[0] - GOAT[0]) ** 2 + (pair[1] - GOAT[1]) ** 2))

        print(distance)
        print(min(distance, key=lambda x: x))
        print(distance.index(min(distance, key=lambda x: x)))

        return thr[distance.index(min(distance, key=lambda x: x))]