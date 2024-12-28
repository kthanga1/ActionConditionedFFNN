import torch
import torch.nn as nn
import torch.utils
import Data_Loaders as DataLoader
import matplotlib.pyplot as plt
import math
import os
from sklearn.model_selection import KFold
from torch.utils.data import Subset

from torcheval.metrics import BinaryF1Score
from torcheval.metrics import BinaryPrecision
from torcheval.metrics import BinaryConfusionMatrix



input_size = 6
hidden_size = 220
hidden_layers = 1
output_size = 1
batch_size =16

trainingloss = []
testingloss = []
trainingf1 = []
testingf1 = []
testingprecision = []

f1metric = BinaryF1Score()
precisionmetric = BinaryPrecision()
confusionmatrix = BinaryConfusionMatrix()

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture
        super(Action_Conditioned_FF, self).__init__()
        self.input_to_hidden = nn.Linear(input_size, hidden_size, bias=True)
        self.nonlinear_activation = nn.Sigmoid()
        # self.dropout = nn.Dropout(0.5)
        self.hidden_to_output = nn.Linear(hidden_size,output_size, bias=True)

    def forward(self, network_input):
# STUDENTS: forward() must complete a single forward pass through your network
# and return the output which should be a tensor
        hidden = self.input_to_hidden(network_input)
        hidden  = self.nonlinear_activation(hidden)
        # hidden = self.dropout(hidden)
        network_output = self.hidden_to_output(hidden)
        network_output = self.nonlinear_activation(network_output)
        return network_output


    # def init_hidden_state(self, batchsize, hiddensize):
    #     hidden_state = torch.zeros(batchsize, hiddensize)
    #     return hidden_state

    def evaluate(self, model, test_loader, loss_function):
# STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
# mind that we do not need to keep track of any gradients while evaluating the
# model. loss_function will be a PyTorch loss function which takes as argument the model's
# output and the desired output.
        size = len(test_loader.dataset)
        num_batches = len(test_loader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for  batch, sample in enumerate(test_loader):
                X = sample['input']
                y = sample['label']
                pred = model(X)
                test_loss += loss_function(pred, y).item()
                # print(test_loss)
                pred = (pred > 0.5).type(torch.float)
                correct += (pred  == y).type(torch.float).sum().item()
                # confusionmatrix.update(input=pred.view(pred.size()[0]), target=y.view(y.size()[0]))

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return test_loss


def train(model, train_dataloader, lossfn, optimizer ):
    size = len(train_dataloader.dataset)
    model.train()
    trainrunloss= 0
    for batch, sample in enumerate(train_dataloader):
        X = sample['input']
        y = sample['label']
        pred = model(X) 
        loss = lossfn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        trainrunloss += loss.item() * X.size(0)
    print(f" Training loss: {trainrunloss/size:>7f}")
    trainingloss.append(trainrunloss/size)

    return trainrunloss/size


def test(model, dataloader, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    f1score = 0
    precisionscore =0
    

    with torch.no_grad():
        for  batch, sample in enumerate(dataloader):
            X = sample['input']
            y = sample['label']
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred = (pred > 0.5).type(torch.float)
            correct += (pred == y).type(torch.float).sum().item()
            f1metric.update(input=pred.view(pred.size()[0]), target=y.view(y.size()[0]))
            f1score += f1metric.compute()
            precisionmetric.update(input=pred.view(pred.size()[0]), target=y.view(y.size()[0]))
            precisionscore += precisionmetric.compute()
            
    test_loss /= num_batches
    correct /= size
    testingloss.append(test_loss)
    testingf1.append(f1score/num_batches)
    testingprecision.append(precisionscore/num_batches)
    print()
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Test: \n F1 Score: {f1score/num_batches} ")
    print(f"Test: \n Precision Score: {precisionscore/num_batches}")    
    return test_loss



def main():
    dataloaders = DataLoader.Data_Loaders(batch_size=batch_size)
    train_dataloader = dataloaders.train_loader
    test_dataloader = dataloaders.test_loader

    model = Action_Conditioned_FF()
    # print(model)

    loss_fn = nn.BCELoss()
    # print(loss_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    epochs = 30
    savedmodels = []

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss= train( model, train_dataloader, loss_fn, optimizer)
        # test_loss = model.evaluate( model, test_dataloader, loss_fn)
        test_loss = test(model, test_dataloader, loss_fn)
        print(abs(train_loss-test_loss))
        # if 0 <= abs(train_loss-test_loss) <= .002:
        #     print('Stopping training with training loss', train_loss , " and test loss", test_loss)
        #     print('saved the model with ', t, 'iterations')
        #     n = "model"+str(t)+".pth"
        #     torch.save(model.state_dict(), n)
        #     savedmodels.append(n)
        #     print("Saved PyTorch Model State to", n)
        print(f"Test Error: Avg loss: {test_loss:>8f} \n")

    print("Done Training!")

    print(len(trainingloss))
    print(len(testingloss))
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(testingloss,label="testing")
    plt.plot(trainingloss,label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig('run.png')

    plt.figure(figsize=(10,5))
    plt.title("Training and Validation F1 Score")
    plt.plot(testingf1,label="testing")
    plt.plot(trainingf1,label="train")
    plt.xlabel("iterations")
    plt.ylabel("F1Score")
    plt.legend()
    plt.show()
    plt.savefig('F1score.png')

    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Precision Score")
    plt.plot(testingprecision,label="testing")
    # plt.plot(trainingf1,label="train")
    plt.xlabel("iterations")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()
    plt.savefig('PrecisonScore.png')

    torch.save(model.state_dict(), "final.pth")
    print("Saved PyTorch Model State to final.pth")


    # for n in savedmodels:
    #     model = Action_Conditioned_FF()
    #     model.load_state_dict(torch.load(n, weights_only=True))
    #     print("Evaluating model", n)
    #     test_loss = model.evaluate(model, test_dataloader,loss_fn)
    #     print("Evaluation test loss for model" , n , " is --> ", test_loss)
    
    model = Action_Conditioned_FF()
    model.load_state_dict(torch.load("final.pth", weights_only=True))
    print("Evaluating final model")
    test_loss = model.evaluate(model, test_dataloader,loss_fn)
    print("Evaluation test loss for final model is --> ", test_loss)


def cleeanmodels():
    if os.path.exists("./*.pth"):
        f = open()


if __name__ == '__main__':
    main()

