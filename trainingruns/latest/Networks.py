import torch
import torch.nn as nn
import torch.utils
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import KFold
from torch.utils.data import Subset

input_size = 6
hidden_size = 220
hidden_layers = 1
output_size = 1
batch_size =16

trainingloss = []
testingloss = []

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture
        super(Action_Conditioned_FF, self).__init__()
        self.input_to_hidden = nn.Linear(input_size, hidden_size, bias=True)
        self.nonlinear_activation = nn.Sigmoid()
        # self.dropout = nn.Dropout(0.4)
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
        test_loss /= num_batches
        correct /= size
        # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return test_loss


def main():
    model = Action_Conditioned_FF()
    
    # kfold = KFold(n_splits=10,shuffle=False)
    # kfolddataset = dataloaders.nav_dataset
    # for kfold, (trainids, valids) in enumerate(kfold.split(kfolddataset)):
    #     print(f"kfold {kfold+1}\n-------------------------------")
    #     kfoldtrain = Subset(kfolddataset, trainids)
    #     kfoldtest = Subset(kfolddataset, valids)
    #     train_dataloader = torch.utils.data.DataLoader(kfoldtrain, batch_size=batch_size)
    #     test_dataloader = torch.utils.data.DataLoader(kfoldtest, batch_size=batch_size)
    #     train_loss= train( model, train_dataloader, loss_fn, optimizer)
    #     # test_loss = model.evaluate( model, test_dataloader, loss_fn)
    #     test_loss = test(model, test_dataloader, loss_fn)
    #     print(abs(train_loss-test_loss))
    #     if 0 <= abs(train_loss-test_loss) <= .002:
    #         print('Stopping training with training loss', train_loss , " and test loss", test_loss)
    #         print('saved the model with ', kfold, 'iterations')
    #         n = "model"+str(kfold)+".pth"
    #         torch.save(model.state_dict(), n)
    #         savedmodels.append(n)
    #         print("Saved PyTorch Model State to", n)
    #     print(f"Test Error: Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    main()
