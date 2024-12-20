import torch
import torch.nn as nn
import torch.utils
import Data_Loaders as DataLoader


device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
print(f"Using {device} device")

input_size = 6
hidden_size = 12
hidden_layers = 1
output_size = 1
batch_size =16

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture
        super(Action_Conditioned_FF, self).__init__()
        self.rnn_layer = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=hidden_layers, batch_first=False, bias=True, nonlinearity='relu')
        self.fc_layer = nn.Linear(hidden_size,output_size)

    def forward(self, network_input):
# STUDENTS: forward() must complete a single forward pass through your network
# and return the output which should be a tensor
        print(network_input.size())
        hidden_state = self.init_hidden_state(batchsize=1, hiddensize=hidden_size)
        print(hidden_state.size())
        network_output,  hidden_final  = self.rnn_layer(network_input, hidden_state)
        print(network_output.size())
        network_output = self.fc_layer(network_output[:,-1,:])
        return network_output


    def init_hidden_state(self, batchsize, hiddensize):
        hidden_state = torch.zeros(batchsize, hiddensize)
        return hidden_state

    def evaluate(self, model, test_loader, loss_function):
# STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
# mind that we do not need to keep track of any gradients while evaluating the
# model. loss_function will be a PyTorch loss function which takes as argument the model's
# output and the desired output.
        return loss_function(model, test_loader)


def train(model, train_dataloader, lossfn, optimizer ):
    # size = len(train_dataloader.dataset)
    model.train()
    for batch, sample in enumerate(train_dataloader):
        X = sample['input']
        y = sample['label']
        print(X)
        print(y.size())
        X , y = X.to(device), y.to(device)
        pred = model(X)
        loss = model.evaluate(pred, y)
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    print(type(dataloader))
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



def main():
    dataloaders = DataLoader.Data_Loaders(batch_size=None)
    train_dataloader = dataloaders.train_loader
    test_dataloader = dataloaders.test_loader
    print(test_dataloader.dataset)


    # model = Action_Conditioned_FF().to(device)
    # print(model)

    # loss_fn = nn.CrossEntropyLoss()
    # print(loss_fn)
    # optimizer = torch.optim.Adam(model.parameters(), lr=.005)

    # epochs = 5
    # for t in range(epochs):
    #     print(f"Epoch {t+1}\n-------------------------------")
    #     train( model, train_dataloader, loss_fn, optimizer)
    #     test( model, test_dataloader, loss_fn)
    # print("Done!")


if __name__ == '__main__':
    main()
