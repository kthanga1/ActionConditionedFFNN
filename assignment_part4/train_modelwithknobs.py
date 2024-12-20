from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF
from torcheval.metrics import BinaryF1Score
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle

trainingloss = []
testingloss = []
trainingf1 = []
testingf1 = []

metric = BinaryF1Score()

def train(model, train_dataloader, lossfn, optimizer ):
    size = len(train_dataloader.dataset)
    model.train()
    trainrunloss= 0
    correct = 0
    f1score = 0
    num_batches = len(train_dataloader)
    for batch, sample in enumerate(train_dataloader):
        X = sample['input']
        y = sample['label']
        pred = model(X) 
        loss = lossfn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        trainrunloss += loss.item() * X.size(0)
        pred = (pred > 0.25).type(torch.float)
        correct += (pred == y).type(torch.float).sum().item()
        # if batch % 100 == 0:
        #     loss, current = loss.item(), (batch + 1) * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        #     trainrunloss = loss
        metric.update(input=pred.view(pred.size()[0]), target=y.view(y.size()[0]))
        f1score += metric.compute().item()
    correct /= size
    print(f"Training: Accuracy: {(100*correct):>0.1f}%, Avg loss: {trainrunloss/size:>7f}")  
    print(f"Training: F1 Score: {f1score/num_batches} \n")    
    trainingloss.append(trainrunloss/size)
    trainingf1.append(f1score/num_batches)
    return trainrunloss/size


def test(model, dataloader, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    f1score = 0
    with torch.no_grad():
        for  batch, sample in enumerate(dataloader):
            X = sample['input']
            y = sample['label']
            pred = model(X)
            # print(pred)
            # print(y)
            test_loss += loss_fn(pred, y).item()
            pred = (pred > 0.25).type(torch.float)
            correct += (pred == y).type(torch.float).sum().item()
            metric.update(input=pred.view(pred.size()[0]), target=y.view(y.size()[0]))
            f1score += metric.compute().item()
    test_loss /= num_batches
    correct /= size
    testingloss.append(test_loss)
    testingf1.append(f1score/num_batches)
    print(f"Test: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    print(f"Test: F1 Score: {f1score/num_batches} \n")    
    return test_loss, f1score/num_batches



def train_model(no_epochs):

    batch_size = 8
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()

    dataloaders = Data_Loaders(batch_size=batch_size)
    model = Action_Conditioned_FF()
    # print(model)

    train_dataloader = dataloaders.train_loader
    test_dataloader = dataloaders.test_loader

    loss_fn = nn.BCELoss()
    # print(loss_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    
    savedmodels = []

    kresults = {}
    k = 5
    
    kfold = KFold(n_splits=k,shuffle=False)
    kfolddataset = dataloaders.nav_dataset
    for kfold, (trainids, valids) in enumerate(kfold.split(kfolddataset)):
        model = Action_Conditioned_FF()
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        global trainingloss, testingloss, trainingf1, testingf1
        
        print(f"kfold training set {kfold+1}\n-------------------------------")
        kfoldtrain = Subset(kfolddataset, trainids)
        kfoldtest = Subset(kfolddataset, valids)
        train_dataloader = torch.utils.data.DataLoader(kfoldtrain, batch_size=batch_size)
        test_dataloader = torch.utils.data.DataLoader(kfoldtest, batch_size=batch_size)
        for t in range(no_epochs):
            train_loss= train( model, train_dataloader, loss_fn, optimizer)
            # test_loss = model.evaluate( model, test_dataloader, loss_fn)
            test_loss,_ = test(model, test_dataloader, loss_fn)
            print(abs(train_loss-test_loss))
        f = "saved/saved_model"+str(kfold)+".pkl"
        torch.save(model.state_dict(), f)
        print("Saved PyTorch Model State with kfold set " + str(kfold)+" to"+f)
        kresults[kfold] = [trainingloss, testingloss, trainingf1, testingf1]
        trainingloss = []
        testingloss = []
        trainingf1 = []
        testingf1 = []
        
    for idx, vals in kresults.items():
        plt.figure(figsize=(10,5))
        plt.title("Training and Validation Loss")
        plt.plot(vals[1],label="testing")
        plt.plot(vals[0],label="train")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        plt.savefig('traintestloss'+str(idx)+'.png')

        plt.figure(figsize=(10,5))
        plt.title("Training and Validation F1 Score")
        plt.plot(vals[3],label="testing")
        plt.plot(vals[2],label="train")
        plt.xlabel("iterations")
        plt.ylabel("F1Score")
        plt.legend()
        plt.show()
        plt.savefig('F1score'+str(idx)+'.png')


    

    for i in range(k):
        model = Action_Conditioned_FF()
        model.load_state_dict(torch.load("saved/saved_model"+str(i)+".pkl", weights_only=True))
        print("Evaluating final model")
        test(model, test_dataloader,loss_fn)
        # print("Evaluation test loss for final model is --> ", test_loss , "f1 score ", f1score)

    
    # losses = []
    # min_loss = model.evaluate(model, data_loaders.test_loader, loss_fn)
    # losses.append(min_loss)


    # for epoch_i in range(no_epochs):
    #     model.train()
    #     for idx, sample in enumerate(data_loaders.train_loader): # sample['input'] and sample['label']
    #         pass



if __name__ == '__main__':
    no_epochs = 150
    train_model(no_epochs)
