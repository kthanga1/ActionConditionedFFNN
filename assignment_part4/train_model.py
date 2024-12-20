from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF
from torcheval.metrics import BinaryF1Score
from torcheval.metrics import BinaryPrecision

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle

trainingloss = []
testingloss = []
trainingf1 = []
testingf1 = []
testingprecision = []

f1metric = BinaryF1Score()
precisionmetric = BinaryPrecision()


def train(model, train_dataloader, lossfn, optimizer ):
    size = len(train_dataloader.dataset)
    model.train()
    trainrunloss= 0
    correct = 0
    f1score = 0
    precisionscore = 0
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
        f1metric.update(input=pred.view(pred.size()[0]), target=y.view(y.size()[0]))
        f1score += f1metric.compute()
        # precisionmetric.update(input=pred.view(pred.size()[0]), target=y.view(y.size()[0]))
        # precisionscore += precisionmetric.compute()
        
    correct /= size
    print(f"Training: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {trainrunloss/size:>7f} \n")  
    print(f"Training: \n F1 Score: {f1score/num_batches}")   
    print(f"Training: \n Precision Score: {precisionscore/num_batches}")   
    trainingloss.append(trainrunloss/size)
    trainingf1.append(f1score/num_batches)
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
            # print(pred)
            # print(y)
            test_loss += loss_fn(pred, y).item()
            pred = (pred > 0.25).type(torch.float)
            correct += (pred == y).type(torch.float).sum().item()
            f1metric.update(input=pred.view(pred.size()[0]), target=y.view(y.size()[0]))
            f1score += f1metric.compute()
            # precisionmetric.update(input=pred.view(pred.size()[0]), target=y.view(y.size()[0]))
            # precisionscore += precisionmetric.compute()
    test_loss /= num_batches
    correct /= size
    testingloss.append(test_loss)
    testingf1.append(f1score/num_batches)
    testingprecision.append(precisionscore/num_batches)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Test: \n F1 Score: {f1score/num_batches} ")    
    return test_loss



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

    for t in range(no_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss= train( model, train_dataloader, loss_fn, optimizer)
        # test_loss = model.evaluate( model, test_dataloader, loss_fn)
        test_loss = test(model, test_dataloader, loss_fn)

    print("Done Training!")

    
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(testingloss,label="testing")
    plt.plot(trainingloss,label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig('traintestloss.png')

    plt.figure(figsize=(10,5))
    plt.title("Training and Validation F1 Score")
    plt.plot(testingf1,label="testing")
    plt.plot(trainingf1,label="train")
    plt.xlabel("iterations")
    plt.ylabel("F1Score")
    plt.legend()
    plt.show()
    plt.savefig('F1score.png')


    torch.save(model.state_dict(), "saved/saved_model.pkl")
    print("Saved PyTorch Model State to saved/saved_model.pkl")

    model = Action_Conditioned_FF()
    model.load_state_dict(torch.load("saved/saved_model.pkl", weights_only=True))
    print("Evaluating final model")
    test_loss = model.evaluate(model, test_dataloader,loss_fn)
    print("Evaluation test loss for final model is --> ", test_loss)



if __name__ == '__main__':
    no_epochs = 150
    train_model(no_epochs)
