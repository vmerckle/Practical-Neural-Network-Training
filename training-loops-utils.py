import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from IPython import display

dtype=torch.float32


def rescale(u,a,b,a1,b1): return a1 + (u-a)*(b1-a1)/(b-a)
def crescents_dataset(m, closeness):
    y = np.sign(np.random.randn(1,m))
    x = np.zeros((2,m))
    r = 1 + .1*np.random.rand(m) # radius
    t = np.pi/2 + np.pi*np.random.rand(m) # angle
    x[0,:] = r * np.sin(t)
    x[1,:] = r * np.cos(t)
    I = (y.flatten()<0) 
    x[0,I] = x[0,I] + 1
    x[1,I] = -closeness-x[1,I]
    x[0,:] = rescale(x[0,:], -1,2,  .2,.8)
    x[1,:] = rescale(x[1,:], -1,.6, .2,.8) 
    return torch.tensor(x, dtype=dtype), torch.tensor(y, dtype=dtype)


def showoutput(model, X, Y, once=False, it=0, loss=0, discretize=False):
    q = 100;
    t = np.linspace(0,1,q)
    [U,V] = np.meshgrid(t,t)
    z = np.vstack([V.flatten(), U.flatten()])
    with torch.no_grad():
        V = model(torch.tensor(z.T, dtype=dtype)).detach().numpy()
    V = V.reshape((q, q))
    fig = plt.figure()
    plt.plot( X[0,Y.flatten()>0], X[1,Y.flatten()>0], '.', c="skyblue")
    plt.plot( X[0,Y.flatten()<0], X[1,Y.flatten()<0], '.', c="indianred" )
    if not once:
        plt.title(f"Loss at iteration {it} = {loss}")
    if discretize:
        U = (V <= 0)*(-1) + (V > 0)*(1)
    else:
        U = V.clip(-2, 2)*(-1)
    plt.imshow(U.T, origin="lower", extent=[0,1,0,1], cmap="bwr")
    if once:
        return
    plt.draw()
    display.clear_output(wait=True)
    display.display(fig)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm_notebook as tqdm
##define train function

def train(model, loader, optimizer, loss_fn, load=1, device="cpu", log_interval=10000):
    model.train()
    train_loader = loader
    tk0 = tqdm(train_loader, total=int(len(train_loader)))
    counter = 0
    batch_size = loader.batch_size
    i = 0
    n = len(loader.dataset)

    for batch_idx, (data, target) in enumerate(tk0):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        counter += 1
        tk0.set_postfix(loss=(loss.item()*data.size(0) / (counter * train_loader.batch_size)))
        i += batch_size
        if i/n >= load:
            break


            
##define test function
def test(model, loss_fn, loader,device="cpu", load=1, noprint=False):
    batch_size = loader.batch_size
    model.eval()
    test_loss = 0
    correct = 0
    i = 0
    n = len(loader.dataset)
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            i += batch_size
            if i/n >= load:
                break

    test_loss /= i
    if not noprint:
        print(f'Average loss: {test_loss:.4f}, Correct/Tested: {correct}/{i} ({100. * correct / i:.0f}%)')
    return test_loss


