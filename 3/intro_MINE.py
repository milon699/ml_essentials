
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.nn.functional import conv2d, max_pool2d, cross_entropy

import torch.multiprocessing as mp

#Necessary for me to not end up in Runtime Error, if someone can give me feedback on why that is, much appreciated!
if __name__ == '__main__':
    #mp.set_start_method('spawn')
    
    plt.rc("figure", dpi=100)
    
    batch_size = 100
    
    # transform images into normalized tensors
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    
    train_dataset = datasets.MNIST(
        "./",
        download=True,
        train=True,
        transform=transform,
    )
    
    test_dataset = datasets.MNIST(
        "./",
        download=True,
        train=False,
        transform=transform,
    )
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )
    
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    
    def init_weights(shape):
        # Kaiming He initialization (a good initialization is important)
        # https://arxiv.org/abs/1502.01852
        std = np.sqrt(2. / shape[0])
        w = torch.randn(size=shape) * std
        w.requires_grad = True
        return w
    
    def init_PRelu_weights(shape):
        
        #Initialize with a as being "normal" ReLU
        a = torch.zeros(shape)
        
        #Make sure gradients are computed 
        a.requires_grad = True
        return a 
    
    
    def rectify(x):
        # Rectified Linear Unit (ReLU)
        return torch.max(torch.zeros_like(x), x)
    
    def PRelu(X, a):
        # Parametric ReLU Implementation
        return torch.where(X > 0, X, a*X)
    
    
    class RMSprop(optim.Optimizer):
        """
        This is a reduced version of the PyTorch internal RMSprop optimizer
        It serves here as an example
        """
        def __init__(self, params, lr=1e-3, alpha=0.5, eps=1e-8):
            defaults = dict(lr=lr, alpha=alpha, eps=eps)
            super(RMSprop, self).__init__(params, defaults)
    
        def step(self):
            for group in self.param_groups:
                for p in group['params']:
                    grad = p.grad.data
                    state = self.state[p]
    
                    # state initialization
                    if len(state) == 0:
                        state['square_avg'] = torch.zeros_like(p.data)
    
                    square_avg = state['square_avg']
                    alpha = group['alpha']
    
                    # update running averages
                    square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                    avg = square_avg.sqrt().add_(group['eps'])
    
                    # gradient update
                    p.data.addcdiv_(grad, avg, value=-group['lr'])
    
    
    # define the neural network
    def model(x, w_h, w_h2, w_o):
        h = rectify(x @ w_h)
        h2 = rectify(h @ w_h2)
        pre_softmax = h2 @ w_o
        return pre_softmax
    
    #Define dropout method
    def dropout(X, p_drop = 0.5):
        if (0 < p_drop < 1):
            return torch.where(torch.from_numpy(np.random.binomial(1, 0.5, size = X.shape)) == 1, 0, X/(1-p_drop))
        else:
            return X
    
    # define the neural network with dropout
    def dropout_model(x, w_h, w_h2, w_o, p_drop_out, p_drop_hidden):
        h = rectify(dropout(x, p_drop_out) @ w_h)
        h2 = rectify(dropout(h, p_drop_hidden) @ w_h2)
        pre_softmax = dropout(h2, p_drop_hidden) @ w_o
        return pre_softmax
    
    # define the neural network with dropout AND PReLU
    def dropout_model_PRelu(x, w_h, w_h2, w_o, a, p_drop_out, p_drop_hidden):
        h = PRelu(dropout(x, p_drop_out) @ w_h, a)
        h2 = PRelu(dropout(h, p_drop_hidden) @ w_h2, a)
        pre_softmax = dropout(h2, p_drop_hidden) @ w_o
        return pre_softmax
    
    # define the neural network with PReLU instead of normal ReLU
    def model_PRelu(x, w_h, w_h2, w_o, a):
        h = PRelu(x @ w_h, a)
        h2 = PRelu(h @ w_h2, a)
        pre_softmax = h2 @ w_o
        return pre_softmax
    
    #Loop over different models
    models = [model, dropout_model, model_PRelu, dropout_model_PRelu]
    models = [dropout_model]
    name_models = ['ReLU', 'ReLU with dropout', 'PReLU', 'PReLU with dropout']
    train_loss_models = []
    test_loss_models = []
    
    i = 0
    for model_chosen in models:
        
        # initialize weights and a of PRelu
        
        # input shape is (B, 784)
        w_h = init_weights((784, 625))
        # hidden layer with 625 neurons
        w_h2 = init_weights((625, 625))
        # hidden layer with 625 neurons
        w_o = init_weights((625, 10))
        # output shape is (B, 10)
        
        #a vector initialization
        a = init_PRelu_weights(625)
        
        
        if model_chosen == model or model_chosen == dropout_model:
            optimizer = RMSprop(params=[w_h, w_h2, w_o])
        else:
            optimizer = RMSprop(params=[w_h, w_h2, w_o, a])
        
        n_epochs = 20
        
        train_loss = []
        test_loss = []
        
        # put this into a training loop over 100 epochs
        for epoch in tqdm(range(n_epochs + 1)):
            train_loss_this_epoch = []
            for idx, batch in enumerate(train_dataloader):
                x, y = batch
        
                # our model requires flattened input
                x = x.reshape(batch_size, 784)
                # feed input through model
                if model_chosen == model:
                    noise_py_x = model_chosen(x, w_h, w_h2, w_o)
                elif model_chosen == dropout_model:
                    noise_py_x = model_chosen(x, w_h, w_h2, w_o, p_drop_out = 0.5, p_drop_hidden = 0.5)
                elif model_chosen == model_PRelu:
                    noise_py_x = model_chosen(x, w_h, w_h2, w_o, a)
                else: 
                    noise_py_x = model_chosen(x, w_h, w_h2, w_o, a, p_drop_out = 0.5, p_drop_hidden = 0.5)
            
        
                # reset the gradient
                optimizer.zero_grad()
        
                # the cross-entropy loss function already contains the softmax
                loss = cross_entropy(noise_py_x, y, reduction="mean")
        
                train_loss_this_epoch.append(float(loss))
        
                # compute the gradient
                loss.backward()
                # update weights
                optimizer.step()
        
            train_loss.append(np.mean(train_loss_this_epoch))
        
            # test periodically
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}")
                print(f"Mean Train Loss: {train_loss[-1]:.2e}")
                test_loss_this_epoch = []
        
                # no need to compute gradients for validation
                with torch.no_grad():
                    for idx, batch in enumerate(test_dataloader):
                        x, y = batch
                        x = x.reshape(batch_size, 784)
                        
                        if model_chosen == model:
                            noise_py_x = model_chosen(x, w_h, w_h2, w_o)
                        elif model_chosen == dropout_model:
                            noise_py_x = model_chosen(x, w_h, w_h2, w_o, p_drop_out = 0.5, p_drop_hidden = 0.5)
                        elif model_chosen == model_PRelu:
                            noise_py_x = model_chosen(x, w_h, w_h2, w_o, a)
                        else: 
                            noise_py_x = model_chosen(x, w_h, w_h2, w_o, a, p_drop_out = 0.5, p_drop_hidden = 0.5)
                            
                        loss = cross_entropy(noise_py_x, y, reduction="mean")
                        test_loss_this_epoch.append(float(loss))
        
                test_loss.append(np.mean(test_loss_this_epoch))
                
                print(f"Mean Test Loss:  {test_loss[-1]:.2e}")
        
        
        train_loss_models.append(train_loss)
        test_loss_models.append(test_loss)
        
        plt.figure()
        plt.plot(np.arange(n_epochs + 1), train_loss, label="Train")
        plt.plot(np.arange(1, n_epochs + 2, 10), test_loss, label="Test")
        plt.title("Train and Test Loss over Training: " + name_models[i])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()
        plt.legend()
        i += 1
        
        
    def model_CNN(x, weightvector_1, weightvector_2, weightvector_3, w_h2, w_o, p_drop_input):
        
        #Replace h with 3 convolutional layers
        convolutional_layer_1 = rectify(conv2d(x, weightvector_1))
        #Reduce (2,2) window to 1 pixel
        subsampling_layer_1 = max_pool2d(convolutional_layer_1, (2,2))
        out_layer_1 = dropout(subsampling_layer_1, p_drop_input)
        
        convolutional_layer_2 = rectify(conv2d(out_layer_1, weightvector_2))
        #Reduce (2,2) window to 1 pixel
        subsampling_layer_2 = max_pool2d(convolutional_layer_2, (2,2))
        out_layer_2 = dropout(subsampling_layer_2, p_drop_input)
        
        convolutional_layer_3 = rectify(conv2d(out_layer_2, weightvector_3))
        #Reduce (2,2) window to 1 pixel
        subsampling_layer_3 = max_pool2d(convolutional_layer_3, (2,2))
        out_layer_3 = dropout(subsampling_layer_3, p_drop_input)
        
        #Reshape last convolutional layer output flattened
        out_layer_3 = out_layer_3.reshape((batch_size, -1))
        
        #Implement h2 layer
        h2 = rectify(out_layer_3 @ w_h2)
        
        pre_softmax = h2 @ w_o
        
        return pre_softmax
    
    # initialize weights and filter weight vectors
    
    
    # input shape is (B, 784)
    w_h = init_weights((784, 625))
    # hidden layer with 625 neurons
    number_of_output_pixels = 128 #number of channels after three convolutional layers
    w_h2 = init_weights((number_of_output_pixels, 625))
    # hidden layer with 625 neurons
    w_o = init_weights((625, 10))
    # output shape is (B, 10)
    
    weightvector_1 = init_weights((32, 1, 5, 5))
    weightvector_2 = init_weights((64, 32, 5, 5))
    weightvector_3 = init_weights((128, 64, 2, 2))
    
    
    optimizer = RMSprop(params=[weightvector_1, weightvector_2, weightvector_3, w_h2, w_o])
    
    n_epochs = 20
    
    train_loss = []
    test_loss = []
    
    # put this into a training loop over 100 epochs
    for epoch in tqdm(range(n_epochs + 1)):
        train_loss_this_epoch = []
        for idx, batch in enumerate(train_dataloader):
            x, y = batch
    
            # our model requires 2-dim input
            x = x.reshape(-1, 1, 28, 28)
            
            # feed input through model
            noise_py_x = model_CNN(x, weightvector_1, weightvector_2, weightvector_3, w_h2, w_o, p_drop_input = 0.5)
    
            # reset the gradient
            optimizer.zero_grad()
    
            # the cross-entropy loss function already contains the softmax
            loss = cross_entropy(noise_py_x, y, reduction="mean")
    
            train_loss_this_epoch.append(float(loss))
    
            # compute the gradient
            loss.backward()
            # update weights
            optimizer.step()
    
        train_loss.append(np.mean(train_loss_this_epoch))
    
        # test periodically
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}")
            print(f"Mean Train Loss: {train_loss[-1]:.2e}")
            test_loss_this_epoch = []
    
            # no need to compute gradients for validation
            with torch.no_grad():
                for idx, batch in enumerate(test_dataloader):
                    x, y = batch
                    
                    # our model requires 2-dim input
                    x = x.reshape(-1, 1, 28, 28)
                    
                    # feed input through model
                    noise_py_x = model_CNN(x, weightvector_1, weightvector_2, weightvector_3, w_h2, w_o, p_drop_input = 0.5)
            
                    loss = cross_entropy(noise_py_x, y, reduction="mean")
                    test_loss_this_epoch.append(float(loss))
    
            test_loss.append(np.mean(test_loss_this_epoch))
            
            print(f"Mean Test Loss:  {test_loss[-1]:.2e}")
    

    plt.figure()
    plt.plot(np.arange(n_epochs + 1), train_loss, label="Train")
    plt.plot(np.arange(1, n_epochs + 2, 10), test_loss, label="Test")
    plt.title("LeNet Train and Test Loss over Training: ")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()


    def model_CNN_var(x, weightvector_1, weightvector_2, w_h2, w_o, p_drop_input):
        
        #Replace h with 3 convolutional layers
        convolutional_layer_1 = rectify(conv2d(x, weightvector_1))
        #Reduce (2,2) window to 1 pixel
        subsampling_layer_1 = max_pool2d(convolutional_layer_1, (2,2))
        out_layer_1 = dropout(subsampling_layer_1, p_drop_input)
        
        convolutional_layer_2 = rectify(conv2d(out_layer_1, weightvector_2))
        #Reduce (2,2) window to 1 pixel
        subsampling_layer_2 = max_pool2d(convolutional_layer_2, (2,2))
        out_layer_2 = dropout(subsampling_layer_2, p_drop_input)
        
        #Reshape last convolutional layer output flattened
        out_layer_2 = out_layer_2.reshape((batch_size, -1))
        
        #Implement h2 layer
        h2 = rectify(out_layer_2 @ w_h2)
        
        pre_softmax = h2 @ w_o
        
        return pre_softmax
    
    # initialize weights and filter weight vectors
    
    
    # input shape is (B, 784)
    w_h = init_weights((784, 625))
    # hidden layer with 625 neurons
    number_of_output_pixels = 128 #number of channels after three convolutional layers
    w_h2 = init_weights((number_of_output_pixels, 625))
    # hidden layer with 625 neurons
    w_o = init_weights((625, 10))
    # output shape is (B, 10)
    
    weightvector_1 = init_weights((64, 1, 5, 5))
    weightvector_2 = init_weights((128, 64, 10, 10))
    
    
    optimizer = RMSprop(params=[weightvector_1, weightvector_2, w_h2, w_o])
    
    n_epochs = 20
    
    train_loss = []
    test_loss = []
    
    # put this into a training loop over 100 epochs
    for epoch in tqdm(range(n_epochs + 1)):
        train_loss_this_epoch = []
        for idx, batch in enumerate(train_dataloader):
            x, y = batch
    
            # our model requires 2-dim input
            x = x.reshape(-1, 1, 28, 28)
            
            # feed input through model
            noise_py_x = model_CNN_var(x, weightvector_1, weightvector_2, w_h2, w_o, p_drop_input = 0.5)
    
            # reset the gradient
            optimizer.zero_grad()
    
            # the cross-entropy loss function already contains the softmax
            loss = cross_entropy(noise_py_x, y, reduction="mean")
    
            train_loss_this_epoch.append(float(loss))
    
            # compute the gradient
            loss.backward()
            # update weights
            optimizer.step()
    
        train_loss.append(np.mean(train_loss_this_epoch))
    
        # test periodically
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}")
            print(f"Mean Train Loss: {train_loss[-1]:.2e}")
            test_loss_this_epoch = []
    
            # no need to compute gradients for validation
            with torch.no_grad():
                for idx, batch in enumerate(test_dataloader):
                    x, y = batch
                    
                    # our model requires 2-dim input
                    x = x.reshape(-1, 1, 28, 28)
                    
                    # feed input through model
                    noise_py_x = model_CNN_var(x, weightvector_1, weightvector_2, w_h2, w_o, p_drop_input = 0.5)
            
                    loss = cross_entropy(noise_py_x, y, reduction="mean")
                    test_loss_this_epoch.append(float(loss))
    
            test_loss.append(np.mean(test_loss_this_epoch))
            
            print(f"Mean Test Loss:  {test_loss[-1]:.2e}")
    

    plt.figure()
    plt.plot(np.arange(n_epochs + 1), train_loss, label="Train")
    plt.plot(np.arange(1, n_epochs + 2, 10), test_loss, label="Test")
    plt.title("LeNet Train and Test Loss over Training: ")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    
    
    
