import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

def get_device():
		if torch.cuda.is_available():
			device = 'cuda:0'
		else:
			device = 'cpu'
		return device

device = get_device()

def test_image_reconstruct(model, test_loader):
     for batch in test_loader:
        img, _ = batch
        img = img.to(device)
        #img = img.view(img.size(0), -1)
        outputs = model(img)
        outputs = outputs.view(256, 256, 3).cpu().data
        save_image(outputs, 'reconstruction.png')
        break

def save_decod_img(img, epoch):
    img = img[0]
    save_image(img, 'Autoencoder_image{}.png'.format(epoch))
    
def make_dir():
    image_dir = 'MNIST_Out_Images'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
def training(model, train_loader, Epochs, optimizer):
    train_loss = []
    for epoch in range(Epochs):
        running_loss = 0.0
        for data in train_loader:
            img, _ = data
            img = img.to(device)
            optimizer.zero_grad()
            
            x_hat = model.forward(img).to(device)
            loss = F.mse_loss(img, x_hat, reduction="none")
            loss = loss.sum(dim=[1,2,3]).mean(dim=[0])

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss = running_loss / len(train_loader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch+1, Epochs, loss))

        if epoch % 5 == 0:
            save_decod_img(x_hat.cpu().data, epoch)

    return train_loss
