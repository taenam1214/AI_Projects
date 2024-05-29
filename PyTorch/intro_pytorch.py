import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



def get_data_loader(training = True):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    data_set = datasets.FashionMNIST('./data', train=training, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(data_set, batch_size=64, shuffle=training)
    return loader

def build_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),  # 28*28 pixels in, 128 out
        nn.ReLU(),
        nn.Linear(128, 64),  # 128 in, 64 out
        nn.ReLU(),
        nn.Linear(64, 10)  # 64 in, 10 out (for 10 classes)
    )
    return model




def train_model(model, train_loader, criterion, T):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    for epoch in range(T):
        total_loss = 0
        correct = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        avg_loss = total_loss / len(train_loader.dataset)
        accuracy = 100. * correct / len(train_loader.dataset)
        print(f'Train Epoch: {epoch} Accuracy: {correct}/{len(train_loader.dataset)}({accuracy:.2f}%) Loss: {avg_loss:.3f}')

    


def evaluate_model(model, test_loader, criterion, show_loss = True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    if show_loss:
        print(f'Average loss: {test_loss:.4f}')
    print(f'Accuracy: {accuracy:.2f}%')
    


def predict_label(model, test_images, index):
    model.eval()
    with torch.no_grad():
        logits = model(test_images[index].unsqueeze(0))
        probabilities = F.softmax(logits, dim=1)
        top_probs, top_idxs = torch.topk(probabilities, 3)
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
        for i in range(top_probs.size(1)):
            print(f'{class_names[top_idxs[0][i]]}: {top_probs[0][i].item()*100:.2f}%')



if __name__ == '__main__':
    train_loader = get_data_loader()
    test_loader = get_data_loader(False)
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    # Train for 5 epochs
    train_model(model, train_loader, criterion, 5)
    # Evaluate on test set
    evaluate_model(model, test_loader, criterion, show_loss=True)
    # Predict the label of the first image in the test set
    test_images = next(iter(test_loader))[0]
    predict_label(model, test_images, 0)
