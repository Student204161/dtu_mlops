import click
import torch
from model import MyAwesomeModel

from data import mnist


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--epochs", default=5, help="epochs to train for")
def train(lr,epochs):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)
    print(epochs)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()

    for epoch in range(epochs):
        running_loss = 0
        for images, labels in train_set:
            # add dim for conv2d
            images.resize_(images.shape[0], 1, 28,28)
            output = model.forward(images)
            
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Training loss: {running_loss/len(train_set)}")
        print(f'epoch: ', epoch+1)
    
    torch.save(model.state_dict(), 'checkpoint.pth')
    print('done training and saved model')
 



@click.command()
@click.argument("model_checkpoint", default='checkpoint.pth')
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    criterion = torch.nn.CrossEntropyLoss()
    model_weights = torch.load(model_checkpoint)
    model.load_state_dict(model_weights)
    _, test_set = mnist()
    running_loss = 0
    correct = 0
    false = 0
    for images, labels in test_set:
        # add dims/reshape for conv2d
        images.resize_(images.shape[0], 1, 28,28)
        output = model.forward(images)
        
        loss = criterion(output, labels)
        running_loss += loss.item()

        for out_index in range(len(labels)):
            
            if torch.argmax(output[out_index]) == labels[out_index]:
                correct +=1
            else:
                false += 1
            
    
    print(f'test_loss: {running_loss / len(test_set)}')
    print(f'accuracy: {correct / (false + correct)}')
    



cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
