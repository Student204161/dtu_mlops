import torch


def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    train = torch.randn(50000, 784)
    test = torch.randn(10000, 784)
    train_images = torch.empty((0,784),dtype=torch.float32) # 784 is num of features.
    train_targets = torch.empty((0),dtype=torch.long)
    for i in range(5):
        train_images = torch.cat((train_images , torch.load(f'data/corruptmnist/train_images_{i}.pt').flatten(1,2)),0)
        train_targets = torch.cat((train_targets , torch.load(f'data/corruptmnist/train_target_{i}.pt')),0)

    test_images = torch.load(f'data/corruptmnist/test_images.pt').flatten(1,2)
    test_targets = torch.load(f'data/corruptmnist/test_target.pt')
    test_images = test_images.type(torch.float32)    
    test_targets = test_targets.type(torch.long)

    trainset = torch.utils.data.TensorDataset(train_images,train_targets)
    testset = torch.utils.data.TensorDataset(test_images,test_targets)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    return train_loader, test_loader





mnist()