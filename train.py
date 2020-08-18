import torch.optim as optim
import torch.nn as nn
import torch
from networks import SpymasterNetwork, GuessingNetwork
from GameboardDataset import GameboardDataset
import main

spymaster = SpymasterNetwork()
guesser = GuessingNetwork()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD([spymaster.parameters(), guesser.parameters()], lr=0.001, momentum=0.9)

trainset = GameboardDataset(main.board)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=False, num_workers=2)

for epoch in range(10):
    running_loss = 0
    for i, board in enumerate(dataloader):
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        clues = spymaster(board)
        guess = guesser(clues)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch, i, running_loss / 2000))
            running_loss = 0

print('Finished Training')