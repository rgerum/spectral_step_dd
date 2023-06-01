import torch
import numpy as np
import os
import time
from pathlib import Path

def get_train(net, trainloader, testloader, device, optimizer, criterion, output_folder):
    best_acc = 0  # best test accuracy
    output_folder = Path(output_folder)

    def func_train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    def func_test(epoch):
        nonlocal best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100. * correct / total
        print(acc)
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, output_folder / f'ckpt_best.pth')
            best_acc = acc
        # save 1, 3, 10, 300, ...
        if np.log10(epoch + 1) % 1 == 0 or np.log10(
                (epoch + 1) / 3) % 1 == 0:  # in [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]:
            print('Saving.. epoch', epoch)
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, output_folder / f'ckpt_{epoch}.pth')
        return [acc, time.time(), epoch]

    return func_train, func_test
