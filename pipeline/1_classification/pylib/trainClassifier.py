import pandas
import pandas as pd
import torch
import time
import copy
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def trainClassifier(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs, class_names):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        print("\n","\n",'Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        test_preds = []
        test_labels = []

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if phase == 'test':
                    for val in preds.cpu():
                        test_preds.append(int(val))
                    for val in labels.data.cpu():
                        test_labels.append(int(val))
                    #test_preds.append(preds.cpu())
                    #test_labels.append(labels.data.cpu())
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'test':
                preds_labels = pd.DataFrame({'Prediction': test_preds, "Label": test_labels})

                # print and save classification report
                print(classification_report(test_labels, test_preds, target_names=class_names,zero_division=0))

                # create and label confusion matrix
                cm = pandas.DataFrame(confusion_matrix(test_labels, test_preds, labels=None, sample_weight=None, normalize=None))
                cm_colnames = []
                cm_rownames = []
                for c in class_names:
                    cm_colnames.append("Predicted " + c)
                    cm_rownames.append("Actual " + c)
                cm.columns = cm_colnames
                cm.index = cm_rownames

                #print("Bad % perfect: " + cm[0,0] / (cm[1,0] + cm[2,0] + cm[3,0]))
                #print("Dorsal % perfect: " + cm[1,1])
                #print("Dorsal % good: ")
                #print("DL % good: ")
                #print("DL % perfect: ")
                print(print(cm.to_string()))

                #preds_labels.to_csv("pylib/models/preds.csv")
            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model