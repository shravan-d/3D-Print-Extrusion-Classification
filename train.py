import torch
import torch.nn as nn
from data import *
from model import *
from util import *
import datetime
import sklearn
import pandas as pd


batch_size = 8
input_size = (256, 128)
aspect_ratio = [2, 1]
testing = True
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU")
else:
    device = torch.device('cpu')
    print('CPU')

full_df = pd.read_csv('/kaggle/input/new-csv/semi-supervised-train.csv')
val_df = full_df.loc[
    full_df['printer_id'] == 22 | (full_df['printer_id'] == 101 & full_df['print_id'].isin([1678580155, 1678593348]))]
train_df = full_df.loc[~(full_df['printer_id'] == 22 | (
            full_df['printer_id'] == 101 & full_df['print_id'].isin([1678580155, 1678593348])))]
if testing:
    num_domains = len(full_df['printer_id'].unique().tolist())
else:
    num_domains = len(train_df['printer_id'].unique().tolist())

class_weights = sklearn.utils.class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(full_df['has_under_extrusion'].tolist()),
    y=full_df['has_under_extrusion'].tolist()
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

model = ResNet10().to(device)
loss_function = nn.CrossEntropyLoss(weight=class_weights)
domain_loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
domain_lambda = 0.6
print_lambda = 4
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
test_df = pd.read_csv('/kaggle/input/early-detection-of-3d-printing-issues/test.csv')
n_epochs = 5
print_every = 3000
train_loss = []
for epoch in range(n_epochs):
    total_step = len(train_df) / batch_size
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    #     full_df = print_job_shuffle(full_df)
    #     full_df = batch_shuffle(full_df)
    generator = generate_batches(train_df)
    running_loss, correct, domain_correct, total, current_step, current_idx = 0.0, 0, 0, 0, 0, 0
    total_step_loss, step_loss, domain_step_loss, print_step_loss = 0.0, 0.0, 0.0, 0.0
    print(f'\nEpoch {epoch + 1}')
    model.train()
    for images, labels, domains, print_jobs in generator:
        p = float(current_idx + epoch * total_step) / n_epochs / total_step
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        current_idx += 1
        optimizer.zero_grad()
        images = torch.stack(images, dim=0, out=None).to(device)
        labels = torch.from_numpy(labels).to(device)
        domains = torch.from_numpy(domains).to(device)
        print_jobs = torch.from_numpy(print_jobs).to(device)
        outputs, domain_outputs = model(images, alpha)

        classification_loss = loss_function(outputs, labels)
        domain_loss = domain_loss_function(domain_outputs, domains)
        print_loss = print_loss_function(outputs, labels, print_jobs)
        loss = classification_loss + domain_lambda * domain_loss + print_lambda * print_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        step_loss += classification_loss.item()
        domain_step_loss += domain_loss.item()
        total_step_loss += loss.item()
        print_step_loss += print_loss.item()

        _, pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred == labels).item()
        _, domain_pred = torch.max(domain_outputs, dim=1)
        domain_correct += torch.sum(domain_pred == domains).item()
        total += labels.size(0)

        current_step += 1
        if current_step % print_every == 0:
            print(
                'Epoch [{}/{}], Step [{}/{}], Classification Loss: {:.4f}, Domain Loss: {:.4f}, Print Loss: {:.4f}, Total Loss: {:.4f}'
                .format(epoch + 1, n_epochs, current_step, total_step, step_loss / print_every,
                        domain_step_loss / print_every, print_step_loss / print_every, total_step_loss / print_every))
            step_loss = 0.0
            domain_step_loss = 0.0
            print_step_loss = 0.0
            total_step_loss = 0.0

    train_loss.append(running_loss / total_step)
    print(
        f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct / total):.4f}, train-domain-acc: {(100 * domain_correct / total):.4f}')
    scheduler.step()
    if epoch > 2:
        run_testing(epoch)
        torch.save(model.state_dict(), 'TempModel' + str(epoch + 1) + '.pt')


def predict_test(images, model):
    images = torch.stack(images, dim=0, out=None).to(device)
    output, _ = model(images, alpha=None)
    _, pred = torch.max(output, dim=1)
    x2 = torch.nn.functional.softmax(output).detach()
    return np.asarray(pred.cpu()), np.asarray(x2.cpu())


def run_testing(epoch=-1):
    test_generator = generate_batches(test_df, train=False)
    results = pd.DataFrame(columns=['img_path', 'has_under_extrusion'])
    start, idx = datetime.datetime.now(), 0
    for images, paths in test_generator:
        progress_bar(start, idx, len(test_df) / batch_size)
        idx += 1
        predictions, _ = predict_test(images, model)
        for i in range(min(len(predictions), batch_size)):
            temp_data = {'img_path': paths[i], 'has_under_extrusion': predictions[i]}
            results = pd.concat([results, pd.DataFrame.from_records([temp_data])])
    if epoch == -1:
        filename = 'submission.csv'
    else:
        filename = 'submission' + str(epoch + 1) + '.csv'
    results.to_csv(filename, index=False)
