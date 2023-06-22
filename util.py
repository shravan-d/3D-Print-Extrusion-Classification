from train import *
import torch
import sys
from matplotlib import pyplot as plt


def progress_bar(start, i, training_batch_count):
    elapsed_time = (datetime.datetime.now() - start).seconds // 60
    sys.stdout.write('\r')
    sys.stdout.write("Validating: [%-50s] %d%% || ETA: %d minutes"
                     % ('=' * int(50 * (i + 1) / training_batch_count), int(100 * (i + 1) / training_batch_count),
                        (elapsed_time / (i + 1)) * (training_batch_count - i)))
    sys.stdout.flush()


def show_image_grid(images, count, title=''):
    _, axs = plt.subplots(int(1 + (count - 1) / 4), 4, figsize=(count * 4, 12))
    axs = axs.flatten()
    for img, ax in zip(images, axs):
        if img.mode == 'RGB':
            ax.imshow(img, vmin=0, vmax=255)
        else:
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title(title)
    plt.show()


def batch_shuffle(df):
    index_list = np.array(df.index)
    index_list = index_list[:int(len(df)/batch_size)*batch_size]
    np.random.shuffle(np.reshape(index_list, (-1, batch_size)))
    return df.loc[index_list, :]


def print_loss_function(output, target, print_jobs):
    unique_print_jobs = torch.unique(print_jobs)
    num_print_jobs = unique_print_jobs.shape[0]
    pj_losses = []
    for i in range(num_print_jobs):
        pj_mask = (print_jobs == unique_print_jobs[i])
        pj_output = output[pj_mask]
        pred = torch.max(pj_output, dim=1)[1]
        count_0, count_1 = 0, 0
        for j in range(len(pj_output)):
            if pred[j] == 0:
                count_0 += 1
            if pred[j] == 1:
                count_1 += 1
        pj_loss = min(count_0, count_1) / len(pj_output)
        pj_losses.append(pj_loss)
    pj_loss = sum(pj_losses) / num_print_jobs
    return torch.tensor(pj_loss, requires_grad=True).to(device)