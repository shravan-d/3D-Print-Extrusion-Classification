import torchvision.transforms as transforms
from train import *
from PIL import Image
import numpy as np


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.43335001, 0.43360192, 0.42602362), (0.28486016, 0.28320433, 0.28699529)),
])
flipTransform = transforms.RandomHorizontalFlip(p=1)
zoomTransform = transforms.RandomResizedCrop(input_size[:: -1], scale=(0.7, 1))
colorTransform = transforms.ColorJitter(brightness=(0.65, 0.9), contrast=(1.1, 1.35))


def resize_image(image):
    w, h = image.size
    cut_h = h - aspect_ratio[1] * w / aspect_ratio[0]
    image = image.crop((0, cut_h / 2, w, h-(cut_h/2)))
    image = image.resize(input_size)
    return image


def generate_batches(df, train=True, validation=False):
    if train:
        labels = df['has_under_extrusion'].tolist()
    printers = df['printer_id'].unique().tolist()
    printer_domain_map = {printer_id: idx for idx, printer_id in enumerate(printers)}
    domains = [printer_domain_map[domain] for domain in df['printer_id'].tolist()]
    prints = df['print_id'].unique().tolist()
    print_domain_map = {print_id: idx for idx, print_id in enumerate(prints)}
    print_jobs = [print_domain_map[print] for print in df['print_id'].tolist()]

    image_paths = df['img_path'].tolist()
    current = 0
    while current < len(df):
        batch_images, batch_domains, batch_labels, batch_paths, batch_prints = [], [], [], [], []
        batch_idx = 0
        reserve_images, reserve_domains, reserve_labels, reserve_prints = [], [], [], []

        while batch_idx < batch_size:
            if current + batch_idx >= len(df):
                break

            image_path = image_paths[current + batch_idx]
            domain = domains[current + batch_idx]
            print_job = print_jobs[current + batch_idx]
            filename = '/kaggle/input/early-detection-of-3d-printing-issues/images/' + image_path
            image = Image.open(filename)
            image = resize_image(image)
            if train:
                label = labels[current + batch_idx]
                batch_labels.append(label)
            batch_domains.append(domain)
            batch_prints.append(print_job)
            batch_images.append(transform(image))
            batch_paths.append(image_path)
            batch_idx += 1
        current += batch_size
        if train:
            yield batch_images, np.array(batch_labels), np.array(batch_domains), np.array(batch_prints)
        else:
            yield batch_images, batch_paths