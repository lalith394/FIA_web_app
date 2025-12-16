import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
import albumentations as A
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd
from torchsummary import summary
import io
import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset


def set_seed(seed=42):
    random.seed(seed)                 # Python random module
    np.random.seed(seed)               # NumPy
    torch.manual_seed(seed)            # PyTorch CPU
    torch.cuda.manual_seed(seed)       # PyTorch GPU
    torch.cuda.manual_seed_all(seed)   # All GPUs
    torch.backends.cudnn.deterministic = True  # Deterministic mode
    torch.backends.cudnn.benchmark = False     # Disable auto-tuner for reproducibility

def resize(dataset):
    image, mask, filename = dataset[0]

    # Convert tensors to NumPy arrays for plotting
    image_np = image.permute(1, 2, 0).numpy()  # CHW → HWC
    mask_np = mask.squeeze().numpy()           # Remove channel dim if present

    original_height, original_width = image_np.shape[:2]
    print(original_height, ", ", original_width)
    # Step 1: Calculate scaling factor
    scale = min(580 / original_width, 580 / original_height)

    # Step 2: Calculate new size with aspect ratio preserved
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Step 3: Round down to nearest multiple of 16
    new_width = (new_width // 16) * 16
    new_height = (new_height // 16) * 16

    """ resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image.shape """
    return new_height, new_width

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform_img=None, transform_mask=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform_img = transform_img
        self.transform_mask = transform_mask

        # Get image file list
        self.image_names = sorted(os.listdir(images_dir))  # Ensure same order
        self.mask_names = sorted(os.listdir(masks_dir))

        # Build mapping from stem → filename
        img_map = {os.path.splitext(f)[0]: f for f in self.image_names}
        mask_map = {os.path.splitext(f)[0]: f for f in self.mask_names}

        # Keep only those present in both
        common_stems = sorted(set(img_map.keys()) & set(mask_map.keys()))

        # Store full paths
        self.pairs = [
            (os.path.join(images_dir, img_map[stem]),
             os.path.join(masks_dir, mask_map[stem]))
            for stem in common_stems
        ]


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)  # keep original mode (e.g., L for label masks)

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)
        filename = os.path.basename(img_path)
        return image, mask, filename

class IDRiDDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)

        # Open as grayscale
        image = Image.open(img_path).convert("L")  # "L" = 1 channel grayscale

        if self.transform:
            image = self.transform(image)

        return image, image, img_name  # input, target, filename

class AutoencoderDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.image_files = file_list
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        filename = os.path.basename(img_path)
        return image, image, filename   # input, target (same for autoencoder)

class RFMiD(Dataset):
    def __init__(self, file_list, transform=None):
        self.image_files = file_list
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
        
        filename = os.path.splitext(os.path.basename(img_path))[0]
        return image, image, filename   # input, target (same for autoencoder)

def load_rfmid(dataset_path, batch_size=16, shuffle=False):
    transform = transforms.ToTensor()  # only convert to tensor, values in [0,1]

    all_files = [os.path.join(dataset_path, f) 
                 for f in os.listdir(dataset_path) 
                 if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    all_files = sorted(all_files)
    dataset = RFMiD(all_files, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

class RFMiD_Color(Dataset):
    def __init__(self, file_list, transform=None):
        self.image_files = file_list
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        filename = os.path.splitext(os.path.basename(img_path))[0]
        return image, image, filename   # input, target (same for autoencoder)

def load_rfmid_color(dataset_path, batch_size=16, shuffle=False):
    transform = transforms.Compose([
    transforms.Resize(128),   # resize for uniformity
    transforms.ToTensor(),           # convert to tensor
    ])

    all_files = [os.path.join(dataset_path, f) 
                 for f in os.listdir(dataset_path) 
                 if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    all_files = sorted(all_files)
    dataset = RFMiD_Color(all_files, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=2)
    return loader

def visualize(dataset, idx = 0):
    # Getting the sample at the index
    image, mask, filename = dataset[idx]

    # Convert tensors to NumPy arrays for plotting
    image_np = image.permute(1, 2, 0).numpy()  # CHW → HWC
    mask_np = mask.squeeze().numpy()           # Remove channel dim if present

    # Plot side by side
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    axs[0].imshow(image_np, cmap='gray')
    axs[0].set_title(f"Image: {filename}")
    axs[0].axis("off")

    axs[1].imshow(mask_np, cmap='gray')
    axs[1].set_title("Mask")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()

def load_dataset(dataset_path, batch_size = 2, shuffle = True):
    # Transforms for images (normalize to [0,1])
    transform_img = transforms.Compose([
        transforms.Resize((384, 576)),
        transforms.ToTensor(),
    ])

    # Transforms for masks (keep as long tensor for class labels)
    transform_mask = transforms.Compose([
        transforms.Resize((384, 576)),
        transforms.ToTensor()
    ])

    train_dataset = SegmentationDataset(
        images_dir=f"{dataset_path}/train/image",
        masks_dir=f"{dataset_path}/train/mask",
        transform_img=transform_img,
        transform_mask=transform_mask
    )

    test_dataset = SegmentationDataset(
        images_dir=f"{dataset_path}/test/image",
        masks_dir=f"{dataset_path}/test/mask",
        transform_img=transform_img,
        transform_mask=transform_mask
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    #visualize(train_dataset, 0)
    #visualize(test_dataset, 0)
    return train_loader, test_loader

def full_dataset(dataset_path, batch_size = 1, shuffle = True):
     train_loader, validation_loader = load_dataset(dataset_path, batch_size=1)
     train_dataset = train_loader.dataset
     test_dataset = validation_loader.dataset
     full_dataset = ConcatDataset([train_dataset, test_dataset])
     full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=shuffle)
     return full_loader

def resize_image(image):
    original_height, original_width = image.shape[:2]

    # Step 1: Calculate scaling factor
    scale = min(580 / original_width, 580 / original_height)

    # Step 2: Calculate new size with aspect ratio preserved
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Step 3: Round down to nearest multiple of 16
    new_width = (new_width // 16) * 16
    new_height = (new_height // 16) * 16

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image.shape


def load_idrid(dataset_path, batch_size=16, shuffle=True):
    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # scale to [-1, 1]
    ])

    all_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".jpg")]

    # Split into train and test based on filename
    train_files = [f for f in all_files if "test" not in os.path.basename(f).lower()]
    test_files  = [f for f in all_files if "test" in os.path.basename(f).lower()]

    train_dataset = AutoencoderDataset(train_files, transform=transform)
    test_dataset  = AutoencoderDataset(test_files, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def load_idrid_grayscale_aug(dataset_path, batch_size=16, shuffle=True):
    # Transform: grayscale → tensor → normalize
    transform = transforms.Compose([
        transforms.ToTensor(),  # shape = [1, H, W] since grayscale
    ])

    train_dataset = IDRiDDataset(os.path.join(dataset_path, "train"), transform=transform)
    test_dataset  = IDRiDDataset(os.path.join(dataset_path, "test"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader



def resize_dataset():
    # Paths
    input_dir = "Data_source/kagglehub/datasets/gyanpr02/indian-diabetic-retinopathy-image-datasetidrid/versions/1"   # folder containing 455 images
    output_dir = "Data_source/idrid"   # folder where resized images will be saved
    os.makedirs(output_dir, exist_ok=True)

    # Resize transformation
    resize_transform = transforms.Compose([
        transforms.Resize((384, 576)),  # (height, width)
    ])


    # Process images
    for img_name in tqdm(os.listdir(input_dir)):
        img_path = os.path.join(input_dir, img_name)

        
        try:
            img = Image.open(img_path).convert("RGB")  # ensure RGB
            img_resized = resize_transform(img)
            img_resized.save(os.path.join(output_dir, img_name))
        except Exception as e:
            print(f"Error processing {img_name}: {e}")


def process_idrid(input_dir = "Data_source/kagglehub/datasets/gyanpr02/indian-diabetic-retinopathy-image-datasetidrid/versions/1", output_dir = 'Data_source/idrid_cleaned'):
    # Input and output paths
    input_dir = input_dir
    output_dir = output_dir
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Define augmentations (Albumentations)
    augmentations = [
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.ElasticTransform(p=1.0, alpha = 120, sigma = 120*0.05),
        A.CLAHE(p=1.0), #Contrast Limited Adaptive Histogram Equalization
    ]

    # Desired size
    target_size = (576, 384)  # (width, height) for cv2.resize

    # Process images
    for img_name in tqdm(os.listdir(input_dir)):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        
        img_path = os.path.join(input_dir, img_name)

        # Load image and convert to grayscale
        img = cv2.imread(img_path)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            print(f"Error loading image: {img_path}")
            continue

        # Convert back to 3 channels (Albumentations requires HWC 3-channel format)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # Resize to 384 x 576
        gray_resized = cv2.resize(gray, target_size)

        # Decide whether test or train
        if "test" in img_name.lower():
            save_base = test_dir
        else:
            save_base = train_dir

        base_name, ext = os.path.splitext(img_name)

        # Save original grayscale resized image
        save_path = os.path.join(save_base, f"{base_name}_0{ext}")
        cv2.imwrite(save_path, gray_resized)

        # Apply augmentations and save
        if "test" not in img_name.lower():
            for i, aug in enumerate(augmentations, start=1):
                augmented = aug(image=gray_resized)
                aug_img = augmented["image"]

                save_path = os.path.join(save_base, f"{base_name}_{i}{ext}")
                cv2.imwrite(save_path, aug_img)
        break


def process_RFMiD(input_dir="Data_source/Raw Dataset/merged",
                  output_dir='Data_source/RFMiD'):
    
    os.makedirs(output_dir, exist_ok=True)

    # Define augmentations
    augmentations = [
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.ElasticTransform(p=1.0, alpha=120, sigma=120*0.05),
        A.CLAHE(p=1.0),
    ]

    # Desired size
    target_size = (576, 384)  # (width, height) for cv2.resize

    # Process images
    for img_name in tqdm(os.listdir(input_dir)):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(input_dir, img_name)

        # Load image and convert to grayscale
        img = cv2.imread(img_path)
        if img is not None:
            gray = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2GRAY)
        else:
            print(f"Error loading image: {img_path}")
            continue

        # Convert back to 3 channels
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # Resize
        gray_resized = cv2.resize(gray, target_size)

        base_name, ext = os.path.splitext(img_name)

        # Save original grayscale resized image
        save_path = os.path.join(output_dir, f"{base_name}_0{ext}")
        cv2.imwrite(save_path, gray_resized)

        # Apply augmentations and save
        for i, aug in enumerate(augmentations, start=1):
            augmented = aug(image=gray_resized)
            aug_img = augmented["image"]
            save_path = os.path.join(output_dir, f"{base_name}_{i}{ext}")
            cv2.imwrite(save_path, aug_img)

def process_RFMiD_color(input_dir="Data_source/Raw Dataset/merged",
                  output_dir='Data_source/RFMiD_color'):
    
    os.makedirs(output_dir, exist_ok=True)

    # Define augmentations
    augmentations = [
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.ElasticTransform(p=1.0, alpha=120, sigma=120*0.05),
        A.CLAHE(p=1.0),
    ]

    # Desired size (width, height) for cv2.resize
    target_size = (576, 384)

    # Process images
    for img_name in tqdm(os.listdir(input_dir)):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(input_dir, img_name)

        # Load image in RGB format
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image: {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize image
        img_resized = cv2.resize(img, target_size)

        base_name, ext = os.path.splitext(img_name)

        # Save original resized image (converted back to BGR for cv2.imwrite)
        save_path = os.path.join(output_dir, f"{base_name}_0{ext}")
        cv2.imwrite(save_path, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))

        # Apply augmentations and save
        for i, aug in enumerate(augmentations, start=1):
            augmented = aug(image=img_resized)
            aug_img = augmented["image"]
            save_path = os.path.join(output_dir, f"{base_name}_{i}{ext}")
            cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

def plot_loss_vs_epochs():
    dataset = r'D:\Projects\Fundal_Images_Analysis\Compact_DL_model\models\model_32_d4\data.csv'
    data = pd.read_csv(dataset)
    data2 = pd.read_csv(r'D:\Projects\Fundal_Images_Analysis\Compact_DL_model\models\model_64_d4\data.csv')
    x1 = data2['epoch']
    y1 = data2['val_loss']
    print(data)
    x = data['epoch']
    y = data['val_loss']
    plt.title('epoch vs val_loss')
    plt.plot(x, y)
    plt.plot(x1, y1)
    plt.xlabel('epoch')
    plt.ylabel('val_loss')
    plt.show()

def model_summary(model, path, images = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    buffer = io.StringIO()
    sys.stdout = buffer
    if images is not None:
        summary(model, input_size=images.shape[1:], batch_size=images.shape[0], device=device.type)
    else:
        summary(model, input_size=(3, 128, 192), device=device.type)

    sys.stdout = sys.__stdout__  # Restore normal stdout

    summary_text = buffer.getvalue()
    with open(path, "w") as f:
        f.write(summary_text)

def load_flowers102(batch_size = 16):
    transform = transforms.Compose([
    transforms.Resize((128, 128)),   # resize for uniformity
    transforms.ToTensor(),           # convert to tensor
    ])

    train_data = datasets.Flowers102(root='./Data_source/flowers102', split='train', download=True, transform=transform)
    val_data = datasets.Flowers102(root='./Data_source/flowers102', split='val', download=True, transform=transform)
    test_data = datasets.Flowers102(root='./Data_source/flowers102', split='test', download=True, transform=transform)
    full_dataset = ConcatDataset([train_data, val_data, test_data])

    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

if __name__ == "__main__":
    set_seed(54)
    dataset_path = 'Data_source/RFMiD_color'
    #train_loader, validation_loader = load_dataset(dataset_path)
    fl = load_rfmid_color(dataset_path)
    img, masks, label = next(iter(fl))
    img = img[0]
    img_np = img.permute(1, 2, 0).cpu().numpy()
    plt.imshow(img_np)
    plt.show()

    #test_data = load_rfmid('Data_source/RFMiD')
    #print(len(test_data))
    #visualize(test_data.dataset)
    #dataset = test_data.dataset
    #im1, m1, name = dataset[15999]
    #print(name)
    #idrid_train, idrid_test = load_idrid_grayscale_aug('Data_source/idrid_grayscale_aug', batch_size=1)
    #print(len(idrid_train), len(idrid_test))
    #process_RFMiD_color()





