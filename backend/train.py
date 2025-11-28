from utils import load_dataset, set_seed, load_idrid, load_idrid_grayscale_aug, load_rfmid, load_rfmid_color, load_flowers102
from model import UNet, AutoEncoder, AutoEncoder_RFMiD, VAE
from torch.optim import Adam
from metrics import DiceLoss, dice_coef, iou, mse
from tqdm import tqdm
import torch
import os
import pandas as pd
from sklearn.metrics import f1_score, jaccard_score, recall_score, precision_score
import numpy as np
import torch.nn as nn
from utils import model_summary
from torch import amp
from metrics import psnr, ssim


def train_Unet(model_name,
               num_outputs = 1,
               num_channels = [64, 128, 256, 512, 1024, 512, 256, 128, 64], 
               dataset_path="Data_source/Augmented/Data", 
               epochs=60, 
               batch_size=2, 
               lr=1e-4,
               patience = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if(torch.cuda.is_available()):
        print("Using Cuda device!")
    else:
        print("Using CPU")
    set_seed(42)
    """ Hyperparameters """
    batch_size = batch_size
    epochs = epochs
    lr = lr
    model_dir = f"models/{model_name}"
    model_name = f"{model_name}.pth"
    data_stats_path = f"{model_dir}/stats.csv"
    dataset_path = dataset_path

    """ Handling model save dir """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # --- Initializing model, loss functions and optimizer ---
    model = UNet(num_outputs=num_outputs, num_channels=num_channels).to(device)
    criterion = DiceLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # --- Loading train and test data ---
    train_loader, test_loader = load_dataset(dataset_path, batch_size=batch_size)
    sample_images, sample_labels, sample_img_name = next(iter(train_loader))
    print("Sample Images shape: ", sample_images.shape)
    print("Sample labels shape: ", sample_labels.shape)

    train_stats = []
    # --- Training ---
    best_loss = float('inf')
    number_of_epochs_without_improvement = 0
    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        running_dice = 0.0
        running_iou = 0.0
        running_recall = 0.0
        running_precision = 0.0

        """ Training images """
        for img, masks, name in tqdm(train_loader):
            img = img.to(device)
            masks = masks.to(device)

            """ Forward """
            outputs = model(img)
            loss = criterion(masks, outputs)

            """ Backward """
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            running_dice += dice_coef(masks, outputs).item()
            running_iou += iou(masks, outputs).item()
    
            ypred = outputs.cpu().detach()
            ypred = (ypred>0.5).int()
            y_true = masks.squeeze().cpu().numpy().astype(int)
            y_pred = ypred.squeeze().cpu().numpy().astype(int)
            y_true_flat = y_true.flatten()
            y_pred_flat = y_pred.flatten()

            running_recall += recall_score(y_true_flat,
                                           y_pred_flat,
                                           average="binary", zero_division=0)
            running_precision += precision_score(y_true_flat,
                                                 y_pred_flat,
                                                 average="binary", zero_division=0)

        val_loss = 0.0

        """ Validation images """
        for img, masks, name in tqdm(test_loader):
            img = img.to(device)
            masks = masks.to(device)
            outputs = model(img)
            loss = criterion(masks, outputs)
            val_loss = val_loss + loss.item()

        avg_val_loss = val_loss/len(test_loader)
        avg_loss = running_loss/len(train_loader)
        avg_dice = running_dice / len(train_loader)
        avg_iou = running_iou / len(train_loader)
        avg_recall = running_recall / len(train_loader)
        avg_precision = running_precision / len(train_loader)

        # Get prediction for validation data
        # compute the average test loss

        print(f"Epoch [{ep+1}/{epochs}] - val_loss: {avg_val_loss} - Loss: {avg_loss:.4f} - Dice_coef: {avg_dice:.4f} - IoU: {avg_iou:.4f} - Recall: {avg_recall:.4f} - Precision: {avg_precision:.4f}")
        cur_stats = [[ep+1, avg_val_loss, avg_loss, avg_dice, avg_iou, avg_recall, avg_precision]]
        train_stats.append(cur_stats)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(model_dir, model_name))
            print(f"✅ Saved new best model at epoch {ep+1} with loss {avg_loss:.4f}")
            number_of_epochs_without_improvement = 0
        else:
            number_of_epochs_without_improvement += 1
    
        data_stats = pd.DataFrame(cur_stats, columns=["Epoch", "Avg_val_loss","Avg_Loss", "Avg_dice_coef", "Avg_IoU", "Avg_Recall", "Avg_Precision"])
        file_exists = os.path.isfile(data_stats_path)
        data_stats.to_csv(data_stats_path, mode='a', header=not file_exists, index=False)
        if(number_of_epochs_without_improvement > patience):
            break

def train_AutoEncoder(model_name,
               num_outputs = 1,
               num_channels = [64, 128, 256, 512, 1024, 512, 256, 128, 64], 
               dataset_path="Data_source/idrid_grayscale_aug", 
               epochs=120, 
               batch_size=4, 
               lr=1e-4,
               patience = 20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if(torch.cuda.is_available()):
        print("Using Cuda device!")
    else:
        print("Using CPU")
    set_seed(42)
    """ Hyperparameters """
    batch_size = batch_size
    epochs = epochs
    lr = lr
    model_dir = f"models/{model_name}"
    model_name = f"{model_name}.pth"
    data_stats_path = f"{model_dir}/stats.csv"
    dataset_path = dataset_path

    """ Handling model save dir """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # --- Initializing model, loss functions and optimizer ---
    model = AutoEncoder(num_outputs=num_outputs, num_channels=num_channels).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # --- Loading train and test data ---
    train_loader, test_loader = load_idrid_grayscale_aug(dataset_path, batch_size=batch_size)
    sample_images, sample_labels, sample_img_name = next(iter(train_loader))
    print("Sample Images shape: ", sample_images.shape)
    print("Sample labels shape: ", sample_labels.shape)

    train_stats = []
    # --- Training ---
    best_loss = float('inf')
    number_of_epochs_without_improvement = 0
    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        

        """ Training images """
        for img,_, name in tqdm(train_loader):
            img = img.to(device)

            """ Forward """
            outputs = model(img)
            loss = criterion(img, outputs)

            """ Backward """
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


        val_loss = 0.0

        """ Validation images """
        for img, _, name in tqdm(test_loader):
            img = img.to(device)
            outputs = model(img)
            loss = criterion(img, outputs)
            val_loss = val_loss + loss.item()

        avg_val_loss = val_loss/len(test_loader)
        avg_loss = running_loss/len(train_loader)
        

        # Get prediction for validation data
        # compute the average test loss

        print(f"Epoch [{ep+1}/{epochs}] - val_loss: {avg_val_loss} - Loss: {avg_loss:.4f}")
        cur_stats = [[ep+1, avg_val_loss, avg_loss]]
        train_stats.append(cur_stats)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(model_dir, model_name))
            print(f"✅ Saved new best model at epoch {ep+1} with loss {avg_loss:.4f}")
            number_of_epochs_without_improvement = 0
        else:
            number_of_epochs_without_improvement += 1
    
        data_stats = pd.DataFrame(cur_stats, columns=["Epoch", "Avg_val_loss","Avg_Loss"])
        file_exists = os.path.isfile(data_stats_path)
        data_stats.to_csv(data_stats_path, mode='a', header=not file_exists, index=False)
        if(number_of_epochs_without_improvement >= patience-1):
            break

def train_AutoEncoder_RFMiD(model_name,
               num_outputs = 1,
               num_channels = [64, 128, 256, 512, 1024, 512, 256, 128, 64], 
               dataset_path="Data_source/RFMiD", 
               epochs=120, 
               batch_size=4, 
               lr=1e-4,
               patience = 20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if(torch.cuda.is_available()):
        print("Using Cuda device!")
    else:
        print("Using CPU")
    set_seed(42)
    """ Hyperparameters """
    batch_size = batch_size
    epochs = epochs
    lr = lr
    model_dir = f"models/{model_name}"
    model_name = f"{model_name}.pth"
    data_stats_path = f"{model_dir}/stats.csv"
    dataset_path = dataset_path

    """ Handling model save dir """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # --- Initializing model, loss functions and optimizer ---
    model = AutoEncoder(num_outputs=num_outputs, num_channels=num_channels).to(device)
    model_summary(model, f'{model_dir}/{model_name}_sm.txt')
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # --- Loading train and test data ---
    train_loader = load_rfmid(dataset_path, batch_size=batch_size)
    sample_images, sample_labels, sample_img_name = next(iter(train_loader))
    print("Sample Images shape: ", sample_images.shape)
    print("Sample labels shape: ", sample_labels.shape)

    train_stats = []
    # --- Training ---
    best_loss = float('inf')
    number_of_epochs_without_improvement = 0
    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        

        """ Training images """
        for img,_, name in tqdm(train_loader):
            img = img.to(device)

            """ Forward """
            outputs = model(img)
            loss = criterion(img, outputs)

            """ Backward """
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss/len(train_loader)
        

        print(f"Epoch [{ep+1}/{epochs}] -Loss: {avg_loss:.4f}")
        cur_stats = [[ep+1, avg_loss]]
        train_stats.append(cur_stats)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(model_dir, model_name))
            print(f"✅ Saved new best model at epoch {ep+1} with loss {avg_loss:.4f}")
            number_of_epochs_without_improvement = 0
        else:
            number_of_epochs_without_improvement += 1
    
        data_stats = pd.DataFrame(cur_stats, columns=["Epoch", "Avg_Loss"])
        file_exists = os.path.isfile(data_stats_path)
        data_stats.to_csv(data_stats_path, mode='a', header=not file_exists, index=False)
        if(number_of_epochs_without_improvement >= patience):
            break

def train_AutoEncoder_RFMiD_color(model_name,
               num_outputs = 3,
               num_channels = [64, 128, 256, 512, 1024, 512, 256, 128, 64], 
               dataset_path="Data_source/RFMiD_color", 
               epochs=120, 
               batch_size=4, 
               lr=1e-4,
               patience = 20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    if(torch.cuda.is_available()):
        print("Using Cuda device!")
    else:
        print("Using CPU")
    set_seed(42)
    """ Hyperparameters """
    batch_size = batch_size
    epochs = epochs
    lr = lr
    model_dir = f"models/AE2/{model_name}"
    model_name = f"{model_name}.pth"
    checkpoint_path = f"{model_dir}/{model_name}"
    data_stats_path = f"{model_dir}/stats.csv"
    dataset_path = dataset_path

    """ Handling model save dir """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)


    # --- Initializing model, loss functions and optimizer ---
    model = AutoEncoder_RFMiD(num_outputs=num_outputs, num_channels=num_channels).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    start_epoch = 0
    best_loss = float('inf')

    if os.path.exists(checkpoint_path):
        print(f"🔁 Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        print(f"Resuming from epoch {start_epoch} with best_loss={best_loss:.6f}")

    # --- Loading train and test data ---
    train_loader = load_rfmid_color(dataset_path, batch_size=batch_size)
    sample_images, sample_labels, sample_img_name = next(iter(train_loader))
    print("Sample Images shape: ", sample_images.shape)
    print("Sample labels shape: ", sample_labels.shape)
    
    model_summary(model, f'{model_dir}/{model_name}_sm.txt', sample_images)

    train_stats = []
    # --- Training ---
    
    number_of_epochs_without_improvement = 0
    scaler = amp.grad_scaler.GradScaler(device="cuda")
    for ep in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        total_psnr, total_ssim = 0.0, 0.0

        # Training images 
        for img,_, name in tqdm(train_loader):
            img = img.to(device)
            optimizer.zero_grad(set_to_none=True)

            # Forward 
            with amp.autocast_mode.autocast(device_type="cuda"):
                outputs = model(img)
                loss = criterion(img, outputs)

            # Backward 
            scaler.scale(loss).backward()
            scaler.step(optimizer=optimizer)
            scaler.update()

            running_loss += loss.item()

            with torch.no_grad():
                total_psnr += psnr(outputs, img)
                total_ssim += ssim(outputs, img)
        
        n_batches = len(train_loader)
        avg_loss = running_loss / n_batches
        avg_psnr = total_psnr / n_batches
        avg_ssim = total_ssim / n_batches
        

        print(f"Epoch [{ep+1}/{epochs}] -Loss: {avg_loss:.4f} || PSNR: {avg_psnr:.4f} || SSIM: {avg_ssim:.4f}")
        cur_stats = [[ep+1, avg_loss, float(avg_psnr), avg_ssim]]
        train_stats.append(cur_stats)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(model_dir, model_name))
            print(f"✅ Saved new best model at epoch {ep+1} with loss {avg_loss:.4f}")
            number_of_epochs_without_improvement = 0
        else:
            number_of_epochs_without_improvement += 1
    
        data_stats = pd.DataFrame(cur_stats, columns=["Epoch", "Avg_Loss", "Avg_PSNR", "Avg_SSIM"])
        file_exists = os.path.isfile(data_stats_path)
        data_stats.to_csv(data_stats_path, mode='a', header=not file_exists, index=False)
        if((ep+1) == 25):
            break



def train_AutoEncoder_flowers102(model_name,
               num_outputs = 3,
               num_channels = [64, 128, 256, 512, 1024, 512, 256, 128, 64],  
               epochs=120, 
               batch_size=16, 
               lr=1e-4,
               patience = 20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if(torch.cuda.is_available()):
        print("Using Cuda device!")
    else:
        print("Using CPU")
    set_seed(42)
    """ Hyperparameters """
    batch_size = batch_size
    epochs = epochs
    lr = lr
    model_dir = f"models/{model_name}"
    model_name = f"{model_name}.pth"
    checkpoint_path = f"{model_dir}/{model_name}"
    data_stats_path = f"{model_dir}/stats.csv"

    """ Handling model save dir """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)


    # --- Initializing model, loss functions and optimizer ---
    model = AutoEncoder(num_outputs=num_outputs, num_channels=num_channels).to(device)
    model_summary(model, f'{model_dir}/{model_name}_sm.txt')
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    start_epoch = 0
    best_loss = float('inf')

    if os.path.exists(checkpoint_path):
        print(f"🔁 Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        print(f"Resuming from epoch {start_epoch} with best_loss={best_loss:.6f}")

    # --- Loading train and test data ---
    train_loader = load_flowers102(batch_size=batch_size)
    sample_images, _ = next(iter(train_loader))
    print("Sample Images shape: ", sample_images.shape)

    train_stats = []
    # --- Training ---
    
    number_of_epochs_without_improvement = 0
    for ep in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        

        """ Training images """
        for img, _ in tqdm(train_loader):
            img = img.to(device)

            """ Forward """
            outputs = model(img)
            loss = criterion(img, outputs)

            """ Backward """
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss/len(train_loader)
        

        print(f"Epoch [{ep+1}/{epochs}] -Loss: {avg_loss:.4f}")
        cur_stats = [[ep+1, avg_loss]]
        train_stats.append(cur_stats)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(model_dir, model_name))
            print(f"✅ Saved new best model at epoch {ep+1} with loss {avg_loss:.4f}")
            number_of_epochs_without_improvement = 0
        else:
            number_of_epochs_without_improvement += 1
    
        data_stats = pd.DataFrame(cur_stats, columns=["Epoch", "Avg_Loss"])
        file_exists = os.path.isfile(data_stats_path)
        data_stats.to_csv(data_stats_path, mode='a', header=not file_exists, index=False)
        if(number_of_epochs_without_improvement >= patience):
            break

def train_Ladder_VAE(model_name,
               num_outputs = 3,
               num_channels = [64, 128, 256, 512, 1024, 512, 256, 128, 64],  
               dataset_path="Data_source/RFMiD_color",
               epochs=240, 
               batch_size=16, 
               lr=1e-4,
               patience = 20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if(torch.cuda.is_available()):
        print("Using Cuda device!")
    else:
        print("Using CPU")
    set_seed(42)


    """ Directories """
    model_dir = f"models/VAE/{model_name}"
    model_name = f"{model_name}.pth"
    checkpoint_path = f"{model_dir}/{model_name}"
    data_stats_path = f"{model_dir}/stats.csv"

    """ Handling model save dir """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    """ Initialization """
    model = LadderVAE().to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    model_summary(model, f'{model_dir}/{model_name}_sm.txt')
    criterion = nn.MSELoss(reduction="sum")
    train_stats = []
    start_epoch = 0
    best_loss = float('inf')

    if os.path.exists(checkpoint_path):
        print(f"🔁 Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        print(f"Resuming from epoch {start_epoch} with best_loss={best_loss:.6f}")
    
    train_loader = load_rfmid_color(dataset_path, batch_size=batch_size)
    sample_images, sample_labels, sample_img_name = next(iter(train_loader))
    print("Sample Images shape: ", sample_images.shape)
    print("Sample labels shape: ", sample_labels.shape)

    scaler = amp.grad_scaler.GradScaler(device="cuda")
    for ep in range(start_epoch, epochs):
        running_loss = 0.0
        for data, _ , _ in tqdm(train_loader):
            # Get a batch of training data and move it to the device
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            with amp.autocast_mode.autocast(device_type="cuda"):
                decoded, mu1, log_var1, mu2, log_var2 = model(data)
                # Compute the loss and perform backpropagation
                recon_loss = criterion(decoded, data)
                KLD1 = -0.5 * torch.sum(1 + log_var1 - mu1.pow(2) - log_var1.exp())
                KLD2 = -0.5 * torch.sum(1 + log_var2 - mu2.pow(2) - log_var2.exp())
                # KL annealing or small beta
                beta = 0
                loss = recon_loss + KLD2 + 0.5*KLD2


            # Backward 
            scaler.scale(loss).backward()
            scaler.step(optimizer=optimizer)
            scaler.update()

            # Update the running loss
            running_loss += loss.item() * data.size(0)

        avg_loss = running_loss/len(train_loader)
        print(f"Epoch [{ep+1}/{epochs}] -Loss: {avg_loss:.4f}")
        cur_stats = [[ep+1, avg_loss]]
        train_stats.append(cur_stats)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(model_dir, model_name))
            print(f"✅ Saved new best model at epoch {ep+1} with loss {avg_loss:.4f}")
    
        data_stats = pd.DataFrame(cur_stats, columns=["Epoch", "Avg_Loss"])
        file_exists = os.path.isfile(data_stats_path)
        data_stats.to_csv(data_stats_path, mode='a', header=not file_exists, index=False)


def train_VAE_RFMiD_color(model_name,
               num_outputs = 3,
               num_channels = [64, 128, 256, 512, 1024, 512, 256, 128, 64], 
               dataset_path="Data_source/RFMiD_color", 
               epochs=120, 
               batch_size=4, 
               lr=1e-4,
               patience = 20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    if(torch.cuda.is_available()):
        print("Using Cuda device!")
    else:
        print("Using CPU")
    set_seed(42)
    """ Hyperparameters """
    batch_size = batch_size
    epochs = epochs
    lr = lr
    model_dir = f"models/VAE_1/{model_name}"
    model_name = f"{model_name}.pth"
    checkpoint_path = f"{model_dir}/{model_name}"
    data_stats_path = f"{model_dir}/stats.csv"
    dataset_path = dataset_path

    """ Handling model save dir """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)


    # --- Initializing model, loss functions and optimizer ---
    model = VAE(num_outputs=num_outputs, num_channels=num_channels).to(device)
    criterion = nn.MSELoss(reduction="sum")
    optimizer = Adam(model.parameters(), lr=lr)
    start_epoch = 0
    best_loss = float('inf')

    if os.path.exists(checkpoint_path):
        print(f"🔁 Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        print(f"Resuming from epoch {start_epoch} with best_loss={best_loss:.6f}")

    # --- Loading train and test data ---
    train_loader = load_rfmid_color(dataset_path, batch_size=batch_size)
    sample_images, sample_labels, sample_img_name = next(iter(train_loader))
    print("Sample Images shape: ", sample_images.shape)
    print("Sample labels shape: ", sample_labels.shape)
    
    model_summary(model, f'{model_dir}/{model_name}_sm.txt', sample_images)

    train_stats = []
    # --- Training ---
    
    number_of_epochs_without_improvement = 0
    scaler = amp.grad_scaler.GradScaler(device="cuda")
    for ep in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        total_mse, total_psnr, total_ssim = 0.0, 0.0, 0.0

        # Training images 
        for img,_, name in tqdm(train_loader):
            img = img.to(device)
            optimizer.zero_grad(set_to_none=True)

            # Forward 
            with amp.autocast_mode.autocast(device_type="cuda"):
                outputs, mu_1, logvar_1, mu_2, logvar_2 = model(img)
                KLD1 = -0.5 * torch.sum(1 + logvar_1 - mu_1.pow(2) - logvar_1.exp())
                KLD2 = -0.5 * torch.sum(1 + logvar_2 - mu_2.pow(2) - logvar_2.exp())
                loss = criterion(img, outputs) + (1e-3 * (KLD1 + KLD2))

            # Backward 
            scaler.scale(loss).backward()
            scaler.step(optimizer=optimizer)
            scaler.update()

            running_loss += loss.item()

            with torch.no_grad():
                total_psnr += psnr(outputs, img)
                total_ssim += ssim(outputs, img)
                total_mse += mse(outputs, img).item()
        
        n_batches = len(train_loader)
        avg_loss = running_loss / n_batches
        avg_psnr = total_psnr / n_batches
        avg_ssim = total_ssim / n_batches
        avg_mse = total_mse / n_batches
        

        print(f"Epoch [{ep+1}/{epochs}] -Loss: {avg_loss:.4f} || MSE: {avg_mse:.4f} || PSNR: {avg_psnr:.4f} || SSIM: {avg_ssim:.4f}")
        cur_stats = [[ep+1, avg_loss, avg_mse, float(avg_psnr), avg_ssim]]
        train_stats.append(cur_stats)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(model_dir, model_name))
            print(f"✅ Saved new best model at epoch {ep+1} with loss {avg_loss:.4f}")
            number_of_epochs_without_improvement = 0
        else:
            number_of_epochs_without_improvement += 1
    
        data_stats = pd.DataFrame(cur_stats, columns=["Epoch", "Avg_Loss", "Avg_MSE","Avg_PSNR", "Avg_SSIM"])
        file_exists = os.path.isfile(data_stats_path)
        data_stats.to_csv(data_stats_path, mode='a', header=not file_exists, index=False)
        if(number_of_epochs_without_improvement == 20):
            break




if __name__ == "__main__":
    pass
