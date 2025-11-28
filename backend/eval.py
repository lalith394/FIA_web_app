import torch
from utils import load_dataset, set_seed, full_dataset, load_idrid_grayscale_aug, load_rfmid, load_rfmid_color, load_flowers102
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from model import UNet, AutoEncoder, AutoEncoder_RFMiD
from metrics import DiceLoss
from sklearn.metrics import accuracy_score, jaccard_score, precision_score, f1_score, recall_score
from timeit import default_timer as timer
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms as T
import torch.nn as nn
from metrics import psnr, ssim
import lpips

def save_results_rgb(ori_y, y_pred, save_image_path):

    pred_image = np.zeros((y_pred.shape[0], y_pred.shape[1], 3))
    _y_pred = y_pred[:, :]
    _ori_y = ori_y[:, :]
    pred_image[:, :, 0] = ((_y_pred > 0.5) & (_ori_y <= 0.5)) * 255
    pred_image[:, :, 1] = ((_y_pred > 0.5) & (_ori_y  > 0.5)) * 255
    pred_image[:, :, 2] = ((_ori_y  > 0.5) & (_y_pred <= 0.5 )) * 255

    print(" saving result", save_image_path)
    cv2.imwrite(save_image_path, pred_image)

def save_normal_results(_, y_pred, save_image_path):
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) * 255
    cv2.imwrite(save_image_path, y_pred)

""" Visualization of predictions """
def visualize_segmentations(model, test_loader, device, output_dir):
    model.eval()
    imgs, masks, names = next(iter(test_loader))
    imgs = imgs.to(device)

    with torch.no_grad():
        outputs = model(imgs)
        outputs = torch.sigmoid(outputs)  # for binary masks

    # Move to CPU for plotting
    imgs = imgs.cpu()
    outputs = outputs.cpu()

    n = min(8, imgs.size(0))  # number of images to show
    plt.figure(figsize=(16, 4))

    for i in range(n):
        # Original RGB image (CHW -> HWC)
        img_np = imgs[i].permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype("uint8")  # if normalized [0,1]

        # Predicted mask
        mask_np = outputs[i].squeeze().numpy()

        # Original
        plt.subplot(2, n, i + 1)
        plt.imshow(img_np)
        plt.title(f"Original: {names}")
        plt.axis('off')

        # Prediction
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(mask_np, cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/visualization.png')
    plt.show()

def visualize_reconstructions(model, test_loader, device, results_dir, avg_loss):
    model.eval()
    imgs, _, name = next(iter(test_loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        outputs = model(imgs)

    imgs = imgs.cpu()
    outputs = outputs.cpu()

    n = 4  # number of images to show
    plt.figure(figsize=(16,6))
    plt.title(f'Avg.loss: {avg_loss}')
    plt.axis('off')
    for i in range(n):
        # Original
        plt.subplot(2, n, i + 1)
        plt.imshow(imgs[i].squeeze(), cmap='gray')
        plt.title(f"{name[i]}")
        plt.axis('off')

        # Reconstructed
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/visualization.png')
    plt.show()

def visualize_reconstructions_color(model, test_loader, device, results_dir, eval_res):
    model.eval()

    imgs, _, names = next(iter(test_loader))
    imgs = imgs.to(device)

    with torch.no_grad():
        outputs = model(imgs)

    imgs = imgs.cpu()
    outputs = outputs.cpu()

    n = min(4, imgs.size(0))  # number of images to show (max 4)
    plt.figure(figsize=(16, 8))
    plt.suptitle(f'{eval_res}', fontsize=10)

    for i in range(n):
        # --- Original Image ---
        plt.subplot(2, n, i + 1)
        img_np = imgs[i].permute(1, 2, 0).numpy()  # (C,H,W) → (H,W,C)
        #img_np = np.clip(img_np, 0, 1)              # ensure values in [0,1]
        plt.imshow(img_np)
        plt.title(f"{names[i]}")
        plt.axis('off')

        # --- Reconstructed Image ---
        plt.subplot(2, n, i + 1 + n)
        out_np = outputs[i].permute(1, 2, 0).numpy()
        #out_np = np.clip(out_np, 0, 1)
        plt.imshow(out_np)
        plt.title("Reconstructed")
        plt.axis('off')

    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'visualization.png'))
    plt.show()

def visualize_reconstructions_flower102(model, test_loader, device, results_dir, avg_loss):
    model.eval()

    imgs, names = next(iter(test_loader))
    imgs = imgs.to(device)

    with torch.no_grad():
        outputs = model(imgs)

    imgs = imgs.cpu()
    outputs = outputs.cpu()

    n = min(4, imgs.size(0))  # number of images to show (max 4)
    plt.figure(figsize=(16, 6))
    plt.suptitle(f'Avg. Loss: {avg_loss:.6f}', fontsize=14)

    for i in range(n):
        # --- Original Image ---
        plt.subplot(2, n, i + 1)
        img_np = imgs[i].permute(1, 2, 0).numpy()  # (C,H,W) → (H,W,C)
        #img_np = np.clip(img_np, 0, 1)              # ensure values in [0,1]
        plt.imshow(img_np)
        plt.title(f"{names[i]}")
        plt.axis('off')

        # --- Reconstructed Image ---
        plt.subplot(2, n, i + 1 + n)
        out_np = outputs[i].permute(1, 2, 0).numpy()
        #out_np = np.clip(out_np, 0, 1)
        plt.imshow(out_np)
        plt.title("Reconstructed")
        plt.axis('off')

    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'visualization.png'))
    plt.show()


def eval(model_name, num_channels = [64, 128, 256, 512, 1024, 512, 256, 128, 64], dataset_path = "Data_source/Augmented/Data", out_dir = None, test_dataset = True, shuffle = True, rgb = False):
    # --- Config ---
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    model_dir = f'models/{model_name}'
    model_path = f'{model_dir}/{model_name}.pth'
    dataset_path = dataset_path
    if out_dir is None:
        out_dir = model_name
    results_dir = f"output/{out_dir}"

    if(not os.path.exists(results_dir) and rgb is not None):
        os.makedirs(results_dir)

    if(not os.path.exists(model_dir)):
        os.makedirs(model_dir)
        
    # --- Load test data ---
    if(test_dataset is True):
        _, test_loader = load_dataset(dataset_path, batch_size=batch_size, shuffle=shuffle)
    elif (test_dataset is False):
       test_loader, _ = load_dataset(dataset_path, batch_size=batch_size, shuffle=shuffle)
    else:
        test_loader = full_dataset(dataset_path, batch_size=batch_size, shuffle=shuffle)

    # --- Load model ---
    model = UNet(num_channels=num_channels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.eval()

    # --- Define loss function ---
    criterion = DiceLoss()

    # --- Evaluate ---
    total_loss = 0.0

    SCORE = []
    with torch.no_grad():
        start_time = timer()
        for imgs, masks, names in tqdm(test_loader):
            imgs = imgs.to(device)
            masks = masks.to(device)
            outputs, d4_feats = model(imgs, return_features = True)
            p1 = (outputs > 0.5).int().squeeze(1) #[1, 384, 576]

            #Saving Predictions
            for i in range(imgs.size(0)):
                img_name = names[i] if isinstance(names[i], str) else f"img_{i}.png"
                save_path = os.path.join(results_dir, img_name)
                if rgb is None:
                    break
                if rgb is True:
                    save_results_rgb(masks[i].cpu().squeeze(), p1[i].cpu(), save_path)
                else:
                    save_normal_results(masks[i].cpu().squeeze(), p1[i].cpu(), save_path)
            
            #Saving Intermediate feature channels
            for i in range(imgs.size(0)):
                img_name = names[i] if isinstance(names[i], str) else f"img_{i}"
                feature_dir = os.path.join(results_dir, f"{img_name}_d4_layer")
                os.makedirs(feature_dir, exist_ok=True)

                # d4_feats shape: [B, C, H, W]
                d4_np = d4_feats[i].cpu().numpy()  # shape [64, H, W]

                for ch in range(d4_np.shape[0]):
                    feat_map = d4_np[ch]

                    # Normalize to [0,255] for saving
                    feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)
                    feat_map = (feat_map * 255).astype("uint8")

                    save_path = os.path.join(feature_dir, f"channel_{ch}.png")
                    cv2.imwrite(save_path, feat_map)
            
            #metrics evaluation
            preds = (outputs > 0.5).int()  # Threshold at 0.5
            # Squeeze and convert to numpy
            y_true = masks.squeeze().cpu().numpy().astype(int)
            y_pred = preds.squeeze().cpu().numpy().astype(int)
            # Flatten
            y_true_flat = y_true.flatten()
            y_pred_flat = y_pred.flatten()

            loss = criterion(outputs, masks)
            total_loss += loss.item()

            """ Calculate the metrics """
            acc_value = accuracy_score(y_true_flat, y_pred_flat)
            f1_value = f1_score(y_true_flat, y_pred_flat, labels=[0, 1], average="binary")
            jac_value = jaccard_score(y_true_flat, y_pred_flat, labels=[0, 1], average="binary")
            recall_value = recall_score(y_true_flat, y_pred_flat, labels=[0, 1], average="binary")
            precision_value = precision_score(y_true_flat, y_pred_flat, labels=[0, 1], average="binary")
            SCORE.append([names, acc_value, f1_value, jac_value, recall_value, precision_value])
        end_time = timer()
        score = [s[1:] for s in SCORE]
        score = np.mean(score, axis=0)
        print(f"Accuracy: {score[0]:0.5f}")
        print(f"F1: {score[1]:0.5f}")
        print(f"Jaccard: {score[2]:0.5f}")
        print(f"Recall: {score[3]:0.5f}")
        print(f"Precision: {score[4]:0.5f}")
        print(f"Avg. Time taken: {end_time - start_time} seconds")
        stats = [[model_dir, f"{len(test_loader)}", f"{score[0]:0.5f}", f"{score[1]:0.5f}", f"{score[2]:0.5f}", f"{score[3]:0.5f}", f"{score[4]:0.5f}", f"{end_time - start_time}"]]
        data_stats = pd.DataFrame(stats, columns=["Model", "# Images", "Accuracy", "F1", "Jaccard", "Recall", "Precision", "Time"])

        # CSV file path
        csv_file = f"{model_dir}/stats.csv"
        file_exists = os.path.isfile(csv_file)

        # Write or append
        data_stats.to_csv(csv_file, mode='a', header=not file_exists, index=False)
        """ Saving """
        df = pd.DataFrame(SCORE, columns=["Image", "Acc", "F1", "Jaccard", "Recall", "Precision"])
        df.to_csv(f"{model_dir}/score_all.csv")
            
            

    avg_loss = total_loss / len(test_loader)

    print(f"\n🔍 Test Segmentation Loss (Dice_loss): {avg_loss:.6f}")
    #print(f"\n🔍 Test Accuracy : {avg_acc:.6f}")

    # --- visualize_reconstructions (8 recontstructions) ---
    #visualize_segmentations(model, test_loader, device, results_dir)


def infer_images(model_name, image_paths, threshold: float = 0.5, out_dir: str | None = None, save_features: bool = False, batch_size: int = 1):
    """
    Run segmentation inference with a trained UNet on a set of image file paths.

    model_name: folder name under models/ (e.g. 'm[64]')
    image_paths: list of file paths (strings) on disk to run inference on
    threshold: float between 0 and 1 used to binarize sigmoid outputs
    out_dir: output folder under output/<out_dir>. If None, defaults to model_name
    save_features: if True, save d4 intermediate feature channels per image
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if out_dir is None:
        out_dir = model_name
    results_dir = f"output/{out_dir}"
    os.makedirs(results_dir, exist_ok=True)

    model_dir = f"models/segmentation/{model_name}"
    model_path = f"{model_dir}/{model_name}.pth"

    # Try to read model-specific num_channels from metadata if available
    try:
        import json
        meta_path = os.path.join(model_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf8') as f:
                meta = json.load(f)
                num_channels = meta.get('parameters', {}).get('num_channels', {}).get('default', None)
                if num_channels is None:
                    num_channels = [64, 128, 256, 512, 1024, 512, 256, 128, 64]
        else:
            num_channels = [64, 128, 256, 512, 1024, 512, 256, 128, 64]
    except Exception:
        num_channels = [64, 128, 256, 512, 1024, 512, 256, 128, 64]

    # Load model
    model = UNet(num_channels=[64, 128, 256, 512, 1024, 512, 256, 128, 64]).to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    state = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state)
    model.eval()

    # Compose PIL-based transform to ensure tensors are floats in [0,1]
    pil_transform = T.Compose([
        T.Resize((384, 576)),
        T.ToTensor(),
    ])

    # No preprocessing is applied here — use raw model outputs (grayscale masks)

    outputs_saved = []

    with torch.no_grad():
        # Process in batches
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            tensors = []
            valid_paths = []

            # Load and preprocess each image in the batch using PIL + ToTensor
            for img_path in batch_paths:
                try:
                    # PIL open + convert ensures we have an RGB image
                    pil_img = Image.open(img_path).convert("RGB")
                    tensor = pil_transform(pil_img)  # C,H,W float32 in [0,1]

                except Exception as e:
                    print(f"Skipping {img_path}: failed to open with PIL ({e})")
                    continue
                tensors.append(tensor)
                valid_paths.append(img_path)

            if len(tensors) == 0:
                continue

            inp = torch.stack(tensors, dim=0).to(device)
            # Debug: report shape of batch
            try:
                print(f"infer_images: batch input shape = {tuple(inp.shape)} (B,C,H,W)")
            except Exception:
                pass

            # run forward with features
            outputs, d4_feats = model(inp, return_features=True)
            # Debug: output shapes
            try:
                print(f"infer_images: model outputs shape = {tuple(outputs.shape)} (B,1,H,W) - d4_feats shape = {tuple(d4_feats.shape) if d4_feats is not None else 'None'}")
            except Exception:
                pass
            outputs = torch.sigmoid(outputs)
            # outputs shape [B, 1, H, W] — threshold and save per sample
            for i in range(outputs.shape[0]):
                out_single = outputs[i]
                print(out_single)
                
                # Apply threshold argument to binarize prediction
                mask = (out_single).squeeze().cpu().numpy()  # shape [1,H,W]
                img_path = valid_paths[i]

                base_name = os.path.splitext(os.path.basename(img_path))[0]
                mask_save_path = os.path.join(results_dir, f"{base_name}_mask.png")

                # Save raw grayscale mask (0/255) using matplotlib to ensure consistent display
                try:
                    # mask is 0/1 int; convert to 0-255 uint8
                    # Squeeze channel dim and scale to 0-255 for saving
                    #mask_uint8 = (np.squeeze(mask, axis=0) * 255).astype('uint8')
                    plt.imsave(mask_save_path, mask, cmap='gray')
                except Exception:
                    # fallback to cv2 if matplotlib save fails
                    cv2.imwrite(mask_save_path, (mask * 255).astype('uint8'))

                outputs_saved.append(os.path.abspath(mask_save_path))

                # optionally save features
                if save_features:
                    feat_dir = os.path.join(results_dir, f"{base_name}_d4_layer")
                    os.makedirs(feat_dir, exist_ok=True)
                    # d4_feats shape [B, C, H, W]
                    d4_np = d4_feats[i].cpu().numpy()
                    for ch in range(d4_np.shape[0]):
                        feat_map = d4_np[ch]
                        feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)
                        feat_map = (feat_map * 255).astype('uint8')
                        feat_out = os.path.join(feat_dir, f"channel_{ch}.png")
                        cv2.imwrite(feat_out, feat_map)
                    print(f"Saved features for {base_name} -> {feat_dir}")

                print(f"Saved mask for {base_name} -> {mask_save_path}")

    return outputs_saved

def save_predictions(model, dataloader, device, results_dir):
    """
    Save reconstructed outputs from the autoencoder.
    """
    os.makedirs(results_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for imgs, _, names in dataloader:
            imgs = imgs.to(device)
            outputs, _ = model(imgs, return_features=True)  # forward pass

            # Move to CPU
            outputs = outputs.cpu()

            for i in range(outputs.size(0)):
                out_img = outputs[i].squeeze(0).numpy()  # shape [H, W]
                out_img = (out_img * 255).astype("uint8")

                img_name = names[i] if isinstance(names[i], str) else f"img_{i}.png"
                save_path = os.path.join(results_dir, f"{os.path.splitext(img_name)[0]}_recon.png")
                cv2.imwrite(save_path, out_img)

def save_intermediate_channels(d4_feats, names, results_dir):
    """
    Save intermediate feature maps from the autoencoder.
    Expects d4_feats shape: [B, C, H, W]
    """
    d4_np = d4_feats.cpu().numpy()

    for i in range(d4_np.shape[0]):
        img_name = names[i] if isinstance(names[i], str) else f"img_{i}"
        feature_dir = os.path.join(results_dir, f"{img_name}_d4_layer")
        os.makedirs(feature_dir, exist_ok=True)

        for ch in range(d4_np.shape[1]):  # iterate channels
            feat_map = d4_np[i, ch]

            # Normalize each channel separately to [0,255]
            feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)
            feat_map = (feat_map * 255).astype("uint8")

            save_path = os.path.join(feature_dir, f"channel_{ch}.png")
            cv2.imwrite(save_path, feat_map)

def eval_autoEncoder(model_name, num_channels = [64, 128, 256, 512, 1024, 512, 256, 128, 64], dataset_path = "Data_source/Augmented/Data", out_dir = None, test_dataset = True, shuffle = True, rgb = False):
    # --- Config ---
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    model_dir = f'models/{model_name}'
    model_path = f'{model_dir}/{model_name}.pth'
    dataset_path = dataset_path
    if out_dir is None:
        out_dir = model_name
    results_dir = f"output/{out_dir}"

    if(not os.path.exists(results_dir) and rgb is not None):
        os.makedirs(results_dir)

    if(not os.path.exists(model_dir)):
        os.makedirs(model_dir)
        
    # --- Load test data ---
    if(test_dataset is True):
        _, test_loader = load_idrid_grayscale_aug(dataset_path, batch_size=batch_size, shuffle=shuffle)
    elif (test_dataset is False):
       test_loader, _ = load_idrid_grayscale_aug(dataset_path, batch_size=batch_size, shuffle=shuffle)
    else:
        test_loader = load_idrid_grayscale_aug(dataset_path, batch_size=batch_size, shuffle=shuffle)

    # --- Load model ---
    model = AutoEncoder(num_channels=num_channels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.eval()

    # --- Define loss function ---
    criterion = nn.MSELoss()

    # --- Evaluate ---
    total_loss = 0.0

    with torch.no_grad():
        start_time = timer()
        for imgs, masks, names in tqdm(test_loader):
            imgs = imgs.to(device)
            masks = masks.to(device)
            outputs, d4_feats = model(imgs, return_features = True)

            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            # --- Save predictions ---
            save_predictions(model, [(imgs.cpu(), masks.cpu(), names)], device, os.path.join(results_dir, "recons"))

            # --- Save intermediate channels ---
            save_intermediate_channels(d4_feats, names, os.path.join(results_dir, "features"))

            
            
            
    avg_loss = total_loss / len(test_loader)

    print(f"\n🔍 Test Loss (MSE): {avg_loss:.6f}")
    #print(f"\n🔍 Test Accuracy : {avg_acc:.6f}")

    # --- visualize_reconstructions (8 recontstructions) ---
    visualize_reconstructions(model, test_loader, device, results_dir, avg_loss)

def eval_RFMiD(model_name, num_channels = [64, 128, 256, 512, 1024, 512, 256, 128, 64], dataset_path = "Data_source/RFMiD", out_dir = None, test_dataset = True, shuffle = True, rgb = False):
    # --- Config ---
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    model_dir = f'models/{model_name}'
    model_path = f'{model_dir}/{model_name}.pth'
    dataset_path = dataset_path
    if out_dir is None:
        out_dir = model_name
    results_dir = f"output/{out_dir}"

    if(not os.path.exists(results_dir) and rgb is not None):
        os.makedirs(results_dir)

    if(not os.path.exists(model_dir)):
        os.makedirs(model_dir)
        
    # --- Load test data ---
    test_loader = load_rfmid(dataset_path, batch_size=batch_size, shuffle=shuffle)

    # --- Load model ---
    model = AutoEncoder(num_channels=num_channels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.eval()

    # --- Define loss function ---
    criterion = nn.MSELoss()

    # --- Evaluate ---
    total_loss = 0.0

    with torch.no_grad():
        start_time = timer()
        for imgs, masks, names in tqdm(test_loader):
            imgs = imgs.to(device)
            masks = masks.to(device)
            outputs = model(imgs)

            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            # --- Save predictions ---
            #save_predictions(model, [(imgs.cpu(), masks.cpu(), names)], device, os.path.join(results_dir, "recons"))

            # --- Save intermediate channels ---
            #save_intermediate_channels(d4_feats, names, os.path.join(results_dir, "features"))

            
            
            
    avg_loss = total_loss / len(test_loader)

    print(f"\n🔍 Test Loss (MSE): {avg_loss:.6f}")
    #print(f"\n🔍 Test Accuracy : {avg_acc:.6f}")

    # --- visualize_reconstructions (8 recontstructions) ---
    visualize_reconstructions(model, test_loader, device, results_dir, avg_loss)

def eval_RFMiD_color(model_name, num_channels = [64, 128, 256, 512, 1024, 512, 256, 128, 64], dataset_path = "Data_source/RFMiD", out_dir = None, test_dataset = True, shuffle = True, rgb = False):
    # --- Config ---
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    model_dir = f'models/AE2/{model_name}'
    model_path = f'{model_dir}/{model_name}.pth'
    dataset_path = dataset_path
    if out_dir is None:
        out_dir = model_name
    results_dir = f"output/{out_dir}"

    if(not os.path.exists(results_dir) and rgb is not None):
        os.makedirs(results_dir)

    if(not os.path.exists(model_dir)):
        os.makedirs(model_dir)
        
    # --- Load test data ---
    test_loader = load_rfmid_color(dataset_path, batch_size=batch_size, shuffle=shuffle)

    # --- Load model ---
    model = AutoEncoder_RFMiD(num_channels=num_channels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.eval()

    # --- Define loss function ---
    criterion = nn.MSELoss()

    # --- Evaluate ---
    total_loss = 0.0
    with torch.no_grad():
        start_time = timer()
        for imgs, masks, names in tqdm(test_loader):
            imgs = imgs.to(device)
            masks = masks.to(device)
            outputs = model(imgs)

            loss = criterion(outputs, masks)
            total_loss += loss.item()


            
            # --- Save predictions ---
            #save_predictions(model, [(imgs.cpu(), masks.cpu(), names)], device, os.path.join(results_dir, "recons"))

            # --- Save intermediate channels ---
            #save_intermediate_channels(d4_feats, names, os.path.join(results_dir, "features"))

            
            
            
    avg_loss = total_loss / len(test_loader)

    print(f"\n🔍 Test Loss (MSE): {avg_loss:.6f}")
    #print(f"\n🔍 Test Accuracy : {avg_acc:.6f}")

    # --- visualize_reconstructions (8 recontstructions) ---
    visualize_reconstructions_color(model, test_loader, device, results_dir, avg_loss)

def eval_RFMiD_color_2(
    model_name,
    num_channels=[64, 128, 256, 512, 1024, 512, 256, 128, 64],
    dataset_path="Data_source/RFMiD_color",
    out_dir=None,
    test_dataset=True,
    shuffle=True,
    rgb=False
):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    model_dir = f"models/AE3/{model_name}"
    model_path = f"{model_dir}/{model_name}.pth"

    if out_dir is None:
        out_dir = model_name
    results_dir = f"output/{out_dir}"

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # --- Load data ---
    test_loader = load_rfmid_color(dataset_path, batch_size=batch_size, shuffle=shuffle)

    # --- Load model ---
    model = AutoEncoder_RFMiD(num_channels=num_channels).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # --- Define losses/metrics ---
    criterion = nn.MSELoss(reduction="mean")
    lpips_loss_fn = lpips.LPIPS(net="vgg").to(device)

    total_mse, total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0, 0.0
    n_batches = len(test_loader)

    with torch.no_grad():
        start_time = timer()
        for imgs, masks, names in tqdm(test_loader, desc="Evaluating"):
            imgs = imgs.to(device)
            masks = masks.to(device)
            outputs = model(imgs)

            # --- Ensure range [0, 1] ---
            outputs = torch.clamp(outputs, 0, 1)
            masks = torch.clamp(masks, 0, 1)

            # --- Compute metrics ---
            mse_loss = criterion(outputs, masks)
            psnr_val = psnr(outputs, masks)
            ssim_val = ssim(outputs, masks)
            lpips_val = lpips_loss_fn(outputs, masks).mean()  # LPIPS is a distance (lower is better)

            total_mse += mse_loss.item()
            total_psnr += float(psnr_val)
            total_ssim += float(ssim_val)
            total_lpips += lpips_val.item()

    # --- Averages ---
    avg_mse = total_mse / n_batches
    avg_psnr = total_psnr / n_batches
    avg_ssim = total_ssim / n_batches
    avg_lpips = total_lpips / n_batches

    print("\n🔍 Evaluation Results:")
    print(f"   MSE Loss      : {avg_mse:.6f}")
    print(f"   PSNR (dB)     : {avg_psnr:.4f}")
    print(f"   SSIM          : {avg_ssim:.4f}")
    print(f"   LPIPS (↓)     : {avg_lpips:.4f}")
    print(f"   Similarity ≈  {avg_ssim * 100:.2f}%")

    print(f"⏱️  Evaluation completed in {(timer() - start_time):.2f} seconds.")
    # --- visualize_reconstructions (8 recontstructions) ---
    evaluation_result = f"MSE: {avg_mse:.6f} || PSNR: {avg_psnr:.4f} || SSIM: {avg_ssim:.4f} || LPIPS (↓): {avg_lpips:.4f}"
    visualize_reconstructions_color(model, test_loader, device, results_dir, evaluation_result)


def eval_flower102_color(model_name, num_channels = [64, 128, 256, 512, 1024, 512, 256, 128, 64], out_dir = None, rgb = False):
    # --- Config ---
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    model_dir = f'models/{model_name}'
    model_path = f'{model_dir}/{model_name}.pth'
    if out_dir is None:
        out_dir = model_name
    results_dir = f"output/{out_dir}"

    if(not os.path.exists(results_dir) and rgb is not None):
        os.makedirs(results_dir)

    if(not os.path.exists(model_dir)):
        os.makedirs(model_dir)
        
    # --- Load test data ---
    test_loader = load_flowers102(batch_size=batch_size)

    # --- Load model ---
    model = AutoEncoder(num_channels=num_channels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.eval()

    # --- Define loss function ---
    criterion = nn.MSELoss()

    # --- Evaluate ---
    total_loss = 0.0
    with torch.no_grad():
        start_time = timer()
        for imgs, _ in tqdm(test_loader):
            imgs = imgs.to(device)
            outputs = model(imgs)

            loss = criterion(outputs, imgs)
            total_loss += loss.item()
            
            # --- Save predictions ---
            #save_predictions(model, [(imgs.cpu(), masks.cpu(), names)], device, os.path.join(results_dir, "recons"))

            # --- Save intermediate channels ---
            #save_intermediate_channels(d4_feats, names, os.path.join(results_dir, "features"))

            
            
            
    avg_loss = total_loss / len(test_loader)

    print(f"\n🔍 Test Loss (MSE): {avg_loss:.6f}")
    #print(f"\n🔍 Test Accuracy : {avg_acc:.6f}")

    # --- visualize_reconstructions (8 recontstructions) ---
    visualize_reconstructions_flower102(model, test_loader, device, results_dir, avg_loss)


def eval_vae(num_samples = 16):
    output_visualizations_directory = "results_VAE"
    set_seed(42)
    if not os.path.exists(output_visualizations_directory):
        os.makedirs(output_visualizations_directory)

    # --- Config ---
    model_path = 'models/VAE/vae_e6_d6_w192_h128_rfmid_sk[d6]_color_latent_conv_beta[0]/vae_e6_d6_w192_h128_rfmid_sk[d6]_color_latent_conv_beta[0].pth'

    # --- Load model ---
    model = LadderVAE()
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()

    samples = model.sample(num_samples).cpu()
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flatten()):
        img = samples[i].permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_visualizations_directory}/vae_3_alpha_1_seeded(42).png')
    plt.show()