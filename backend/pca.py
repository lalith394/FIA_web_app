import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import glob
import re
import io
from natsort import natsorted  # Ensures proper numerical sorting
from sklearn.decomposition import PCA

def create_dir(path):
    os.makedirs(path, exist_ok=True)

def read_images(folder_path, data, image_size):
    """Reads images, converts them to grayscale, and resizes them."""
    filenames = natsorted(os.listdir(folder_path))
    if not filenames:
        print("No filenames")
        return None, None, None
    match = re.match(r"(.+?)_channel", filenames[0])
    if not match:
        print("No match")
        return None, None, None

    img_name = match.group(1)
    ori_image_path = os.path.join(f"{data}/image", f"{img_name}.png")
    ori_image = cv2.imread(ori_image_path, cv2.IMREAD_UNCHANGED)
    if ori_image is None:
        print("No original image")
        return None, None, None
    ori_resized = cv2.resize(ori_image, image_size)
    
    images = [cv2.resize(cv2.imread(os.path.join(folder_path, f), cv2.IMREAD_GRAYSCALE), image_size)
              for f in filenames if cv2.imread(os.path.join(folder_path, f), cv2.IMREAD_GRAYSCALE) is not None]
    
    return np.array(images, dtype=np.float32), ori_resized, img_name

def perform_pca(folder_path, num_components, data, img_size):
    """Performs PCA using eigen decomposition."""
    images, original, img_name = read_images(folder_path, data, img_size)
    if(original == None):
        return
    num_images, num_pixels = images.shape[0], img_size[0] * img_size[1]
    data_matrix = images.reshape(num_images, num_pixels)
    print(data_matrix.shape)
    mean_face = np.mean(data_matrix, axis=0)
    centered_matrix = data_matrix - mean_face
    
    covariance_matrix = np.dot(centered_matrix, centered_matrix.T) / num_images
    eigenvalues, eigenvectors_small = np.linalg.eigh(covariance_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors_small = eigenvalues[idx], eigenvectors_small[:, idx]
    
    eigenvectors = np.dot(centered_matrix.T, eigenvectors_small)
    eigenvectors /= np.linalg.norm(eigenvectors, axis=0, keepdims=True)
    
    eigenfaces = [eigenvectors[:, i].reshape(img_size[::-1]) for i in range(num_components)]
    
    return mean_face.reshape(img_size[::-1]), eigenfaces, eigenvalues[:num_components], eigenvectors[:, :num_components]

def save_pca_res(eigen_faces, ori_img, mean_img, save_path, index, img_size):

    fig, axes = plt.subplots(3, 4, figsize=(14, 8))
    left = 0.017
    right = 1 - left
    bottom = 0.05
    top = 1 - bottom
    wspace =  0.01
    hspace = 0.2
    plt.subplots_adjust(left=0.017, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
    
    axes[0,0].imshow(ori_img)
    axes[0,0].axis("off")
    axes[0,0].set_title("Original Image", size = 14)

    axes[0,1].imshow(mean_img.reshape(img_size[::-1]), cmap = "gray")
    axes[0,1].axis("off")
    axes[0,1].set_title("Mean Image", size = 14)
    
    c = 0
    for i in range(3):
        j = 0
        if i == 0:
            j = 2
        while j<4:
            axes[i, j].imshow(eigen_faces[c+index].reshape(img_size[::-1]), cmap="gray")
            axes[i, j].axis("off")
            axes[i, j].set_title(f"Eigenface {c+1}", size=14)
            c += 1
            j += 1
    
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

def save_plot_eigen_vals(path, eigen_values):
    eigen_values = eigen_values[1:]
    xvals = np.arange(1,len(eigen_values)+1)
    plt.figure(figsize=(12, 8))
    plt.scatter(xvals, eigen_values, color = "green", label = 'Eigen Values')
    plt.xlabel("Channel Number", fontsize=18)
    plt.ylabel("Eigenvalues", fontsize=18)
    plt.grid(True, linestyle="--", alpha=0.6)

    num_xticks = min(len(xvals), 7)  # Adjusted for better clarity
    xtick_positions = np.linspace(xvals[0], xvals[-1], num_xticks, dtype=int)
    plt.xticks(xtick_positions, fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(loc="upper right", fontsize=16)  # Legend at top right
    
    plt.savefig(path)
    plt.close()

def resize(image):
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

def add_positional_encoding(features, row_num_encoding, column_num_encoding):
    row_encoding = np.expand_dims(row_num_encoding, axis=0)
    col_encoding = np.expand_dims(column_num_encoding, axis=0)
    return np.concatenate([features, row_encoding, col_encoding], axis=0)

def perform_pca_optimized(folder_path, num_components, data, img_size, extract_faces = False):
    """Performs PCA using scikit-learn for efficiency and stability."""
    images, originals, img_names = read_images(folder_path, data, img_size)
    """ Encoding """
    feature_width = img_size[0]
    feature_height = img_size[1]
    column_num_encoding = np.tile(np.linspace(1,feature_width, feature_width), (feature_height, 1))
    row_num_encoding = np.transpose(np.tile(np.linspace(1,feature_height, feature_height), (feature_width, 1)))

    images = add_positional_encoding(images, row_num_encoding, column_num_encoding)

    if(originals is None):
        return None, None, None, None 
    num_images, num_pixels = images.shape[0], img_size[0] * img_size[1]
   
    data_matrix = images.reshape(num_images, num_pixels).astype(np.float32)
   
    
    # Perform PCA using scikit-learn
    pca = PCA(n_components=num_components, svd_solver='randomized', whiten=False, copy=False)
    pca.fit(data_matrix)
    mean_face = eigenfaces = eigenvectors = 0
    if extract_faces:
        mean_face = pca.mean_.reshape(img_size[::-1])
        eigenfaces = [component.reshape(img_size[::-1]) for component in pca.components_]
        eigenvectors = pca.components_.T  # shape: (num_pixels, num_components)
    eigenvalues = pca.explained_variance_
    return mean_face, eigenfaces, eigenvalues, eigenvectors

def pca(input, output, data):
    for folder in os.listdir(input):
        folder_path = os.path.join(input, folder)
        if os.path.isdir(folder_path):
            count = len(glob.glob(os.path.join(folder_path, '*')))
            sample = glob.glob(os.path.join(folder_path, '*'))[0]
            sample_img = cv2.imread(sample, cv2.IMREAD_UNCHANGED)
            img_size = resize(sample_img)
            img_size = img_size[::-1]

            match = re.match(r'^(.*?)_hidden_layer', folder)
            img_name= ''
            if match:
                img_name = match.group(1)
            print(img_name, img_size)
            perform_pca_optimized(folder_path, count, data, img_size)

            original_img_path = os.path.join(f"{data}/image", f"{img_name}.png")
            if not os.path.exists(original_img_path):
                continue
            original_img = cv2.imread(original_img_path, cv2.IMREAD_UNCHANGED)[:,:,::-1] #BGR to RGB
            

            #top 10 ef
            img_output_path = os.path.join(output, f"{img_name}")
            if not os.path.exists(img_output_path):    
                os.makedirs(img_output_path)
            top_10_ef_path = os.path.join(img_output_path, f"{img_name}_top_10_eigenfaces.png")
            last_10_ef_path = os.path.join(img_output_path, f"{img_name}_last_10_eigenfaces.png")
            plot_path = os.path.join(img_output_path, f"{img_name}_ev_plot.png")
            eigen_vals_path = os.path.join(img_output_path, f"{img_name}_ev.csv")
            if(os.path.exists(eigen_vals_path)):
                continue
            m, ef, ev, vect = perform_pca_optimized(folder_path, count, data, img_size, True)
            if(m is None):
                continue
            save_pca_res(ef, original_img, m, top_10_ef_path, 0, img_size)
            save_pca_res(ef, original_img, m, last_10_ef_path, (count-10), img_size)
            save_plot_eigen_vals(plot_path, ev)
            np.savetxt(eigen_vals_path, ev, delimiter=",")

def pca2(input, output, data, k):
    for folder in os.listdir(input):
        folder_path = os.path.join(input, folder)
        if os.path.isdir(folder_path):
            count = len(glob.glob(os.path.join(folder_path, '*')))
            sample = glob.glob(os.path.join(folder_path, '*'))[0]
            sample_img = cv2.imread(sample, cv2.IMREAD_UNCHANGED)
            img_size = resize(sample_img)
            img_size = img_size[::-1]

            match = re.match(r'^(.*?)_hidden_layer', folder)
            img_name= ''
            if match:
                img_name = match.group(1)
            print(img_name)     
            
            image_folder = os.path.join(output,img_name)
            if(os.path.exists(image_folder)):
                continue
            
            m, ef, ev, vect = perform_pca_optimized(folder_path, count, data, img_size, extract_faces=True)
            if(m is None):
                continue
            print(np.shape(ef))
            create_dir(image_folder)
            print(f"Saving {k} principal components to {image_folder} as .npy file\n")
            for i in range(k):
                np.save(os.path.join(image_folder, f"eigenface_{i+1:02d}.npy"), ef[i])

if __name__ == "__main__":
    all_layers = {'d4': 'results_d4', 
                'd3': 'results_d3', 
                'd2': 'results_d2', 
                'd1': 'results_d1', 
                'b1': 'results_b1', 
                's4': 'results_s4', 
                's3': 'results_s3', 
                's2': 'results_s2', 
                's1': 'results_s1'}


    for keys in all_layers:
        print(f"-----PERFORMING PCA FOR {keys} -------------")  
        pca(
            f'output/r1_rgb_m[64]',
            f'results/pca_results_check2/pca_results_{keys}',
            'Data_source/Raw Dataset/hrf_dataset'
        )
        
        print(f"-----COMPLETED PCA FOR {keys} -------------")
        break 
