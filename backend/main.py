from train import train_Unet, train_AutoEncoder, train_AutoEncoder_RFMiD, train_AutoEncoder_RFMiD_color, train_AutoEncoder_flowers102, train_Ladder_VAE, train_VAE_RFMiD_color
from eval import eval, eval_autoEncoder, eval_RFMiD, eval_RFMiD_color, eval_flower102_color, eval_vae, eval_RFMiD_color_2

if __name__ == "__main__":
    #Initialize dataset paths
    augmented_data = 'Data_source/Augmented'
    unaugmented_data = 'Data_source/unaug_resized'

    #Initialize model name
    # Naming conventions. 
    # m - specifies conventional unet model
    # [64] - represents 64 channels in both e1, d4 layers. 
    #model_name = "vae_e6_d6_w192_h128_rfmid_sk[d6]_color_latent_conv_beta[0]"
    model_name = "ae_e6_d6_w192_h128_rfmid_sk[]_color_sig"
    #Initialize results directory
    results_dir = f"r2_{model_name}"
    #train_AutoEncoder(model_name=model_name, batch_size=4)

    """ Train RFMID """
    #train_AutoEncoder_RFMiD_color(model_name=model_name, batch_size=64, dataset_path='Data_source/RFMiD_color', epochs=240)
    eval_RFMiD_color_2(model_name=model_name, dataset_path='Data_source/RFMiD_color')

    #training the model
    #train_Unet(model_name=model_name, num_channels=[64, 128, 256, 512, 1024, 512, 256, 128, 26])
    
    #train_AutoEncoder_flowers102(model_name=model_name, batch_size=64)
    #eval_flower102_color(model_name=model_name)
    

    #evaluation
    #for i in [True, False, None]:
    #    eval(model_name = model_name, num_channels=[64, 128, 256, 512, 1024, 512, 256, 128, 64], dataset_path = unaugmented_data, out_dir = results_dir, test_dataset = i, shuffle = True, rgb=False)

    #eval_autoEncoder(model_name=model_name, dataset_path='Data_source/idrid_grayscale_aug', out_dir=results_dir, test_dataset=False)

    #eval_RFMiD_color(model_name=model_name, num_channels=[64, 128, 256, 512, 1024, 512, 256, 128, 64], dataset_path='Data_source/RFMiD_color')

    """ Training Ladder VAE """
    #train_Ladder_VAE(model_name=model_name, batch_size=32)
    #eval_vae()

    """ Train VAE """
    #train_VAE_RFMiD_color(model_name=model_name, batch_size=64, patience=10)