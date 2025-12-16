import os
from PIL import Image
from eval import infer_images

os.makedirs('uploads/demo', exist_ok=True)
img_path = os.path.join('uploads', 'demo', 'demo_image.png')
Image.open(r"D:\Projects\Fundal_Images_Analysis\fia_classification_pytorch\Data_source\rfmid\images\testing\5.png").convert('RGB')
Image.new('RGB', (192, 128)).save(img_path)
impath = r"D:\Projects\Fundal_Images_Analysis\fia_classification_pytorch\Data_source\rfmid\images\testing\5.png"
print('Created input image:', img_path)
outs = infer_images('ae_e6_d6_w192_h128_rfmid_sk[d6]_color_[LATEST]_sig', [impath], out_dir='demo_results', batch_size=1)
print('Outputs written:')
for p in outs:
    print('-', p)
    print('exists?', os.path.exists(p))
