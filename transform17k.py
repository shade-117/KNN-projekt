import math
import os
import shutil
import glob

orig_path = '/storage/brno2/home/xmojzi08/storage/17k/lsar_depths'
fovs_path = '/storage/brno2/home/xmojzi08/storage/17k/lsar_depths_metadata'

assert os.path.isdir(orig_path), "Not a directory [orig_path]"
depth_pfm = 'distance_crop.pfm'  # depth ground-truth

# # move pfm from pinhole
all_dirs = glob.glob(orig_path + '/*')
all_dirs_len = len(all_dirs)
for j, curr in enumerate(all_dirs):
    if not os.path.isdir(curr):
        continue
    print(f'Processing folder {j} / {all_dirs_len}')
    pin_path = os.path.join(curr, 'pinhole', depth_pfm)
    if os.path.exists(pin_path):
        shutil.move(pin_path, os.path.join(curr, 'pinhole_' + depth_pfm))
    else:
        continue

    pinhole_dir = os.path.join(curr, 'pinhole')
    shutil.rmtree(pinhole_dir)
#
# # move images to their folders
image_files = glob.glob(orig_path + '/*.jpg') + glob.glob(orig_path + '/*.png')
leng_if = len(image_files)
for i, image in enumerate(image_files):
    print(f'Processing folder {i} / {leng_if}')
    img_name = os.path.basename(image)
    wo_extension = os.path.splitext(img_name)[0]
    extension = os.path.splitext(img_name)[1]
    corresponding_dir_path = 'lsar_' + wo_extension
    if not os.path.isdir(corresponding_dir_path):
        continue
    shutil.move(image, os.path.join(orig_path, corresponding_dir_path, 'photo' + extension))

# add FOV
fov_dirs = glob.glob(fovs_path + '/*')
for k, fov_dir in enumerate(fov_dirs):
    if not os.path.isdir(fov_dir):
        continue
    print(f'Processing folder {k} / {len(fov_dirs)}')
    info_path = os.path.join(fov_dir, 'info.txt')
    if os.path.exists(info_path):
        corresponding_dir_path_name = 'lsar_' + os.path.basename(fov_dir)
        corr_dir_real_path = os.path.join(orig_path, corresponding_dir_path_name)
        if not os.path.isdir(corr_dir_real_path):
            continue
        new_info_path = os.path.join(orig_path, corresponding_dir_path_name, 'info.txt')
        with open(info_path, 'r') as f:
            fov_info = f.read().splitlines()
        fov_info = fov_info[0].split(' ')
        radians = float(fov_info[4]) * (math.pi / 180)
        fov_info[4] = str(radians)
        fov_info.append(str(radians))  # add it to 5th line like in geopose3k
        # create new file in corresponding dir and write converted info.txt
        with open(new_info_path, 'w') as f:
            f.write("\n".join(str(item) for item in fov_info))
    else:
        print("Info path does not exist ", info_path)