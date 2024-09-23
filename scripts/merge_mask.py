import os
from tqdm import tqdm
from PIL import Image
import fire


def merge_images(img_folder, mask_folder, res_folder):
    # Ensure the result folder exists
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    # Iterate over images in the image folder
    for img_name in tqdm(os.listdir(img_folder)):
        img_path = os.path.join(img_folder, img_name)
        mask_path = os.path.join(mask_folder, img_name)

        # Check if corresponding mask exists
        if os.path.exists(mask_path):
            result_path = os.path.join(res_folder, img_name)
            if os.path.exists(result_path):
                continue
            with Image.open(img_path) as img:
                with Image.open(mask_path) as mask:
                    # Convert mask to 'L' mode (grayscale) if it's not
                    if mask.mode != "L":
                        mask = mask.convert("L")

                    # Combine image and mask into an RGBA image
                    img.putalpha(mask)
                    img.save(result_path, format="PNG")
        else:
            print(f"No corresponding mask found for {img_name}")


if __name__ == "__main__":
    fire.Fire(merge_images)
