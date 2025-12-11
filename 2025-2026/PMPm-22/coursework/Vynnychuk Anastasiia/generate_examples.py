import albumentations as A
import cv2 as cv

img_path = "./data/dfg/JPEGImages/0000002.jpg"

augmentations = {
    "RandomRain": A.RandomRain(
        slant_range=(-10, 10),
        drop_length=20,
        drop_width=1,
        drop_color=(200, 200, 200),
        blur_value=3,
        brightness_coefficient=0.9,
        rain_type="drizzle",  # or 'heavy' for more extreme
        p=1.0,
    ),
    "RandomFog": A.RandomFog(fog_coef_range=(0.1, 0.2), alpha_coef=0.1, p=1.0),
    "RandomSnow": A.RandomSnow(
        snow_point_range=(0.1, 0.3),
        brightness_coeff=1.5,
        p=1.0,
    ),
    "RandomSunFlare": A.RandomSunFlare(
        flare_roi=(0, 0, 1, 0.5),
        angle_range=(0, 1),
        num_flare_circles_range=(4, 8),
        src_radius=200,
        p=1.0,
    ),
    "RandomBrightnessContrast": A.RandomBrightnessContrast(
        brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=1.0
    ),
    "RandomToneCurve": A.RandomToneCurve(scale=0.3, p=1.0),
    "RandomGamma": A.RandomGamma(gamma_limit=(70, 130), p=1.0),
    "GaussianBlur": A.GaussianBlur(blur_limit=(3, 7), p=1.0),
    "MotionBlur": A.MotionBlur(blur_limit=7, p=1.0),
    "MedianBlur": A.MedianBlur(blur_limit=5, p=1.0),
    "ImageCompression": A.ImageCompression(quality_range=(60, 90), p=1.0),
    "GaussNoise": A.GaussNoise(std_range=(0.05, 0.15), p=1.0),
    "ISONoise": A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1.0),
    "HueSaturationValue": A.HueSaturationValue(
        hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=1.0
    ),
}

img = cv.imread(img_path, cv.IMREAD_COLOR_RGB)
for k, v in augmentations.items():
    out_file_name = f"./data/augmentations-examples/{k}.jpg"
    augmented = v(image=img)
    cv.imwrite(out_file_name, cv.cvtColor(augmented["image"], cv.COLOR_RGB2BGR))
