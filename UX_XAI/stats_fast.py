import numpy as np
from PIL import Image
from sklearn.metrics import mean_squared_error
from glob import glob
from natsort import natsorted
import os
import csv
from multiprocessing import Pool
from tqdm import tqdm
from numba import jit

@jit(nopython=True)
def compute_dice(gt_binary, pred_binary):
    intersection = np.sum(gt_binary * pred_binary)
    return 2. * intersection / (np.sum(gt_binary) + np.sum(pred_binary))

@jit(nopython=True)
def compute_mse(gt, pred):
    return np.mean((gt - pred) ** 2)

def convert_rgb_to_label(msk_rgb):
    msk_np = np.array(msk_rgb)
    flat_rgb = np.dot(msk_np, [65536, 256, 1])
    unique_colors, flat_labels = np.unique(flat_rgb, return_inverse=True)
    label_image = flat_labels.reshape(msk_np.shape[:2])
    return Image.fromarray(label_image.astype(np.uint16)).resize((512, 512), Image.NEAREST)

def process_model(model):
    mse_model_results = []
    dice_scores = []
    cell_counts = []

    try:
        pred_model_probs = natsorted(glob(f'./results_stats_no_aug/{model}/cell_count_*/*.tif'))
    except Exception as e:
        print(f"Error processing model {model}: {e}")
        return model, 0, mse_model_results, dice_scores

    for gt_path, pred_prob_path in zip(gt_labels, pred_model_probs):
        try:
            gt_lbl = np.array(convert_rgb_to_label(Image.open(gt_path)))
            pred_prob_img = np.array(Image.open(pred_prob_path))
            cell_count = int(pred_prob_path.split('/')[3].split('_')[2])
            cell_counts.append(cell_count)

            gt_binary = (gt_lbl > 0).astype(np.float32)
            pred_binary = (pred_prob_img > 0.5).astype(np.float32)

            mse_model_results.append(compute_mse(gt_binary, pred_prob_img))
            dice_scores.append(compute_dice(gt_binary, pred_binary))
        except Exception as e:
            print(f"Error processing file {gt_path} or {pred_prob_path}: {e}")

    return model, cell_counts, mse_model_results, dice_scores

# def save_results_to_csv(results, filename):
#     with open(filename, 'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Model', 'Cell Count', 'Scores'])

#         for model, data in results.items():
#             for cell_count, scores in data.items():
#                 writer.writerow([model, cell_count, scores])


def save_results_to_csv(results, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'Cell Count', 'Scores'])

        for (model, cell_count), scores in results.items():
            scores_str = ', '.join(map(str, scores))
            writer.writerow([model, cell_count, scores_str])

if __name__ == '__main__':
    models = natsorted(os.listdir('./results_stats_no_aug'))

    gt_labels = []
    for file_name in natsorted(os.listdir('./stats_csv')):
        if file_name.endswith('.csv'):
            file_path = os.path.join('./stats_csv', file_name)
            with open(file_path, mode='r') as file:
                reader = csv.reader(file)
                next(reader)
                for row in reader:
                    mask_path = row[1]
                    gt_labels.append(mask_path)

    mse_results = {model: {} for model in models}
    dice_results = {model: {} for model in models}

    with Pool() as pool:
        results = list(tqdm(pool.imap(process_model, models), total=len(models)))

    aggregated_mse_results = {}
    aggregated_dice_results = {}

    for model, cell_counts, mse_model_results, model_dice_scores in results:
        for i, cell_count in enumerate(cell_counts):
            key = (model, cell_count)
            if key not in aggregated_mse_results:
                aggregated_mse_results[key] = []
                aggregated_dice_results[key] = []
            aggregated_mse_results[key].append(mse_model_results[i])
            aggregated_dice_results[key].append(model_dice_scores[i])

    # Save MSE and Dice results into CSV files
    save_results_to_csv(aggregated_mse_results, 'no_aug_mse_results.csv')
    save_results_to_csv(aggregated_dice_results, 'no_aug_dice_results.csv')
