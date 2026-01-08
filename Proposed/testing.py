import os
import torch
import warnings
import numpy as np
import __dataset__
import pandas as pd
import matplotlib.pyplot as plt
# import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from network import Network
from __parser__ import args_m2
import training
from training import load_model, SSIMLoss
from wandb_logger import get_wandb_logger
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def testing(args, wandb_logger=None):
    test_dataloader = __dataset__.load_dataset(args.samples, args.batch, 2, pin_memory=True, num_workers=1)
    count = 0
    # OCR model for Brazilian license plates (relative to Proposed/ directory)
    path_ocr = Path('../saved_models/RodoSol-SR')
    criterion = SSIMLoss(path_ocr)

    # Initialize wandb logger if not provided
    if wandb_logger is None:
        wandb_logger = get_wandb_logger(args, mode='test')

    # Device detection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    if device.type == 'cpu':
        print("⚠️  CUDA not available, testing on CPU (will be slower)\n")

    # Set device_list in training module for load_model function
    training.device_list = [device, device]

    # device_ids = GPU_devices([0, 1])
    model = Network(3, 3).to(device)
    # current_epoch, model, train_loss, val_loss, optimizer, early_stopping = load_model(model, args.model, device_ids)
    _, model, optimizer_G, train_loss_g, val_loss_g, _ = load_model(model, args.model)
    
    char = [[], [], [], []]
    pred = [[], [], []]
    lev = [[], [], []]
    psnr_ssim = [[], []]
    imgs=0
    SSIM_param = {
        'gaussian_weights':True,
        'channel_axis':2,  # Updated from deprecated 'multichannel'
        'win_size':3,  # Smaller window size for small images (min is 3)
        'sigma':1.5,
        'use_sample_covariance':False,
        'data_range':1.0
        }
    
    with torch.no_grad():
        prog_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        
        for i, batch in prog_bar:
            batch['SR'] = model(batch['LR'].to(device))
         
            for item in range(len(batch['HR'])):
                imgLR = np.array(criterion.to_numpy(batch['LR'][item].cpu())).astype('uint8')
                imgHR = np.array(criterion.to_numpy(batch['HR'][item].cpu())).astype('uint8')
                imgSR = np.array(criterion.to_numpy(batch['SR'][item].cpu())).astype('uint8')
                
                char[0].append(batch['plate'][item])
                char[1].append(batch['type'][item])
                char[2].append(batch['layout'][item])
                char[3].append(batch['file'][item])
               
                Image.fromarray(np.array(imgHR).astype('uint8')).save(Path(args.save) / Path('HR_'+char[3][-1]))
                Image.fromarray(np.array(imgLR).astype('uint8')).save(Path(args.save) / Path('LR_'+char[3][-1]))
                Image.fromarray(np.array(imgSR).astype('uint8')).save(Path(args.save) / Path('SR_'+char[3][-1]))

                pred[0].append(criterion.OCR_pred(imgHR)[0])
                pred[1].append(criterion.OCR_pred(imgLR)[0])
                pred[2].append(criterion.OCR_pred(imgSR)[0])
                
                lev[0].append(7- criterion.levenshtein(char[0][-1], pred[0][-1]))
                lev[1].append(7- criterion.levenshtein(char[0][-1], pred[1][-1]))
                lev[2].append(7- criterion.levenshtein(char[0][-1], pred[2][-1]))
                
                psnr_ssim[0].append(peak_signal_noise_ratio(imgHR, imgSR))
                psnr_ssim[1].append(structural_similarity(imgHR, imgSR, **SSIM_param))
        
            count += len(batch['HR']) 
        
        df = pd.DataFrame({
            'Type': char[1],
            'Layout': char[2],
            'GT Plate': char[0],
            'file': char[3],
            
            'HR Prediction': pred[0],
            'Accuracy (HR)': lev[0],
            
            'LR Prediction': pred[1],
            'Accuracy (LR)': lev[1],
            
            'OCR SR Prediction': pred[2],
            'Accuracy (SR)': lev[2],
            
            'PSNR HR/SR': psnr_ssim[0],
            'SSIM HR/SR': psnr_ssim[1]
            })
        
        HR = (df['Accuracy (HR)'].value_counts(normalize=True)*100).sort_index()
        LR = (df['Accuracy (LR)'].value_counts(normalize=True)*100).sort_index()
        SR = (df['Accuracy (SR)'].value_counts(normalize=True)*100).sort_index()

        print(HR, '\n')
        print(LR, '\n')
        print(SR, '\n')
        # df_motor = df[df['Type'] == 'motorcycle']
        df_car = df[df['Type'] == 'car']
        with open(os.path.join(args.save, 'resultsHR.csv'), 'w+', encoding='utf8') as fp:
            HR.to_csv(fp, line_terminator='\n')

        with open(os.path.join(args.save, 'resultsSR.csv'), 'w+', encoding='utf8') as fp:
            SR.to_csv(fp, line_terminator='\n')

        with open(os.path.join(args.save, 'resultsLR.csv'), 'w+', encoding='utf8') as fp:
            LR.to_csv(fp, line_terminator='\n')


        df.loc['Average'] = df.mean(axis=0)
        df = df.round(1)
        with open(os.path.join(args.save, 'eval.csv'), 'w+', encoding='utf8') as fp:
            df.to_csv(fp, index=False, line_terminator='\n')
            fp.write('Total images ' + str(count)+'\n')

        # Log test results to wandb
        if wandb_logger.enabled:
            # Aggregate metrics (exclude 'Average' row for mean calculation)
            df_no_avg = df.iloc[:-1]  # Exclude last row which is 'Average'
            avg_metrics = {
                'test/psnr_mean': df_no_avg['PSNR HR/SR'].mean(),
                'test/ssim_mean': df_no_avg['SSIM HR/SR'].mean(),
                'test/accuracy_hr_mean': df_no_avg['Accuracy (HR)'].mean(),
                'test/accuracy_lr_mean': df_no_avg['Accuracy (LR)'].mean(),
                'test/accuracy_sr_mean': df_no_avg['Accuracy (SR)'].mean(),
                'test/total_images': count,
            }
            wandb_logger.log_metrics(avg_metrics)

            # Accuracy distributions
            acc_dist_metrics = {}
            for i in range(8):
                if i in HR.index:
                    acc_dist_metrics[f'test/accuracy_hr_dist_{i}'] = HR[i]
                if i in LR.index:
                    acc_dist_metrics[f'test/accuracy_lr_dist_{i}'] = LR[i]
                if i in SR.index:
                    acc_dist_metrics[f'test/accuracy_sr_dist_{i}'] = SR[i]
            wandb_logger.log_metrics(acc_dist_metrics)

            # Detailed results table (first 100 rows)
            table_data = []
            for idx in range(min(100, len(df_no_avg))):
                row = df_no_avg.iloc[idx]
                table_data.append([
                    row['file'], row['GT Plate'], row['HR Prediction'],
                    row['Accuracy (HR)'], row['LR Prediction'], row['Accuracy (LR)'],
                    row['OCR SR Prediction'], row['Accuracy (SR)'],
                    row['PSNR HR/SR'], row['SSIM HR/SR']
                ])

            wandb_logger.log_table('test/detailed_results', table_data,
                columns=['File', 'GT Plate', 'HR Pred', 'HR Acc',
                        'LR Pred', 'LR Acc', 'SR Pred', 'SR Acc', 'PSNR', 'SSIM'])

            print("✅ Test results logged to W&B")


def histogram(save_path, df, name, columns_to_drop=['PSNR HR/SR', 'SSIM HR/SR']):
    hist_df = df.select_dtypes(exclude=['object'])
    hist_df = hist_df.drop(columns_to_drop, axis=1)
    hist_df = hist_df.round(1)
    columns = hist_df.columns.values.tolist()
    x_counts = np.arange(0, 8, dtype=int)
    y_counts = np.zeros(8)
    
    for col in columns:  
        for i in range(0, 8):
            count = (hist_df[col] == i).sum()
            
            y_counts[i] = count
            
        y_counts = ((y_counts/df.shape[0])*100).round(4)        
        plt.figure(figsize = (10, 5))
        bars = plt.bar(x_counts, y_counts)
        labels = [f"{i}" for i in y_counts]
        
        for bar, label in zip(bars, labels):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2, height + 0.01, label, ha="center", va="bottom"
                )
        
        plt.title(col)
        plt.ylabel('Percentage (%)')
        plt.yticks(range(0, 110, 10))
        
        plt.xlabel('Total Correct Characters ({})'.format(name))
        
        plt.savefig(os.path.join(save_path, col.replace('/', '').replace(' ', '')+'-'+name+".pdf"), format="pdf", bbox_inches="tight")
        y_counts = np.zeros(8)
    
if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        args = args_m2()
        wandb_logger = get_wandb_logger(args, mode='test')
        testing(args, wandb_logger)
        wandb_logger.finish()
