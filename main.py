import os
import logging
from parse_args import parse_arguments
from load_data import build_splits
from experiments.autoencoder import AutoencoderExperiment
from experiments.gan import GANExperiment

WINDOW_SIZE = 50
LATENT_SIZE = 8
THRESHOLD = 0.1
REG_WEIGHT = 1.0


def main(opt):
    opt['window_size'] = WINDOW_SIZE
    opt['latent_size'] = LATENT_SIZE
    opt['threshold'] = THRESHOLD
    opt['reg_weight'] = REG_WEIGHT
    
    experiment = AutoencoderExperiment(opt) if opt['experiment'] == 'autoencoder' else GANExperiment(opt)
    
    train_loader, validation_loader, test_loader = build_splits(opt)
    
    if opt['test']: # Test the model if '--test' flag is set
        experiment.load_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth')
        recall = experiment.test(test_loader, opt['threshold'])
        logging.info(f'[TEST] Recall: {(100 * recall):.2f}')
    
    else: # Skip training if '--test' flag is set
        iteration = 0
        # Restore last checkpoint
        if os.path.exists(f'{opt["output_path"]}/last_checkpoint.pth'):
            iteration = experiment.load_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth')
        else:
            logging.info(opt)
        # Train loop
        while iteration < opt['max_iterations']:
            train_loss = 0
            
            for data in train_loader:
                train_loss += experiment.train_iteration(data)

            if iteration % opt['print_every'] == 0:
                logging.info(f'[TRAIN - {iteration}] Loss: {train_loss}')
                # experiment.save_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth', iteration, train_loss)

            if iteration % opt['validate_every'] == 0:
                    # Run validation
                    auc, val_loss = experiment.validate(validation_loader)
                    logging.info(f'[VAL - {iteration}] Loss: {val_loss} | AUC: {(100 * auc):.2f}')
                    experiment.save_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth', iteration)

            iteration += 1  


if __name__ == '__main__':

    opt = parse_arguments()

    # Setup output directories
    os.makedirs(opt['output_path'], exist_ok=True)

    # Setup logger
    logging.basicConfig(filename=f'{opt["output_path"]}/log.txt', format='%(message)s', level=logging.INFO, filemode='a')

    main(opt)