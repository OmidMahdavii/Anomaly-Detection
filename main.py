import os
import logging
from parse_args import parse_arguments
from load_data import build_splits
from experiments.autoencoder import AutoencoderExperiment
from experiments.gan import GANExperiment

WINDOW_SIZE = 10
THRESHOLD = 0.1
REG_WEIGHT = 1.0


def main(opt):
    experiment = AutoencoderExperiment() if opt['experiment'] == 'autoencoder' else GANExperiment()

    opt['window_size'] = WINDOW_SIZE
    opt['threshold'] = THRESHOLD
    opt['reg_weight'] = REG_WEIGHT
    
    train_loader, validation_loader, test_loader = build_splits(opt)
    

    if opt['test']: # Test the model if '--test' flag is set
        experiment.load_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth')
        test_accuracy, _ = experiment.validate(test_loader)
        logging.info(f'[TEST] Accuracy: {(100 * test_accuracy):.2f}')
    
    else: # Skip training if '--test' flag is set
        iteration = 0

        # Restore last checkpoint
        if os.path.exists(f'{opt["output_path"]}/last_checkpoint.pth'):
            iteration, train_loss = experiment.load_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth')
        else:
            logging.info(opt)

        # Train loop
        while iteration < opt['max_iterations']:
            train_loss = 0
            
            for data in train_loader:
                train_loss += experiment.train_iteration(data)

            if iteration % opt['print_every'] == 0:
                logging.info(f'[TRAIN - {iteration}] Loss: {train_loss}')
                experiment.save_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth', iteration, train_loss)

            iteration += 1

        # Run validation after the training
        val_accuracy, val_loss = experiment.validate(validation_loader)
        logging.info(f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}')        


if __name__ == '__main__':

    opt = parse_arguments()

    # Setup output directories
    os.makedirs(opt['output_path'], exist_ok=True)

    # Setup logger
    logging.basicConfig(filename=f'{opt["output_path"]}/log.txt', format='%(message)s', level=logging.INFO, filemode='a')

    main(opt)