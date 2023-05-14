import os
import logging
from parse_args import parse_arguments
from load_data import build_splits
from experiments.autoencoder import AutoencoderExperiment
from experiments.gan import GANExperiment


def main(opt):
    experiment = AutoencoderExperiment()
    # experiment = GANExperiment()
    
    train_loader, validation_loader, test_loader = build_splits(opt)

    if not opt['test']: # Skip training if '--test' flag is set
        iteration = 0
        total_train_loss = 0

        # Restore last checkpoint
        if os.path.exists(f'{opt["output_path"]}/last_checkpoint.pth'):
            iteration, total_train_loss = load_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth')
        else:
            logging.info(opt)

        # Train loop
        while iteration < opt['max_iterations']:
            for data in train_loader:

                total_train_loss += experiment.train_iteration(data)

                if iteration % opt['print_every'] == 0:
                    logging.info(f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')
                    experiment.save_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth', iteration, total_train_loss)
                
                # if iteration % opt['validate_every'] == 0:
                #     # Run validation
                #     val_accuracy, val_loss = experiment.validate(validation_loader)
                #     logging.info(f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}')
                #     if val_accuracy > best_accuracy:
                #         best_accuracy = val_accuracy
                #         experiment.save_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth', iteration, best_accuracy, total_train_loss)
                #     experiment.save_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth', iteration, best_accuracy, total_train_loss)

                iteration += 1
                if iteration > opt['max_iterations']:
                    break

    # Test
    opt['test'] = True
    experiment.load_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth')
    test_accuracy, _ = experiment.validate(test_loader)
    logging.info(f'[TEST] Accuracy: {(100 * test_accuracy):.2f}')


if __name__ == '__main__':

    opt = parse_arguments()

    # Setup output directories
    os.makedirs(opt['output_path'], exist_ok=True)

    # Setup logger
    logging.basicConfig(filename=f'{opt["output_path"]}/log.txt', format='%(message)s', level=logging.INFO, filemode='a')

    main(opt)