import os
import logging
from parse_args import parse_arguments
from load_data import build_splits
from experiments.autoencoder import AutoencoderExperiment
from experiments.gan import GANExperiment


def main(opt):  
    experiment = AutoencoderExperiment(opt) if opt['experiment'] == 'autoencoder' else GANExperiment(opt)
    
    train_loader, validation_loader, test_loader = build_splits(opt)
    
    if opt['test']: # Test the model if '--test' flag is set
        experiment.load_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth')
        recall = experiment.evaluate(test_loader, opt['threshold'])
        logging.info(f'[TEST] Recall: {(100 * recall):.2f}')
    
    elif opt['validate']:
        experiment.load_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth')
        f1 = experiment.validate(validation_loader, threshold=opt['threshold'])
        logging.info(f'[VAL] F1 score: {(100 * f1):.2f} | Threshold: {opt["threshold"]}')
    
    else: # Start training only if '--test' flag is not set
        iteration = 0
        bestAP = 0
        # Restore last checkpoint
        if os.path.exists(f'{opt["output_path"]}/best_checkpoint.pth'):
            iteration, bestAP = experiment.load_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth')
        else:
            logging.info(opt)
        # Train loop
        while iteration < opt['max_iterations']:
            train_loss = 0
            count = 0
            
            for data in train_loader:
                train_loss += experiment.train_iteration(data)
                count += data[0].shape[0]

            tot_loss = train_loss/count
            
            if iteration % opt['print_every'] == 0:
                logging.info(f'[TRAIN - {iteration}] Loss: {tot_loss}')

            if iteration % opt['validate_every'] == 0:
                    # Run validation
                    ap, threshold = experiment.validate(validation_loader, threshold=None)
                    logging.info(f'[VAL - {iteration}] Average precision: {(100 * ap):.2f} | Optimal threshold: {threshold}')
                    if ap >= bestAP:
                        bestAP = ap
                        experiment.save_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth', iteration, bestAP)

            exit()
            iteration += 1  


if __name__ == '__main__':

    opt = parse_arguments()

    # Setup output directories
    os.makedirs(opt['output_path'], exist_ok=True)

    # Setup logger
    logging.basicConfig(filename=f'{opt["output_path"]}/log.txt', format='%(message)s', level=logging.INFO, filemode='a')

    main(opt)