
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

def build_splits(opt):
    source_domain = 'art_painting'
    target_domain = opt['target_domain']

    source_examples = read_lines(opt['data_path'], source_domain)
    target_examples = read_lines(opt['data_path'], target_domain)

    # Compute ratios of examples for each category
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}

    # Build splits - we train only on the source domain (Art Painting)
    val_split_length = source_total_examples * 0.2 # 20% of the training split used for validation

    train_examples = []
    val_examples = []
    test_examples = []

    for category_idx, examples_list in source_examples.items():
        split_idx = round(source_category_ratios[category_idx] * val_split_length)
        for i, example in enumerate(examples_list):
            if i > split_idx:
                train_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
            else:
                val_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
    
    for category_idx, examples_list in target_examples.items():
        for example in examples_list:
            test_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
    
    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    train_loader = DataLoader(PACSDatasetBaseline(train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetBaseline(val_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetBaseline(test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader