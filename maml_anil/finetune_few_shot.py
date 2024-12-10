import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import learn2learn as l2l
from learn2learn.data.transforms import FusedNWaysKShots, LoadData, RemapLabels, ConsecutiveLabels
from datasets.vggface2_dataset import VGGFace2Dataset
from datasets.buptcbface12_dataset import BUPTCBFaceDataset
from datasets.demogpairs_dataset import DemogPairsDataset
from models.simple_cnn import SimpleCNN

def parse_args():
    parser = argparse.ArgumentParser(description='Few-shot evaluation using standard support/query splits.')
    parser.add_argument('--ways', type=int, default=5, help='Number of ways (classes) per task')
    parser.add_argument('--shots', type=int, default=5, help='Number of shots per class in the support set')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for finetuning')
    parser.add_argument('--adaptation_steps', type=int, default=5, help='Number of adaptation steps (epochs) on the support set')
    parser.add_argument('--use_cuda', type=int, default=1, help='Use CUDA if available')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_tasks', type=int, default=500, help='Number of tasks to sample and evaluate')
    parser.add_argument('--feature_extractor_path', type=str, default='best_feature_extractor.pth', help='Path to the saved feature extractor weights.')
    return parser.parse_args()

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def split_support_query(data, labels, ways, shots):
    """
    Given a task with N ways and 2*K shots per class, we split:
    - First K shots for support
    - Next K shots for query

    data shape: [N*(2*K), ...]
    labels shape: [N*(2*K)]

    Returns:
        support_data, support_labels, query_data, query_labels
    """
    data = data.view(ways, 2*shots, *data.shape[1:])
    labels = labels.view(ways, 2*shots)

    support_data = data[:, :shots].contiguous().view(-1, *data.shape[2:])
    support_labels = labels[:, :shots].contiguous().view(-1)
    query_data = data[:, shots:].contiguous().view(-1, *data.shape[2:])
    query_labels = labels[:, shots:].contiguous().view(-1)

    return support_data, support_labels, query_data, query_labels

def main():
    args = parse_args()

    # Set random seed and device
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')

    ways = args.ways
    shots = args.shots

    # Load datasets and merge them
    # Combine train, val, test splits of the dataset into one big dataset for simplicity.
    #train_dataset = VGGFace2Dataset(mode='train', root="data/vggface2/data", force_new_split=True)
    #val_dataset = VGGFace2Dataset(mode='val', root="data/vggface2/data", force_new_split=True)
    #test_dataset = VGGFace2Dataset(mode='test', root="data/vggface2/data", force_new_split=True)
    train_dataset = BUPTCBFaceDataset(mode='train', root="data/bupt_balanced_face/BUPT-CBFace-12", cache_images=False, force_new_split=False)
    val_dataset = BUPTCBFaceDataset(mode='val', root="data/bupt_balanced_face/BUPT-CBFace-12", cache_images=False, force_new_split=False)
    test_dataset = BUPTCBFaceDataset(mode='test', root="data/bupt_balanced_face/BUPT-CBFace-12", cache_images=False, force_new_split=False)

    # train_dataset = DemogPairsDataset(mode='train', root="data/demogpairs/DemogPairs/DemogPairs", cache_images=False, force_new_split=False)
    # val_dataset = DemogPairsDataset(mode='val', root="data/demogpairs/DemogPairs/DemogPairs", cache_images=False, force_new_split=False)
    # test_dataset = DemogPairsDataset(mode='test', root="data/demogpairs/DemogPairs/DemogPairs", cache_images=False, force_new_split=False)
    combined_meta = l2l.data.UnionMetaDataset([
        l2l.data.MetaDataset(train_dataset),
        l2l.data.MetaDataset(val_dataset),
        l2l.data.MetaDataset(test_dataset),
    ])

    # For standard few-shot: we only need 2*K shots per class (K support + K query)
    transforms = [
        FusedNWaysKShots(combined_meta, n=ways, k=2 * shots),
        LoadData(combined_meta),
        RemapLabels(combined_meta),
        ConsecutiveLabels(combined_meta),
    ]
    tasks = l2l.data.Taskset(combined_meta, task_transforms=transforms, num_tasks=args.num_tasks)

    loss_fn = nn.CrossEntropyLoss()
    accuracies = []

    for i in range(args.num_tasks):
        # Sample one task
        data, labels = tasks.sample()
        data, labels = data.to(device), labels.to(device)

        # Split into support and query
        support_data, support_labels, query_data, query_labels = split_support_query(data, labels, ways, shots)

        # Load the pre-trained feature extractor
        base_model = SimpleCNN(output_size=ways, hidden_size=64, embedding_size=64*4)
        feature_extractor = base_model.features
        feature_extractor.load_state_dict(torch.load(args.feature_extractor_path, map_location=device))
        feature_extractor.to(device)

        # Create a fresh classifier
        classifier = base_model.classifier
        classifier.to(device)


        optimizer = torch.optim.Adam(list(feature_extractor.parameters()) + list(classifier.parameters()), lr=args.lr)

        # Adapt on the support set
        # Standard few-shot often uses a small number of gradient steps on the support set
        for step in range(args.adaptation_steps):
            feature_extractor.train()
            classifier.train()
            optimizer.zero_grad()
            support_preds = classifier(feature_extractor(support_data))
            train_loss = loss_fn(support_preds, support_labels)
            train_loss.backward()
            optimizer.step()

        # Evaluate on the query set
        feature_extractor.eval()
        classifier.eval()
        with torch.no_grad():
            query_preds = classifier(feature_extractor(query_data))
            query_acc = accuracy(query_preds, query_labels)

        accuracies.append(query_acc.item())
        print(f"Task {i+1}/{args.num_tasks}: Query Accuracy = {query_acc.item()*100:.2f}%")

    # Average accuracy across all tasks
    avg_acc = np.mean(accuracies)
    print(f"Average Query Accuracy over {args.num_tasks} tasks: {avg_acc*100:.2f}%")

if __name__ == '__main__':
    main()
