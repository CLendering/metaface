import time
import torch
import torch.nn as nn
import numpy as np
import random
import learn2learn as l2l
from learn2learn.data.transforms import FusedNWaysKShots, LoadData, RemapLabels, ConsecutiveLabels
from models.camile_net import CamileNet
from models.resnet import Resnet18Model
from models.simple_cnn import SimpleCNN
from datasets.buptcbface12_dataset import BUPTCBFaceDataset
from datasets.demogpairs_dataset import DemogPairsDataset
from datasets.vggface2_dataset import VGGFace2Dataset

from config import parse_args
import wandb

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def fast_adapt(batch,
               learner,
               feature_extractor,
               loss_fn,
               adaptation_steps,
               shots,
               ways,
               device=None):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)
    data = feature_extractor(data)

    # Split into adaptation/evaluation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots * ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)

    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    for _ in range(adaptation_steps):
        train_error = loss_fn(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error)

    predictions = learner(evaluation_data)
    valid_error = loss_fn(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy

def main(
    ways=5,
    shots=5,
    meta_learning_rate=0.001,
    fast_learning_rate=0.1,
    adaptation_steps=5,
    meta_batch_size=32,
    iterations=1000,
    use_cuda=1,
    seed=42,
    number_train_tasks=20000,
    number_valid_tasks=600,
    number_test_tasks=600,
    patience=10,  # Number of iterations to wait for improvement
    save_path='best_feature_extractor.pth',
    debug_mode=False,
    use_wandb=False
):
    
    if use_wandb:
        wandb.init(
            project="ibb-maml-anil",
            entity="benchmark_bros",
            config={
                "meta_learning_rate": meta_learning_rate,
                "fast_learning_rate": fast_learning_rate,
                "adaptation_steps": adaptation_steps,
                "meta_batch_size": meta_batch_size,
                "iterations": iterations,
                "number_train_tasks": number_train_tasks,
                "number_valid_tasks": number_valid_tasks,
                "number_test_tasks": number_test_tasks,
            },
        )

    use_cuda = bool(use_cuda)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load datasets
    bupt_train_dataset = BUPTCBFaceDataset(mode='train', root="data/bupt_balanced_face/BUPT-CBFace-12", cache_images=False, force_new_split=False)
    bupt_valid_dataset = BUPTCBFaceDataset(mode='val', root="data/bupt_balanced_face/BUPT-CBFace-12", cache_images=False, force_new_split=False)
    bupt_test_dataset = BUPTCBFaceDataset(mode='test', root="data/bupt_balanced_face/BUPT-CBFace-12", cache_images=False, force_new_split=False)

    demog_train_dataset = DemogPairsDataset(mode='train', root="data/demogpairs/DemogPairs/DemogPairs", cache_images=False, force_new_split=False)
    demog_valid_dataset = DemogPairsDataset(mode='val', root="data/demogpairs/DemogPairs/DemogPairs", cache_images=False, force_new_split=False)
    demog_test_dataset = DemogPairsDataset(mode='test', root="data/demogpairs/DemogPairs/DemogPairs", cache_images=False, force_new_split=False)

    # vgg_face_2_train_dataset = VGGFace2Dataset(mode='train', root="data/vggface2/data", force_new_split=False)
    # vgg_face_2_val_dataset = VGGFace2Dataset(mode='val', root="data/vggface2/data", force_new_split=False)
    # vgg_face_2_test_dataset = VGGFace2Dataset(mode='test', root="data/vggface2/data", force_new_split=False)
    
    # vgg_face_2_meta_train_dataset = l2l.data.MetaDataset(vgg_face_2_train_dataset)
    # vgg_face_2_meta_valid_dataset = l2l.data.MetaDataset(vgg_face_2_val_dataset)
    # vgg_face_2_meta_test_dataset = l2l.data.MetaDataset(vgg_face_2_test_dataset)


    start_time = time.time()
    bupt_meta_train_dataset = l2l.data.MetaDataset(bupt_train_dataset)
    bupt_meta_valid_dataset = l2l.data.MetaDataset(bupt_valid_dataset)
    bupt_meta_test_dataset = l2l.data.MetaDataset(bupt_test_dataset)
    print('Time to load BUPT META datasets:', time.time() - start_time)
    
    start_time = time.time()
    demog_meta_train_dataset = l2l.data.MetaDataset(demog_train_dataset)
    demog_meta_valid_dataset = l2l.data.MetaDataset(demog_valid_dataset)
    demog_meta_test_dataset = l2l.data.MetaDataset(demog_test_dataset)
    print('Time to load DEMOG META datasets:', time.time() - start_time)
    

    train_datasets = [bupt_meta_train_dataset, demog_meta_train_dataset]
    valid_datasets = [bupt_meta_valid_dataset, demog_meta_valid_dataset]
    test_datasets = [bupt_meta_test_dataset, demog_meta_test_dataset]

    start_time = time.time()
    union_train = l2l.data.UnionMetaDataset(train_datasets)
    union_valid = l2l.data.UnionMetaDataset(valid_datasets)
    union_test = l2l.data.UnionMetaDataset(test_datasets)
    print('Time to load UNION META datasets:', time.time() - start_time)
    if len(union_train.labels_to_indices) == len(bupt_meta_train_dataset.labels_to_indices) + len(demog_meta_train_dataset.labels_to_indices):
        print('Union dataset is working properly')
    else:
        raise ValueError('Union dataset is not working properly')

    train_transforms = [
        FusedNWaysKShots(union_train, n=ways, k=2 * shots),
        LoadData(union_train),
        RemapLabels(union_train),
        ConsecutiveLabels(union_train),
    ]
    train_tasks = l2l.data.Taskset(
        union_train,
        task_transforms=train_transforms,
        num_tasks=number_train_tasks if not debug_mode else 50,
    )

    valid_transforms = [
        FusedNWaysKShots(union_valid, n=ways, k=2 * shots),
        LoadData(union_valid),
        ConsecutiveLabels(union_valid),
        RemapLabels(union_valid),
    ]
    valid_tasks = l2l.data.Taskset(
        union_valid,
        task_transforms=valid_transforms,
        num_tasks=number_valid_tasks if not debug_mode else 50,
    )

    test_transforms = [
        FusedNWaysKShots(union_test, n=ways, k=2 * shots),
        LoadData(union_test),
        RemapLabels(union_test),
        ConsecutiveLabels(union_test),
    ]
    test_tasks = l2l.data.Taskset(
        union_test,
        task_transforms=test_transforms,
        num_tasks=number_test_tasks if not debug_mode else 50,
    )

    model = SimpleCNN(
        output_size=ways,
        hidden_size=64,
        embedding_size=64*4,
    )
    # model = Resnet18Model(
    #     embedding_size=512,
    #     output_size=ways,
    #     dropout_p=0.2,
    # )
    model = CamileNet(
        input_channels=3,
        hidden_size=64,
        embedding_size=64,
        output_size=ways
    )
    feature_extractor = model.features
    #feature_extractor.load_state_dict(torch.load("best_feature_extractor.pth", map_location=device)) #######################
    head = model.classifier
    feature_extractor.to(device)
    head = l2l.algorithms.MAML(head, lr=fast_learning_rate)
    head.to(device)

    all_parameters = list(feature_extractor.parameters()) + list(head.parameters())
    num_params = sum([np.prod(p.size()) for p in all_parameters])
    print('Total number of parameters:', num_params / 1e6, 'Millions')

    optimizer = torch.optim.Adam(all_parameters, lr=meta_learning_rate)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')

    best_meta_test_error = float('inf')
    patience_counter = 0

    for iteration in range(iterations):
        optimizer.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0

        # Meta-train & Meta-validation steps
        for _ in range(meta_batch_size):
            # Meta-training
            learner = head.clone()
            batch = train_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(
                batch, learner, feature_extractor, loss_fn, adaptation_steps, shots, ways, device
            )
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

        print('\nIteration:', iteration)
        print('Meta Train Error:', meta_train_error / meta_batch_size)
        print('Meta Train Accuracy:', meta_train_accuracy / meta_batch_size)

        for p in all_parameters:
            p.grad.data.mul_(1.0 / meta_batch_size)
        optimizer.step()

        # Evaluate on Meta-Test tasks for early stopping
        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for _ in range(meta_batch_size):
            # Validation Set

            learner = head.clone()
            batch = valid_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(
                batch, learner, feature_extractor, loss_fn, adaptation_steps, shots, ways, device
            )
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

            # Test set
            learner = head.clone()
            batch = test_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(
                batch, learner, feature_extractor, loss_fn, adaptation_steps, shots, ways, device
            )
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()

        meta_test_error /= meta_batch_size
        meta_test_accuracy /= meta_batch_size
        meta_valid_error /= meta_batch_size
        meta_valid_accuracy /= meta_batch_size

        # Combine test and validation set results
        meta_test_valid_error = (meta_test_error + meta_valid_error) / 2
        meta_test_valid_accuracy = (meta_test_accuracy + meta_valid_accuracy) / 2

        print('Meta Test Error:', meta_test_valid_error)
        print('Meta Test Accuracy:', meta_test_valid_accuracy)

        if use_wandb:
            wandb.log({
                "meta_train_error": meta_train_error / meta_batch_size,
                "meta_train_accuracy": meta_train_accuracy / meta_batch_size,
                "meta_test_error": meta_test_valid_error,
                "meta_test_accuracy": meta_test_valid_accuracy,
            })

        # Early stopping logic
        if meta_test_error < best_meta_test_error:
            print(f"New best meta-test error ({best_meta_test_error} -> {meta_test_error}). Saving feature extractor.")
            best_meta_test_error = meta_test_error
            patience_counter = 0
            # Save the feature extractor’s state for later fine-tuning
            torch.save(feature_extractor.state_dict(), save_path)
        else:
            patience_counter += 1
            print("No improvement in meta-test error. Patience:", patience_counter)
            if patience_counter >= patience:
                print("Early stopping triggered. No improvement in meta-test error for", patience, "iterations.")
                break

if __name__ == '__main__':
    options = parse_args()
    main(
        ways=options.ways,
        shots=options.shots,
        meta_learning_rate=options.meta_learning_rate,
        fast_learning_rate=options.fast_learning_rate,
        adaptation_steps=options.adaptation_steps,
        meta_batch_size=options.meta_batch_size,
        iterations=options.iterations,
        use_cuda=options.use_cuda,
        seed=options.seed,
        number_train_tasks=options.number_train_tasks,
        number_valid_tasks=options.number_valid_tasks,
        number_test_tasks=options.number_test_tasks,
        patience=options.patience,
        debug_mode=options.debug_mode,
        use_wandb=options.use_wandb
    )
