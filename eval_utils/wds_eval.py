"""Evaluate on standard classification webdatasets."""

import os

import open_clip
import torch
from clip_benchmark.datasets.builder import build_dataset
from clip_benchmark.metrics import zeroshot_classification as zsc
from sklearn.metrics import balanced_accuracy_score


def create_model(model_arch, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    model_path = str(model_path)
    model, _, transform = open_clip.create_model_and_transforms(
        model_arch, pretrained=model_path
    )
    model.eval()
    # model.half()
    model = model.to(device)

    return model, transform, device


def create_webdataset(
    task, transform, data_root=None, dataset_len=None, batch_size=64, num_workers=4
):
    data_folder = f"wds_{task.replace('/','-')}_test"
    if data_root is None:
        data_root = f"https://huggingface.co/datasets/djghosh/{data_folder}/tree/main"
    else:
        data_root = os.path.join(data_root, data_folder)
    dataset = build_dataset(
        dataset_name=f"wds/{task}",
        root=data_root,
        transform=transform,
        split="test",
        download=False,
    )
    if dataset_len:
        dataset = dataset.with_length((dataset_len + batch_size - 1) // batch_size)
    dataloader = torch.utils.data.DataLoader(
        dataset.batched(batch_size),
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
    )
    return dataset, dataloader


def evaluate_webdataset(
    task,
    model_arch,
    model_path,
    data_root=None,
    dataset_len=None,
    batch_size=64,
    num_workers=4,
    return_preds=False,
    return_topk=False,
):
    """Evaluate CLIP model on classification task."""
    
    
    # Create model
    model, transform, device = create_model(model_arch, model_path)

    # Load data
    dataset, dataloader = create_webdataset(
        task, transform, data_root, dataset_len, batch_size, num_workers
    )

    zeroshot_templates = dataset.templates if hasattr(dataset, "templates") else None
    
    # if task == 'imagenet1k':# store the classifier
    #     print('task = ', task)
    #     # print all the inputs in a efficient way
    #     print(f'locals() = {locals()}')
    #     print('len(zeroshot_templates) = ', len(zeroshot_templates))
    
    classnames = dataset.classes if hasattr(dataset, "classes") else None
    assert (
        zeroshot_templates is not None and classnames is not None
    ), "Dataset does not support classification"

    # Evaluate
    classifier = zsc.zero_shot_classifier(
        model,
        open_clip.get_tokenizer(model_arch),
        classnames,
        zeroshot_templates,
        device,
    )
    
    # if task == 'imagenet1k':# store the classifier
    #     print(f'classifier.shape = {classifier.shape}')
    #     save_path = '/homes/gws/ypwang61/nobackup/files/imagenet_1k/save'
        
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #         print(f'make dir {save_path}')
        
    #     save_file_path = os.path.join(save_path, f'imagenet_1k_classifier.pt')
    #     # store classifier if not exists
    #     if not os.path.exists(save_file_path):
    #         torch.save(classifier, save_file_path)
    #         print(f'save classifier to {save_file_path}')
    #     else:
    #         print(f'{save_file_path} exists, skip saving classifier')
        
        
    logits, target = zsc.run_classification(
        model, classifier, dataloader, device, amp=False
    )
    with torch.no_grad():
        pred = logits.argmax(axis=1).cpu()
        target = target.cpu()

    # Compute metrics
    if len(dataset.classes) >= 5:
        acc1, acc5 = zsc.accuracy(logits, target, topk=(1, 5))
    else:
        (acc1,) = zsc.accuracy(logits, target, topk=(1,))
        acc5 = None
    mean_per_class_recall = balanced_accuracy_score(target, pred)
    metrics = {
        "acc1": acc1,
        "acc5": acc5,
        "mean_per_class_recall": mean_per_class_recall,
    }

    if return_preds:
        if return_topk:
            with torch.no_grad():
                _, topk_pred = torch.topk(logits, int(return_topk), dim=1)
                topk_pred = topk_pred.cpu()
            return metrics, topk_pred, target
        return metrics, pred, target
    return metrics
