import multiprocessing as mp
import os
import time
from queue import Empty
from typing import Union

import fasttext
import fsspec
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


fasttext.FastText.eprint = lambda x: None

import math

from baselines.vis_utils import ImageCaptionVisualizer





# datacomp
text_name_assigned = "text"
key_name_assigned = "clip_l14_similarity_score" 
feature_name_assigned = "l14_img"

# cc12m
# text_name_assigned = "caption"
# key_name_assigned = "oai-clip-vit-l14-score"
# feature_name_assigned = "oai-clip-vit-l14-image"

@torch.no_grad()
def filter_by_given_uids(given_uids: np.ndarray, uids: np.ndarray, given_uids_index_in_ordered_uids_path: str) -> np.ndarray:
    """ return the index of uids that are in the given uids in a parallel way

    Args:
        given_uids (np.ndarray): given uids
        uids (np.ndarray): uids to be filtered
        
        The format of uids is 
        np.concatenate(
            [np.array(
                    [(int(uid[:16], 16), int(uid[16:32], 16)) for uid in uids],
                    np.dtype("u8,u8"),
                )]
        )
        
    Returns:
        np.ndarray: index of uids that are in the given uids
    
    """
    
    # sort to accelerate the search, and obtain the sorted indices in order for align vas and css
    start = time.time()
    sort_indices = np.argsort(uids)
    uids = uids[sort_indices]
    print(f'end to sort uids, time = {time.time() - start}')
    # given_uids dont need to be sorted
    
    if given_uids_index_in_ordered_uids_path is not None and os.path.exists(given_uids_index_in_ordered_uids_path):
        indices = np.load(given_uids_index_in_ordered_uids_path)
        print(f'load given_uids_index_in_ordered_uids, len = {len(indices)}, path = {given_uids_index_in_ordered_uids_path}')
    else:
        # find the index of uids that are in the given uids
        start2 = time.time()
        print('begin to search uids')
        indices = np.searchsorted(uids, given_uids)
        print(f'end to search uids, time = {time.time() - start2}')
        indices = indices[indices < len(uids)]
        indices = indices[uids[indices] == given_uids] # filter out the uids that are not in the given uids
        
        if given_uids_index_in_ordered_uids_path is not None:
            np.save(given_uids_index_in_ordered_uids_path, indices)
            print(f'save given_uids_index_in_ordered_uids, len = {len(indices)}, path = {given_uids_index_in_ordered_uids_path}')
        
    # align the indices to the original order
    indices = sort_indices[indices]
    
    print(f'end to filter_by_given_uids, total time = {time.time() - start}')
    
    return indices

@torch.no_grad()
def filter_by_score(scores: np.ndarray, fraction: float, threshold: float, total_num: int, name: str = '') -> np.ndarray:
    """ filter the score by fraction and threshold
    
    Args:
        scores (np.ndarray): score to be filtered
        fraction (float): fraction to be filtered
        threshold (float): threshold to be filtered
        total_num (int): total number of scores
        name (str): name of the score
    
    Returns:
        np.ndarray: index of the score that are in the given uids
    """
    assert fraction is not None or threshold is not None, "fraction or threshold should be specified"
    
    if fraction is not None:
        
        n = int(total_num * fraction)
        
        print(f'The threshold for {name} is not specified, select top {fraction} fraction. begin to sort score.')
        select_indices = np.argpartition(scores, -n)[-n:] 
    
    else: # threshold is not None
        print(f'The fraction for {name} is not specified, threshold is {threshold}.')
        select_indices = np.where(scores >= threshold)[0]
    
    scores_tmp = scores[select_indices]
    print(f'After filtering {name}, the fraction = {len(scores_tmp)/total_num}, the threshold = {np.min(scores_tmp)}, mean = {np.mean(scores_tmp)}, max = {np.max(scores_tmp)}. len = {len(scores_tmp)}')
        
    return select_indices
        

@torch.no_grad()
def get_vas_gpu(
    embeddings: torch.Tensor, target_variance: torch.Tensor, device: int
) -> torch.Tensor:
    """ calculate vas for each embeddings. VAS(i) = f_i^T S f_i, where S is the target variance matrix, f_i is the i-th embedding
    
    Args:
        embeddings (torch.Tensor): embeddings to calculate vas
        target_variance (torch.Tensor): target variance matrix
        device (int): gpu number
        
    Returns:
        torch.Tensor: variance alignment score for each embeddings
    """
    device_string = f"cuda:{device}"
    target_variance_gpu = target_variance.float().to(device_string)
    embeddings_gpu = embeddings.float().to(device_string)
    
    vas = torch.sum((embeddings_gpu @ target_variance_gpu) * embeddings_gpu, dim=1).cpu()
    
    return vas

@torch.no_grad()
def vas_filter_helper(
    device_index: int,
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    arch: Union[str, None] = None,
    
    target_variance: torch.Tensor = None,
    is_vas_d: Union[bool, None] = False,
    
) -> None:
    """worker function to variance alignment score filtering, pulling off a queue of tasks
    
    Args:
        target_variance (torch.Tensor): target variance matrix
        device_index (int): device on which to run the gpu processing
        in_queue (mp.Queue): task queue with fsspec, metadata path pairs
        out_queue (mp.Queue): output queue to send filtred uids
        arch: (Union[str, None]): If specified, we want to apply a threshold to arch=B/32 or L/14 clip scores. Defaults to None.
        if_vas_d (Union[bool, None]): if True,  we will return the candidate_embedding and don't return vas. Defaults to False.
    
    """
    
    while True:
        fs_root = None
        try:
            fs_root = in_queue.get(timeout=1)
        except Empty:
            # case where the queue is depleated, worker should return
            break
        
        fs, path_root = fs_root
        
        feature_name = feature_name_assigned
        if arch is not None:
            key = key_name_assigned
            if arch == "b32":
                key = "clip_b32_similarity_score"
                feature_name = "b32_img"
            
            
            df = pd.read_parquet(
                f"{path_root}.parquet", columns=["uid", text_name_assigned, key], filesystem=fs
            )
        else:
            df = pd.read_parquet(
                f"{path_root}.parquet", columns=["uid", text_name_assigned], filesystem=fs
            )
        # print(f'feature_name = {feature_name}, key = {key}, arch = {arch}')

        candidate_embedding = None
        with fs.open(f"{path_root}.npz") as f: # -> float32
            candidate_embedding = torch.from_numpy(np.load(f)[feature_name])#.float()

        uids = df["uid"].values
        
        uids_standard = np.array(
                            [(int(uid[:16], 16), int(uid[16:32], 16)) for uid in uids],
                            np.dtype("u8,u8"),
                        )
        
        # clip scores
        if arch is not None:
            css = df[key].values
        else:
            css = None
        
        if is_vas_d:
            out_queue.put(
                (
                    uids_standard,
                    candidate_embedding.cpu(), # will calculate VAS in the main thread
                    css,
                )
            )
        else:
            vass = get_vas_gpu(
                candidate_embedding,
                target_variance,
                device_index,
            )
            out_queue.put(
                (
                    uids_standard,
                    vass.numpy(),
                    css,
                )
            )      




@torch.no_grad()
def load_all_data(
        metadata_dir_path: str,
        arch: str, 
        num_gpus: int,
        
        given_uids_path: str = None,
        target_variance: torch.Tensor = None,
        is_vas_d: bool = False,
        batch_size: int = 100000,
        
    ):
    """Load embeddings, UIDs, and CLIP scores from files, filter by given UIDs and CLIP score threshold, and calculate initial target variance.
    
    Args:
        metadata_dir_path (str): Directory path containing metadata files.
        arch (str): Kind of features for CLIP filtering.
        out_queue (mp.Queue): Output queue to send loaded data.
        num_gpus (int): Number of GPU workers.
        
        given_uids_path (str): Path to the given UIDs.
        target_variance (torch.Tensor): Target variance matrix.
        is_vas_d (bool): If True, we will return the candidate_embedding and don't return VAS.
    
        batch_size (int): Batch size for calculating target variance (just for VAS-D) on device.
        target_variance (torch.Tensor): Target variance matrix.
    """
    
        
        
    fs, url = fsspec.core.url_to_fs(metadata_dir_path)
    root_paths = [
        (fs, str(x.split(".parquet")[0])) for x in fs.ls(url) if ".parquet" in x
    ]

    # initializing task queues
    receive_queue = mp.Queue()
    send_queue = mp.Queue()

    for job in root_paths:
        send_queue.put(job)

    processes = []
    print("starting gpu workers")
    for worker_index in tqdm(range(num_gpus)):
        p = mp.Process(
            target=vas_filter_helper,
            kwargs=dict(
                device_index=worker_index,
                in_queue=send_queue,
                out_queue=receive_queue,
                arch=arch,
                target_variance=target_variance,
                is_vas_d=is_vas_d,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.05)

    print("processing metadata with gpu workers")
    pbar = tqdm(total=len(root_paths))

    # running in main thread
    all_uids = []
    all_embs_or_vass = []
    all_css = []
    
    def update_p(receive_queue, all_uids, all_embs_or_vass, all_css, pbar):
        # utility function to update the progress bar and store results
        uids, embs_or_vass, css = receive_queue.get(timeout=10)
        all_uids.append(uids)
        all_embs_or_vass.append(embs_or_vass)
        all_css.append(css)
        pbar.update(1)
        
        return pbar.n
                            
    while True:
        # keep checking for jobs finishing and update uids
        try:
            counter = update_p(receive_queue, all_uids, all_embs_or_vass, all_css, pbar)
            # if debug == 1 and counter == 20: break
        except TimeoutError:
            if all(not p.is_alive() for p in processes):
                try:
                    update_p(receive_queue, all_uids, all_embs_or_vass, all_css, pbar)
                    
                except TimeoutError:
                    raise RuntimeError("All processes dead but nothing in queue!")
        except Empty:
            pass

        if all(not p.is_alive() for p in processes):
            # case where all workers have exited
            try:
                update_p(receive_queue, all_uids, all_embs_or_vass, all_css, pbar)
                
            except Empty:
                print("Result queue is empty and all workers have exited")
                break

    pbar.close()
    for p in processes:
        p.join(timeout=1)

    
    uids = np.concatenate(all_uids)
    
    if is_vas_d: # embed -> torch.Tensor float32
        embs_or_vass = torch.cat(all_embs_or_vass)
    else: # vass -> np.ndarray
        embs_or_vass = np.concatenate(all_embs_or_vass).astype(np.float32)

    css = np.concatenate(all_css)
    
    print(f'uids.shape = {uids.shape}, embs_or_vass.shape = {embs_or_vass.shape}, css.shape = {css.shape}')
    
    # Filter by given UIDs, which may be the subet from other filtering methods
    if given_uids_path is not None:
        given_uids = np.load(given_uids_path)
        
        given_uids_mask = np.isin(uids, given_uids)
        uids = uids[given_uids_mask]
        embs_or_vass = embs_or_vass[given_uids_mask] # note that tensor can take np.ndarray as index
        css = css[given_uids_mask]
        print(f'================== after filtering by given_uids ==================')
        print(f'uids.shape = {uids.shape}, embs_or_vass.shape = {embs_or_vass.shape}, css.shape = {css.shape}')
    
    # Calculate initial target variance as the sum of outer products
    if is_vas_d:
        target_variance = cal_target_variance(embs_or_vass, num_gpus, batch_size)
        return uids, css, embs_or_vass, target_variance
    else:
        return uids, embs_or_vass, css
    
######################################################## VAS-D ###############################################################

@torch.no_grad()
def cal_target_variance_worker(
    emb: torch.Tensor,
    send_queue: mp.Queue,
    result_queue: mp.Queue,
    gpu_index: int,
):  
    # sum_i f_i f_i^T, use gpu and vector multiplication to accelerate the calculation
    while True:
        try:
            start, end = send_queue.get(timeout=1)
            emb_chunk = emb[start:end].float().to(f'cuda:{gpu_index}') # has copied the emb to the gpu, and don't influence the main emb #.float()
            # target_variance = torch.einsum('ni,nj->ij', emb_chunk, emb_chunk).cpu()
            
            target_variance = (emb_chunk.T @ emb_chunk).cpu()
            
            result_queue.put(target_variance)
            # delete the emb_chunk to save the gpu memory
            
            del emb_chunk # delete the emb_chunk to save the gpu memory
            
            
        except Empty:
            break
    
@torch.no_grad()   
def cal_target_variance(
    emb: torch.Tensor,
    num_gpus: int,
    batch_size: int,
):
    """
        calculate the target variance using multiple queues to accelerate the calculation
        
        S = \sum_i^N f_i f_i^T / N, so we can calculate the sum of outer products in each queue, and then sum them up
        
    """
    emb_num = emb.shape[0]
    emb_dim = emb.shape[1]

    idx_chunks = []
    
    num_batches = math.ceil(emb_num / batch_size)
    
    for i in range(num_batches):
        start = i * batch_size
        end = min((i+1) * batch_size, emb_num)
        idx_chunks.append((start, end))
    
    result_queue = mp.Queue()
    send_queue = mp.Queue()
    
    for idx_pair in idx_chunks:
        send_queue.put(idx_pair)
    
    print(f'len(send_queue) = {send_queue.qsize()}, len(idx_chunks) = {len(idx_chunks)}, num_batches = {num_batches}, emb_num = {emb_num}')
    processes = []
    print("starting gpu workers")
    
    for worker_index in tqdm(range(num_gpus)):
        p = mp.Process(
            target=cal_target_variance_worker,
            kwargs=dict(
                emb=emb,
                send_queue=send_queue,
                result_queue=result_queue,
                gpu_index=worker_index,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.05)
        
    target_variance = torch.zeros(emb_dim, emb_dim)
    
    print("processing target_variance with gpu workers")
    pbar = tqdm(total=len(idx_chunks))
    
    def update_target_variance(result_queue, target_variance, pbar):
        # utility function to update the progress bar and store results
        target_variance += result_queue.get()
        pbar.update(1)
        return pbar.n
    
    while True:
        try:
            if update_target_variance(result_queue, target_variance, pbar) == num_batches: break
            
        except TimeoutError:
            if all(not p.is_alive() for p in processes):
                try:
                    if update_target_variance(result_queue, target_variance, pbar) == num_batches: break
                except TimeoutError:
                    raise RuntimeError("All processes dead but nothing in queue!")
        except Empty:
            pass

        if all(not p.is_alive() for p in processes):
            # case where all workers have exited
            try:
                if update_target_variance(result_queue, target_variance, pbar) == num_batches: break
            except Empty:
                print("Result queue is empty and all workers have exited")
                break
    
    pbar.close()
    
    
    for p in processes:
        p.join(timeout=0.1) 
    
    
    target_variance /= emb_num
    
    print(f'target_variance.shape = {target_variance.shape}, mean, min, max = {target_variance.mean()}, {target_variance.min()}, {target_variance.max()}')
    
    return target_variance
    


@torch.no_grad()
def load_uids_with_vas_filter(
    metadata_dir_path: str,
    files_path: str,
    
    num_gpus: int,
    arch: Union[str, None] = None,
    threshold: Union[float, None] = None,
    fraction: Union[float, None] = None,
    
    threshold_vas: Union[float, None] = None,
    fraction_vas: Union[float, None] = None,
    target_variance_name: str = 'imagenet-1k',
    given_uids_path: Union[str, None] = None,
    
    higher_is_better_vas: int = 1
) -> np.ndarray:
    """vas filter

    Args:
        metadata_dir_path (str): Directory path containing metadata files.
        files_path (str): Path to the embedding files.
        num_gpus (int): Number of GPU workers.
        arch (Union[str, None], optional): Kind of features for CLIP filtering. Defaults to None.
        threshold (Union[float, None], optional): Threshold to apply to CLIP features. Defaults to None.
        fraction (Union[float, None], optional): Top k fraction to apply to CLIP features. Defaults to None.
        threshold_vas (Union[float, None], optional): Threshold to apply to variance alignment scores. Defaults to None.
        fraction_vas (Union[float, None], optional): Top k fraction to apply to variance alignment scores. Defaults to None.
        target_variance_name (str, optional): Target variance name. Defaults to 'imagenet-1k'.
        given_uids_path (Union[str, None], optional): Path to the given UIDs. Defaults to None.
    """
    
    
    # load target variance
    print("loading target variance")
    
    target_path = os.path.join(files_path, 'variance', f"variance_{target_variance_name}.pt") # VAS(target proxy), like VAS(imagenet_1k)
    target_variance = torch.load(target_path)
    print(f'load target_variance from {target_path}')
    
    
    uids, vass, css = load_all_data(metadata_dir_path, arch, num_gpus, given_uids_path, target_variance=target_variance, is_vas_d=False)
    total_num = len(uids) # original number of uids
    
    # first filter by clip score
    select_indices = filter_by_score(css, fraction, threshold, total_num, name='clip_score')
    uids = uids[select_indices]
    vass = vass[select_indices]
    
    print(f'================== after filtering by clip_score ==================')
    print(f'uids.shape = {uids.shape}, vass.shape = {vass.shape}')
    
    if not higher_is_better_vas: # lower is better
        print(f'!!! vass is lower is better, so we will change the sign of vass')
        vass = -vass

    # Perform VAS filtering
    select_indices_vas = filter_by_score(vass, fraction_vas, threshold_vas, total_num, name='vas')
    uids = uids[select_indices_vas]
    
    print(f'================== after filtering by vas ==================')
    print(f'uids.shape = {uids.shape}, vass.shape = {vass.shape}')
    
    return uids
    
    



@torch.no_grad()
def load_uids_with_vas_d_filter(
    metadata_dir_path: str,
    
    num_gpus: int,
    arch: Union[str, None] = None,
    threshold: Union[float, None] = None,
    fraction: Union[float, None] = None,
    
    given_uids_path: Union[str, None] = None,
    num_iters: Union[int, None] = 100,
    fraction_vas: Union[float, None] = None,
    
    batch_size: Union[int, None] = 100000,
    batch_size_vass: Union[int, None] = None,
):
    """Perform VAS-D filtering algorithm with initial filtering by given UIDs and CLIP score threshold.
    If num_iters is 1, it's equivalent to VAS(Traindata) filtering.
    
    Args:
        metadata_dir_path (str): Directory path containing metadata files.
        files_path (str): Path to the embedding files.
        num_gpus (int): Number of GPU workers.
        arch (Union[str, None], optional): Kind of features for CLIP filtering. Defaults to None.
        threshold (Union[float, None], optional): Threshold to apply to CLIP features. Defaults to None.
        fraction (Union[float, None], optional): Top k fraction to apply to CLIP features. Defaults to None.
        given_uids_path (Union[str, None], optional): Path to the given UIDs. Defaults to None.
        num_iters (Union[int, None], optional): Number of iterations for VAS-D. Defaults to 100.
        target_size (Union[int, None], optional): Target size of the final UIDs. Defaults to None.
        batch_size (Union[int, None], optional): Batch size for VAS-D on device.
    """
    
    
    
    # Load all data and calculate initial target variance
    uids, css, embs, target_variance = load_all_data(metadata_dir_path, arch, num_gpus, given_uids_path, is_vas_d=True, batch_size=batch_size)
    total_num = len(uids) # original number of uids
    
    target_size = int(fraction_vas * total_num)
    
    # first filter by clip score
    select_indices = filter_by_score(css, fraction, threshold, total_num, name='clip_score')
    uids = uids[select_indices]
    embs = embs[select_indices]

    print(f'================== after filtering by clip_score ==================')
    print(f'uids.shape = {uids.shape}, embs.shape = {embs.shape}, target_size = {target_size}')
    
    
    # Perform VAS-D filtering
    chunk_size = math.ceil((total_num - target_size) / num_iters)
    
    for i in range(num_iters):
        
        print(f'======================================= iter {i}, total_num = {total_num}, batch_size = {batch_size}, chunk_size = {chunk_size}, target_size = {target_size}, batch_size_vass = {batch_size_vass}')
        
        vass = cal_vass_iter(embs, num_gpus, target_variance, batch_size_vass, total_num)
        
        if i == num_iters - 1:
            topk = target_size
        else:
            topk = total_num - chunk_size
            
        _, indices = torch.topk(vass, topk)
        uids = uids[indices]
        embs = embs[indices]
        
        target_variance = cal_target_variance(embs, num_gpus, batch_size)
        
    return uids



@torch.no_grad()
def cal_vass_iter(
    embs: torch.Tensor,
    num_gpus: int,
    target_variance: torch.Tensor,
    batch_size_vass: int,
    total_size: int,

):
    """
        calculate the VAS-D using multiple queues to accelerate the calculation
        
    """
    data_queue = mp.Queue()
        
    if batch_size_vass is not None:        
        for j in range(0, total_size, batch_size_vass):
            data_queue.put((j, j+batch_size_vass))
    else: # process in one batch
        data_queue.put((0, total_size))
    
    result_queue = mp.Queue()
    processes = []
    
    target_variance_list = []
    
    # just define the new target_variance on each device
    for worker_index in range(num_gpus):
        target_variance_list.append(target_variance.clone().to(f'cuda:{worker_index}'))
    
    print("starting gpu workers for VAS-D")
    for worker_index in tqdm(range(num_gpus)):
        p = mp.Process(
            target=vas_d_worker,
            kwargs=dict(
                emb = embs,
                in_queue=data_queue,
                out_queue=result_queue,
                device_id=worker_index,
                target_variance_list=target_variance_list,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.05)

    vass = torch.empty(total_size)
    
    print("processing VAS-D with gpu workers")
    
    # len of data_queue
    if batch_size_vass is None:
        total_pbar_size = 1
    else:
        total_pbar_size = math.ceil(total_size / batch_size_vass)
    
    pbar = tqdm(total=total_pbar_size)
    
    
    def update_vas_d(result_queue, vass, pbar):
        # utility function to update the progress bar and store results
        vas, start, end = result_queue.get() # timeout=20)
        vass[start:end] = vas
        pbar.update(1)
        # return the value of pbar
        return pbar.n
    
    while True:
        try:
            if update_vas_d(result_queue, vass, pbar) == total_pbar_size: break
        except TimeoutError:
            if all(not p.is_alive() for p in processes):
                try:
                    if update_vas_d(result_queue, vass, pbar) == total_pbar_size: break
                except TimeoutError:
                    raise RuntimeError("All processes dead but nothing in queue!")
        except Empty:
            pass

        if all(not p.is_alive() for p in processes):
            try:
                if update_vas_d(result_queue, vass, pbar) == total_pbar_size: break
            except Empty:
                print("Result queue is empty and all workers have exited")
                break
    
    
    pbar.close()
    for p in processes:
        p.join(timeout=0.1)
    
    print(f'mean, min, max of VAS = {vass.mean()}, {vass.min()}, {vass.max()}')
    
    return vass



@torch.no_grad()
def vas_d_worker(emb, in_queue, out_queue, device_id, target_variance_list):
    """Worker process to calculate VAS for a chunk of embeddings.
    
    Args:
        in_queue (mp.Queue): Input queue to get embedding chunk.
        out_queue (mp.Queue): Output queue to send calculated VAS.
        device_id (int): GPU device index.
        target_variance (torch.Tensor): Target variance matrix.
    """
    while True:
        try:
            start, end = in_queue.get(timeout=1)
            emb_chunk = emb[start:end].float().to(f'cuda:{device_id}') # more efficient to copy the emb to the gpu, don't leave emb_chunk results on the cpu #.float()
            
            target_variance2 = target_variance_list[device_id]
            
            # vas = torch.einsum('ni,ij,nj->n', emb_chunk, target_variance2, emb_chunk) # calculate with @, which is more efficient 
            vas = torch.sum((emb_chunk @ target_variance2) * emb_chunk, dim=1).cpu()
            
            out_queue.put((vas, start, end))
            
            # delete the emb_chunk to save the gpu memory
            del emb_chunk
            
        except Empty:
            break
        
    