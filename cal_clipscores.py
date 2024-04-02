import numpy as np
import matplotlib.pyplot as plt
import os 
import argparse

from baselines.vis_utils import visualize_images_captions

def visualize_clipscores(
    clipscores: np.ndarray,
    extra_id_name: str,
    save_path: str,
    name: str = '',
    bins: int = 100,
):
    # stats of the clipscores
    mean = np.mean(clipscores)
    std = np.std(clipscores)
    max = np.max(clipscores)
    min = np.min(clipscores)
    print(f'mean: {mean:.4f}\nstd: {std:.4f}\nmax: {max:.4f}\nmin: {min:.4f}')

    # print top 10 clipscores
    print(f'top 10 clipscores: {np.sort(clipscores)[-10:]}')
    
    # plot the histogram of the clipscores
    
    # clear the figure
    plt.clf()
    plt.hist(clipscores, bins=bins)
    plt.xlabel(f'clipscores')
    plt.ylabel('count')
    plt.title(f'histogram of clipscores_{extra_id_name}{name}')

    # show the stats on the figure

    plt.text(0.8, 0.8, f'mean: {mean:.4f}\nstd: {std:.4f}\nmax: {max:.4f}\nmin: {min:.4f}', 
            horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

    plt.savefig(save_path + f'hist_clipscores_{extra_id_name}{name}.png')
    print(f'saving histogram of clipscores_{extra_id_name}{name}.png')


    
    
# visualize the images and captions

def generate_p_from_clipscores(
    clipscores: np.ndarray,
    bins: int = 100,
):
    """
    generate the probability for each bin of the histogram of the clipscores
    """
    hist, bin_edges = np.histogram(clipscores, bins=bins)
    p = hist / np.sum(hist)
    return p, bin_edges

    
    

# select and save the uids that have the top k small clipscores
def select_top_k_uids(
        clipscores: np.ndarray,
        uids: np.ndarray,
        # fraction: float,
        extra_id_name: str,
        k: int,
        save_path: str,
        uids_path: str,
        start_index: int = 0,
        choose_small: bool = True, # choose the top k small clipscores if True, otherwise choose the top k large clipscores
        # yp add after 11.29.2020
        urls: np.ndarray = None,
        captions: np.ndarray = None,
        need_save: bool = True,
    ):
    # sort the clipscores
    clipscores_sorted_index = np.argsort(clipscores)
    
    if not choose_small:
        clipscores_sorted_index = clipscores_sorted_index[::-1]
        
    # get the top k clipscores
    clipscores_sorted_index = clipscores_sorted_index[start_index : start_index + k]
    # sort the clipscores
    # clipscores_sorted_index.sort()
    # get the top k small clipscores
    clipscores_sorted = clipscores[clipscores_sorted_index]
    # save the clipscores
    smallorlarge = 'small' if choose_small else 'large'
    
    extra_str = f'{extra_id_name}_top{k}_{smallorlarge}_from_{start_index}'
    
    if need_save:
        clipscores_sorted_path = os.path.join(save_path, f'clipscores_{extra_str}.npy')
        np.save(clipscores_sorted_path, clipscores_sorted)
        print(f'saving {k} clipscores with top {k} {smallorlarge} clipscores from {start_index} to {start_index + k}')
    
    
    # get the urls
    if urls is not None and captions is not None:
        urls_sorted = urls[clipscores_sorted_index]
        captions_sorted = captions[clipscores_sorted_index]
        visualize_images_captions(
            urls_sorted,
            captions_sorted,
            clip_scores=clipscores_sorted,
            num_show=5,
            extra_id_name=extra_str,
            save_path=save_path,
        )
    
    if need_save:
        uids_path_name = save_uids(
            index=clipscores_sorted_index,
            uids=uids,
            extra_str=extra_str,
            uids_path=uids_path,
        )
        print(f'saving {k} uids with top {k} {smallorlarge} clipscores from {start_index} to {start_index + k}, path: {uids_path_name}')
    
            
        
    
    return clipscores_sorted


def random_sample(
    clip_scores: np.ndarray,
    uids: np.ndarray,
    num_sample: int,
    save_path: str,
    uids_path: str,
):
    # randomly sample num_sample clipscores
    rs_index = np.random.choice(clip_scores.shape[0], num_sample, replace=False)
    
    ############# save the random sample clipscores ###########
    rs_clip_scores = clip_scores[rs_index]
    rs_clip_scores_path = os.path.join(save_path, f'clipscores_random{num_sample}.npy')
    np.save(rs_clip_scores_path, rs_clip_scores)
    print(f'saving random sample clipscores to {rs_clip_scores_path}')
    
    ############# save the random sample uids ###########
    rs_uids_path_name = save_uids(
        index=rs_index,
        uids=uids,
        extra_str=f'random{num_sample}',
        uids_path=uids_path,
    )
    print(f'saving random sample uids to {rs_uids_path_name}')
    
    return rs_index


def test_uids(
    uids: np.ndarray,
):
    """
    test if the uids has been the final state
    """
    uid = uids[0]
    # if it is the final state, it should be of the shape (u8, u8), else it should be str
    
    if isinstance(uid, str):
        return False
    else:
        return True
    
    

def save_uids(
    index: np.ndarray,
    uids: np.ndarray,
    extra_str: str = '',
    uids_path: str = '',
):
    # get the uids
    uids_sorted = uids[index]
    
    
    print(f'uids_sorted[0] = {uids_sorted[0]}')
    print(f'uids_sorted[-1] = {uids_sorted[-1]}')
    
    if test_uids(uids_sorted):
        print(f'uids has been the final state, no need to save')
        fianl_u8_u8_uids = uids_sorted
    else:
        fianl_u8_u8_uids = np.array(
            [
                (int(uid[:16], 16), int(uid[16:32], 16))
                for uid in uids_sorted
            ],
            np.dtype("u8,u8"),
        )
            
    ### final output
    print(f"sorting {len(fianl_u8_u8_uids)} uids, ori uids num = {len(uids)}")
    fianl_u8_u8_uids.sort()
    
    # save the uids
    uids_path_name = os.path.join(uids_path, f'uids_{extra_str}.npy')
    np.save(uids_path_name, fianl_u8_u8_uids)
    
    return uids_path_name




def main1(args):
    
    """
    function: 
    select the top k uids with the top k clipscores (or the bottom k clipscores) starting from start_index for the clipscore filtered dataset with given fraction
    and then visualize the botttom / top k clipscores and the corresponding image/caption pairs
    finally save the uids and clipscores
    """

    dataset_name = args.dataset_name
    save_path = os.path.join(args.files_path, dataset_name, 'save')
    fraction = args.fraction
    text_or_image = args.text_or_image
    need_save = args.need_save 
    
    
    
    extra_id_name = f'f{fraction}'
    extra_id_name += f'_text' if text_or_image == 'text' else ''

    
    #### load clipscores ####
    clipscores_path = os.path.join(save_path, f'clipscores_{extra_id_name}.npy')
    clipscores = np.load(clipscores_path, allow_pickle=True)
    print(f'load clipscores from {clipscores_path} with {clipscores.shape[0]} entries')


    n = len(clipscores) # n_total = 9247550 # total number of the 1.0 fraction clipscores
    print(f'n = {n}')

    
    #### load uids ####
    uids_path = os.path.join(args.files_path, dataset_name, 'uids')
    uids_file_path = os.path.join(uids_path, f'uids_{extra_id_name}.npy')
    uids = np.load(uids_file_path, allow_pickle=True)
    print(f'load uids from {uids_file_path} with {uids.shape[0]} entries')
    
    
    #### select top k uids ####
    num_select = args.num_sample
    start_index = int(n * args.start_index_ratio)
    choose_small = args.choose_small


    ### load urls and captions for visualization ###
    urls, captions = None, None
    if args.vis_urls_captions:
        urls_path = os.path.join(save_path, f'urls_{extra_id_name}.npy')
        if os.path.exists(urls_path):
            urls = np.load(urls_path, allow_pickle=True)
            print(f'load urls from {urls_path}')
        else:
            print(f'no urls file found in {urls_path}')
            
        captions_path = os.path.join(save_path, f'captions_{extra_id_name}.npy')
        if os.path.exists(captions_path):
            captions = np.load(captions_path, allow_pickle=True)
            print(f'load captions from {captions_path}')
        else:
            print(f'no captions file found in {captions_path}')
    else:
        print(f'no need to load urls and captions for visualization')
    

    print(f'===================== need_save: {need_save}, choose_small:{choose_small} =====================')



    # # visualize initial clipscores
    # visualize_clipscores(
    #     clipscores=clipscores,
    #     fraction=fraction,
    #     save_path=save_path,
    #     name='_initial',
    #     bins=400
    # )


    clipscores_selected = select_top_k_uids(
        clipscores=clipscores,
        uids=uids,
        extra_id_name = extra_id_name,
        k=num_select,
        start_index= start_index,
        save_path=save_path,
        uids_path=uids_path,
        choose_small=choose_small,
        # add after 11.29.2023
        urls=urls,
        captions=captions,
        need_save=need_save,
    )

    # visualize the clipscores
    choose_str = 'small' if choose_small else 'large'

    visualize_clipscores(
        clipscores=clipscores_selected,
        # fraction=fraction,
        extra_id_name=extra_id_name,
        save_path=save_path,
        name=f'_top{num_select}_{choose_str}_from_{start_index}',
    )


def main2(args):
    
    dataset_name = args.dataset_name
    save_path = os.path.join(args.files_path, dataset_name, 'save')
    fraction = args.fraction
    text_or_image = args.text_or_image
    need_save = args.need_save 
    
    
    
    extra_id_name = f'f{fraction}'
    extra_id_name += f'_text' if text_or_image == 'text' else ''

    
    #### load clipscores ####
    clipscores_path = os.path.join(save_path, f'clipscores_{extra_id_name}.npy')
    clipscores = np.load(clipscores_path, allow_pickle=True)
    print(f'load clipscores from {clipscores_path} with {clipscores.shape[0]} entries')


    n = len(clipscores) 
    print(f'n = {n}')

    
    #### load uids ####
    uids_path = os.path.join(args.files_path, dataset_name, 'uids')
    uids_file_path = os.path.join(uids_path, f'uids_{extra_id_name}.npy')
    uids = np.load(uids_file_path, allow_pickle=True)
    print(f'load uids from {uids_file_path} with {uids.shape[0]} entries')
    
    # randomly sample num_sample clipscores
    # N is the num for clipscores_f1.0
    random_sample_index = random_sample(
        clip_scores=clipscores,
        uids=uids,
        num_sample=args.num_sample,
        save_path=save_path,
        uids_path=uids_path,
    )
    
    
    
    
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='cc12m', help='dataset name')
    parser.add_argument('--clipscores_path', type=str, default=None, help='clipscores path')
    parser.add_argument('--files_path', type=str, default='/homes/gws/ypwang61/nobackup/files/', help='files path')
    parser.add_argument('--fraction', type=float, default=1.0, help='fraction of embeddings to use')
    parser.add_argument('--text_or_image', type=str, default='image', help='text or image')
    parser.add_argument('--num_select_ratio', type=float, default=0.3, help='num_select_ratio')
    # parser.add_argument('--start_index', type=int, default=0, help='start_index')
    parser.add_argument('--start_index_ratio', type=float, default=0.0, help='start_index_ratio')
    parser.add_argument('--choose_small', type=int, default=0, help='choose the top k small clipscores if True, otherwise choose the top k large clipscores')
    parser.add_argument('--need_save', type=int, default=0, help='need_save clipscores and uids')
    parser.add_argument('--vis_urls_captions', type=int, default=0, help='visualize urls and captions')
    
    
    # random sample
    parser.add_argument('--num_sample', type=int, help='num_sample')
    args = parser.parse_args()

    for arg in vars(args):
        print(f'{arg}: \t{getattr(args, arg)}')
    
    main1(args)
    # main2(args)
    

    