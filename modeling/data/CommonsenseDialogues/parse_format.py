import json
import random
random.seed(1234)
from modeling.data.load import load_data, save_data
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np
from tqdm import tqdm
from collections import Counter

cs_prefix = {
    "What could have caused the last thing said to happen?": "The previous dialogue turn was caused by",
    "How does Listener feel because of what Speaker just said?": "The listener feels",
    "How is Speaker feeling after what they just said?": "The speaker feels",
    "What might happen after what Speaker just said?": "Next,",
    "What is a likely characteristic of Speaker based on what they just said?": "The speaker is",
    "What will Listener want to do next based on what Speaker just said?": "The listener wants",
    "What is a breakdown of the last thing said into a series of required subevents?": "The previous dialogue turn depends on",
    "What is an emotion or human drive that motivates Speaker based on what they just said?": "The speaker is motivated",
    "What prerequisites are required for the last thing said to occur?": "The previous dialogue turn requires",
    "What does Speaker want to do next?": "The speaker wants"
}

SENTENCEBERT = SentenceTransformer('all-mpnet-base-v2')
def sentencebert(cs):
    cs_embed = SENTENCEBERT.encode(cs, convert_to_tensor=True)
    cosine_scores = cos_sim(cs_embed, cs_embed).cpu().numpy()
    return cosine_scores

def greedy_diversify(groups, num_in_group=5):
    cs_ls = [item for group, items in groups for item in items[:num_in_group]]
    cs_to_cs_scores = sentencebert(cs_ls)
    selected_cs = []
    selected_cs_idx = []
    selected_cs_group = []
    # cold start: pick item with lowest avg similarity to all others in other types to initialize set {selected_cs}
    avg_scores = []
    for row_idx, row in enumerate(cs_to_cs_scores):
        group_idx = row_idx // num_in_group
        group_start_idx = group_idx * num_in_group
        group_end_idx = group_start_idx + (num_in_group - 1)
        scores_for_nongroup = np.concatenate([row[:group_start_idx], row[group_end_idx+1:]])
        avg_sim_score = np.mean(scores_for_nongroup)
        avg_scores.append(avg_sim_score)
    min_cs_idx = np.argmin(avg_scores)
    min_cs = cs_ls[min_cs_idx]
    selected_cs.append(min_cs)
    selected_cs_idx.append(min_cs_idx)
    selected_cs_group.append(min_cs_idx // num_in_group)
    # iteratively pick item with lowest avg similarity to all in {selected_cs} and add to {selected_cs}
    while len(selected_cs) < len(groups):
        avg_scores = []
        for row_idx, row in enumerate(cs_to_cs_scores):
            group_idx = row_idx // num_in_group
            if group_idx not in selected_cs_group:
                scores = [row[i] for i in selected_cs_idx]
                avg_score = np.mean(scores)
                avg_scores.append(avg_score)
            else:
                avg_scores.append(100)
            min_cs_idx = np.argmin(avg_scores)
        min_cs_idx = np.argmin(avg_scores)
        min_cs = cs_ls[min_cs_idx]
        selected_cs.append(min_cs)
        selected_cs_idx.append(min_cs_idx)
        selected_cs_group.append(min_cs_idx // num_in_group)
    selected_cs_with_groups = [(groups[group][0], selection.replace(f'{cs_prefix[groups[group][0]]} ', '')) for group, selection in zip(selected_cs_group, selected_cs)]
    return selected_cs_with_groups

def shuffle_and_resave(dir, file, cs_source='predicted', diversify=True, begin_at=None, stop_at=None):
    """
    (1) Shuffles order of CS types in turn dictionary to have different CS orderings when linearized into prompt
    (2) Greedily optimizes CS inference diversity between types

    Usage:

    shuffle_and_resave(
        dir='soda', 
        file='convosense_mono_dbs_minidev.pickle'
    )
    """
    data = load_data(dir=dir, file=file, all_cs=True)
    if stop_at is None:
        stop_at = len(data)
    if begin_at is None:
        begin_at = 0
    data = data[begin_at:stop_at]
    lengths = []
    num_in_groups = 5
    for d in tqdm(data, desc='Processing data...'):
        if cs_source == 'predicted':
            cs_items = list(d.turns[-1].beam_cs.items())
            if not all([len(items) >= num_in_groups for group, items in cs_items]):
                continue
            lengths.extend([len(items) for group, items in cs_items])
            if diversify:
                cs_items = greedy_diversify(cs_items, num_in_group=num_in_groups)
            random.shuffle(cs_items)
            d.turns[-1].cs = dict(cs_items)
        elif cs_source == 'silver':
            cs_items = list(d.turns[-1].silver_beam_cs.items())
            if not all([len(items) >= num_in_groups for group, items in cs_items]):
                continue
            lengths.extend([len(items) for group, items in cs_items])
            if diversify:
                cs_items = greedy_diversify(cs_items, num_in_group=num_in_groups)
            random.shuffle(cs_items)
            d.turns[-1].silver_cs = dict(cs_items)
    print(json.dumps(Counter(lengths), indent=2))
    save_data(data, dir, f'{file[:file.index(".")]}_{cs_source}div5.json')



if __name__ == '__main__':

    shuffle_and_resave(
        dir='CommonsenseDialogues',
        file='reflect_shuffled_predictedcs_0_100.json',
        cs_source='predicted',
        diversify=True,
        # begin_at=0,
        # stop_at=100
    )


