
import pandas as pd
import json
import os
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import re


SEED = 42
np.random.seed(SEED)

NUM_PERIODS = 10
TRAIN_PERIODS = 8
VAL_PERIODS = 1
TEST_PERIODS = 1
SEQ_LEN = 10
MIN_CORE = 5
TRAIN_SAMPLE_SIZE = 65536
VAL_SAMPLE_SIZE = 5000
TEST_SAMPLE_SIZE = 5000

OUTPUT_DIR = "process_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

inter_df = pd.read_csv('ml-1m.inter', sep='\t')

inter_df.columns = [col.split(':')[0] for col in inter_df.columns]

item_df = pd.read_csv('ml-1m.item', sep='\t')

item_df.columns = [col.split(':')[0] for col in item_df.columns]

item_id_to_title = dict(zip(item_df['item_id'], item_df['movie_title']))
item_id_to_genre = dict(zip(item_df['item_id'], item_df['genre']))


inter_df = inter_df[inter_df['rating'] >= 4.0]


iteration = 0
while True:
    iteration += 1
    
    old_len = len(inter_df)

    user_counts = inter_df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= MIN_CORE].index
    inter_df = inter_df[inter_df['user_id'].isin(valid_users)]

    item_counts = inter_df['item_id'].value_counts()
    valid_items = item_counts[item_counts >= MIN_CORE].index
    inter_df = inter_df[inter_df['item_id'].isin(valid_items)]

    if len(inter_df) == old_len:
        break

item_df = item_df[item_df['item_id'].isin(inter_df['item_id'].unique())]

item_id_to_title = dict(zip(item_df['item_id'], item_df['movie_title']))
item_id_to_genre = dict(zip(item_df['item_id'], item_df['genre']))

inter_df = inter_df.sort_values('timestamp').reset_index(drop=True)

period_size = len(inter_df) // NUM_PERIODS
inter_df['period'] = inter_df.index // period_size
inter_df.loc[inter_df['period'] >= NUM_PERIODS, 'period'] = NUM_PERIODS - 1

for period in range(NUM_PERIODS):
    count = len(inter_df[inter_df['period'] == period])

train_df = inter_df[inter_df['period'] < TRAIN_PERIODS].copy()
val_df = inter_df[inter_df['period'] == TRAIN_PERIODS].copy()
test_df = inter_df[inter_df['period'] == TRAIN_PERIODS + VAL_PERIODS].copy()


def build_sequences_and_samples(df, dataset_name):
    df = df.sort_values(['user_id', 'timestamp'])
    user_sequences = defaultdict(list)
    for _, row in df.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        user_sequences[user_id].append(item_id)

    samples = []
    for user_id, seq in tqdm(user_sequences.items()):
        if len(seq) < SEQ_LEN + 1:
            continue

        for i in range(SEQ_LEN, len(seq)):
            history_items = seq[i - SEQ_LEN: i]
            target_item = seq[i]

            history_titles = [item_id_to_title.get(iid, '') for iid in history_items]
            history_titles = [t for t in history_titles if t]
            target_title = item_id_to_title.get(target_item, None)
            
            if target_title and len(history_titles) == SEQ_LEN:
                instruction = "Based on the user's historical movie preferences, recommend the next movie."
                input_text = "User watched: " + ", ".join([f'"{title}"' for title in history_titles])
                output_text = f'"{target_title}"'
                
                samples.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": output_text,
                    "user_id": str(user_id),
                    "history_item_id": [str(iid) for iid in history_items],
                    "target_item_id": str(target_item)
                })
    
    return samples

train_samples = build_sequences_and_samples(train_df)
val_samples = build_sequences_and_samples(val_df)
test_samples = build_sequences_and_samples(test_df)

if len(train_samples) > TRAIN_SAMPLE_SIZE:
    np.random.shuffle(train_samples)
    train_samples = train_samples[:TRAIN_SAMPLE_SIZE]

if len(val_samples) > VAL_SAMPLE_SIZE:
    np.random.shuffle(val_samples)
    val_samples = val_samples[:VAL_SAMPLE_SIZE]

if len(test_samples) > TEST_SAMPLE_SIZE:
    np.random.shuffle(test_samples)
    test_samples = test_samples[:TEST_SAMPLE_SIZE]

with open(f"{OUTPUT_DIR}/train.json", 'w', encoding='utf-8') as f:
    json.dump(train_samples, f, indent=2, ensure_ascii=False)

with open(f"{OUTPUT_DIR}/valid.json", 'w', encoding='utf-8') as f:
    json.dump(val_samples, f, indent=2, ensure_ascii=False)

with open(f"{OUTPUT_DIR}/test.json", 'w', encoding='utf-8') as f:
    json.dump(test_samples, f, indent=2, ensure_ascii=False)

test_df = pd.DataFrame(test_samples)
test_df.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)

all_genres = set()
for genre_str in item_df['genre']:
    genres = str(genre_str).split()
    all_genres.update(genres)

all_genres = sorted(list(all_genres))

genre_data = []
for _, row in item_df.iterrows():
    item_id = row['item_id']
    title = row['movie_title']
    genres = str(row['genre']).split()
    
    genre_dict = {}
    for genre in all_genres:
        genre_dict[genre] = 1 if genre in genres else 0
    genre_dict['Title'] = title
    genre_data.append(genre_dict)

genre_df = pd.DataFrame(genre_data)

cols = [g for g in all_genres] + ['Title']
genre_df = genre_df[cols]
genre_df.to_csv(f"{OUTPUT_DIR}/movies_genre.csv", index=False)

item_popularity = train_df['item_id'].value_counts().to_dict()

popularity_data = []
for _, row in item_df.iterrows():
    item_id = row['item_id']
    title = row['movie_title']
    pop_count = item_popularity.get(item_id, 0)
    popularity_data.append({
        'Title': title,
        'count': pop_count
    })

pop_df = pd.DataFrame(popularity_data)

pop_df = pop_df.sort_values('count', ascending=False).reset_index(drop=True)
items_per_group = len(pop_df) // 5
pop_df['Pop'] = pop_df.index // items_per_group
pop_df.loc[pop_df['Pop'] >= 5, 'Pop'] = 4

pop_df[['Title', 'Pop']].to_csv(
    f"{OUTPUT_DIR}/movie_pop_count_5.csv", index=False
)

for level in range(5):
    count = len(pop_df[pop_df['Pop'] == level])
    avg_interactions = pop_df[pop_df['Pop'] == level]['count'].mean()

with open(f"{OUTPUT_DIR}/movies.dat", 'w', encoding='utf-8') as f:
    for _, row in item_df.iterrows():
        item_id = row['item_id']
        title = row['movie_title']
        genre = row['genre']
        f.write(f"{item_id}::{title}::{genre}\n")


stats = {
    "dataset": "MovieLens-1M",
    "preprocessing_method": "5-core filtering + 8:1:1 time-based split + sampling",
    "time_based_split": True,
    "num_periods": NUM_PERIODS,
    "train_periods": f"0-{TRAIN_PERIODS-1}",
    "val_periods": str(TRAIN_PERIODS),
    "test_periods": str(TRAIN_PERIODS + VAL_PERIODS),
    "seq_len": SEQ_LEN,
    "min_core": MIN_CORE,
    "train_sample_size": TRAIN_SAMPLE_SIZE,
    "val_sample_size": VAL_SAMPLE_SIZE,
    "test_sample_size": TEST_SAMPLE_SIZE,
    "total_users": int(inter_df['user_id'].nunique()),
    "total_items": len(item_df),
    "total_positive_interactions": len(inter_df),
    "train_interactions": len(train_df),
    "val_interactions": len(val_df),
    "test_interactions": len(test_df),
    "train_samples": len(train_samples),
    "val_samples": len(val_samples),
    "test_samples": len(test_samples),
    "genres": all_genres,
    "num_genres": len(all_genres),
    "popularity_distribution": {
        f"level_{i}": {
            "count": int(len(pop_df[pop_df['Pop'] == i])),
            "avg_interactions": float(pop_df[pop_df['Pop'] == i]['count'].mean())
        } for i in range(5)
    }
}

with open(f"{OUTPUT_DIR}/dataset_stats.json", 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)


def compute_popularity_distribution(train_samples, item_titles):

    title_counts = {}
    total_count = 0
    
    for sample in tqdm(train_samples):
        input_text = sample['input']
        history_titles = re.findall(r'"([^"]*)"', input_text)

        target_title = sample['output'].strip('"')

        for title in history_titles:
            if title:
                title_counts[title] = title_counts.get(title, 0) + 1
                total_count += 1
        
        if target_title:
            title_counts[target_title] = title_counts.get(target_title, 0) + 1
            total_count += 1

    popularity_dict = {}
    for title, count in title_counts.items():
        popularity_dict[title] = {
            "count": count,
            "p_data": count / total_count
        }

    min_count = 1
    for title in item_titles:
        if title not in popularity_dict:
            popularity_dict[title] = {
                "count": 0,
                "p_data": min_count / (total_count + min_count * len(item_titles))
            }
    
    return popularity_dict

def generate_p_target_distributions(popularity_dict, alpha_list):

    
    p_target_dict = {}
    
    for alpha in tqdm(alpha_list):
        p_alpha = {}
        Z_alpha = 0.0
        
        for title, pop_info in popularity_dict.items():
            p_data = pop_info['p_data']
            p_alpha[title] = p_data ** alpha
            Z_alpha += p_alpha[title]

        p_target = {}
        for title, val in p_alpha.items():
            p_target[title] = val / Z_alpha if Z_alpha > 0 else 1.0 / len(p_alpha)
        
        p_target_dict[alpha] = p_target
    
    return p_target_dict

all_item_titles = list(item_id_to_title.values())

popularity_dict = compute_popularity_distribution(train_samples, all_item_titles)

with open(f"{OUTPUT_DIR}/popularity.json", 'w', encoding='utf-8') as f:
    json.dump(popularity_dict, f, indent=2, ensure_ascii=False)

alpha_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
p_target_dict = generate_p_target_distributions(popularity_dict, alpha_list)

for alpha, p_target in p_target_dict.items():
    output_file = f"{OUTPUT_DIR}/p_target_alpha_{alpha:.2f}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "alpha": alpha,
            "p_target": p_target
        }, f, indent=2, ensure_ascii=False)
