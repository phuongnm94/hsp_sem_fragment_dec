import os
import random
from utils import *
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import argparse
from torchmetrics.text.rouge import ROUGEScore
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
 

from torchmetrics.functional.text.rouge import rouge_score

rouge_params = {'rouge_keys': ('rouge1', 'rouge2', 'rougeL'), 'use_stemmer': True}
# rouge_func = ROUGEScore( normalizer=None,)
rouge_func = rouge_score



 
stop_words = set(stopwords.words('english'))
 
def split_sentence_to_words(input_sent):
    word_tokens = word_tokenize(input_sent) 
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    results = [] + filtered_sentence
    for n_gram in range(2, 5):
        if len(filtered_sentence) >= n_gram:
            for i_w in range(len(filtered_sentence) - n_gram + 1):
                results.append(" ".join(filtered_sentence[i_w: i_w+n_gram]))
    return set(results)

# Load data
def load_data(args):
    path_folder_data, prefix = args.path_folder_data, args.data_prefix
    train_data = read_json_file(f"{path_folder_data}/{prefix}mrc-ner.train") #  if args.predicate_retriever == "oracle" else read_json_file(f"data/top/mrc-ner-{args.predicate_retriever}.train")
    dev_data = read_json_file(f"{path_folder_data}/{prefix}mrc-ner.{args.data_type}") #  if args.predicate_retriever == "oracle" else read_json_file(f"data/top/mrc-ner-{args.predicate_retriever}.dev")
    train_entity_levels = [item["entity_level"] for item in train_data]
    train_utterances = [item["context"] for item in train_data]
    train_golds = [item["org_label"] for item in train_data]
    
    if args.predicate_retriever=='wordngram':
        train_labels_list = [split_sentence_to_words(item["context"]) for item in train_data] # if args.predicate_retriever == "oracle" else [item["predicted_predicates"] for item in train_data]
    else:
        train_labels_list = [get_list_labels(item["org_label"]) for item in train_data] # if args.predicate_retriever == "oracle" else [item["predicted_predicates"] for item in train_data]
    label_set = set().union(*train_labels_list)

    args.num_sample = int(args.num_sample) if args.num_sample != "all" else args.num_sample
    if args.num_sample == "all":
        dev_utterances = [item["context"] for item in dev_data]
        dev_golds = [item["org_label"] for item in dev_data]
        if not os.path.isfile(f"{path_folder_data}/golds_{args.data_type}_{args.num_sample}.tsv"):
            save_list_to_tsv(f"{path_folder_data}/golds_{args.data_type}_{args.num_sample}.tsv", dev_golds)
            
        if args.predicate_retriever == "oracle":
            dev_labels_list = [get_list_labels(item["org_label"]) for item in dev_data] 
        elif args.predicate_retriever=='nopredicate':
            dev_labels_list = [[] for l in dev_utterances]
        elif args.predicate_retriever=='wordngram':
            dev_labels_list = [split_sentence_to_words(item["context"]) for item in dev_data] 
        else:
            with open(f'{path_folder_data}/{args.data_type}.sent_predicate') as f:
                dev_labels_list = [[predicate for predicate in l.strip().split("[sep]")[-1].split(" ")if len(predicate) > 0] for l in f.readlines()]
    else:
        if not os.path.isfile(f"{path_folder_data}/{args.num_sample}_random_{args.data_type}_sample_ids.txt"):
            dev_ids = random.sample(range(len(dev_data)), args.num_sample)
            with open(f'{path_folder_data}/{args.num_sample}_random_{args.data_type}_sample_ids.txt', 'w') as file:
                file.write(','.join(map(str, dev_ids)))
        else:
            with open(f'{path_folder_data}/{args.num_sample}_random_{args.data_type}_sample_ids.txt', 'r') as file:
                dev_ids = file.read().split(',')
                dev_ids = [int(item) for item in dev_ids]
        dev_utterances = [item["context"] for idx, item in enumerate(dev_data) if idx in dev_ids]
        dev_golds = [item["org_label"] for idx, item in enumerate(dev_data) if idx in dev_ids]
        # dev_labels_list = [get_list_labels(item["org_label"]) for idx, item in enumerate(dev_data) if idx in dev_ids] if args.predicate_retriever == "oracle" else [item["predicted_predicates"] for idx, item in enumerate(dev_data) if idx in dev_ids]
        with open(f'{path_folder_data}/predicate_prediction/{args.data_type}.sent_predicate') as f:
            dev_labels_list = [[predicate for predicate in l.strip().split("[sep]")[-1].split(" ")if len(predicate) > 0] for l in f.readlines()]
        if not os.path.isfile(f"{path_folder_data}/golds_{args.data_type}_{args.num_sample}.tsv"):
            save_list_to_tsv(f"{path_folder_data}/golds_{args.data_type}_{args.num_sample}.tsv", dev_golds)
    
    return train_utterances, train_labels_list, dev_utterances, dev_labels_list, train_entity_levels, label_set, train_golds

def sbert_retriver(train_utterances, train_ids, query_uterrance, cached_sim_scores):
    new_train_ids = []
    new_train_utterances = []
    for idx, check_id in enumerate(train_ids):
        if check_id not in cached_sim_scores:
            new_train_ids.append(check_id)
            new_train_utterances.append(train_utterances[idx])
            
    if len(new_train_utterances) > 0:
        corpus_embeddings = embedder.encode(new_train_utterances, convert_to_tensor=True)
        query_embedding = embedder.encode(query_uterrance, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0].cpu().numpy()
        
        for idx, check_id in enumerate(new_train_ids):
            cached_sim_scores[check_id] = cos_scores[idx]
    
    cos_scores = np.array([cached_sim_scores[check_id] for check_id in train_ids])
    
    return train_ids[np.argmax(cos_scores)]

def rouge_retriver(train_utterances, train_ids, query_uterrance, cached_sim_scores):
    new_train_ids = []
    new_train_utterances = []
    for idx, check_id in enumerate(train_ids):
        if check_id not in cached_sim_scores:
            new_train_ids.append(check_id)
            new_train_utterances.append(train_utterances[idx])
            
    if len(new_train_utterances) > 0:
        sim_scores = [rouge_func(query_uterrance, new_train_utterance, **rouge_params) for new_train_utterance in new_train_utterances]
        sim_scores = [sum([e['rouge1_fmeasure'], e['rouge2_fmeasure'], e['rougeL_fmeasure']]).item()/3.0 for e in sim_scores]
        
        for idx, check_id in enumerate(new_train_ids):
            cached_sim_scores[check_id] = sim_scores[idx]
    
    cos_scores = np.array([cached_sim_scores[check_id] for check_id in train_ids])
    
    return train_ids[np.argmax(cos_scores)]

def jaccard_similarity(A, B):
    #Find intersection of two sets
    nominator = A.intersection(B)
    #Find union of two sets
    denominator = A.union(B)
    if len(denominator) == 0:
        return -1
    
    #Take the ratio of sizes
    similarity = len(nominator)/len(denominator)
    return similarity

def find_samples(args, cur_labels, query_uterrance, train_labels_list, train_utterances, no_predicate=False,
                 avoiding_indexes=[]):
    expected_num = args.top_k_demos
    visited = [0] * len(train_labels_list)
    cur_labels = set([item for item in cur_labels if item in label_set])
    uncover_labels = cur_labels
    result = []
    cached_sim_scores = {} 
    while (len(uncover_labels) != 0 or no_predicate==True) and len(result) < expected_num:
        
        # Find the most coverest samples -> highest jaccard score
        max_jc = 0
        max_union_list = []
        for idx in range(len(train_labels_list)):
            if idx in avoiding_indexes:
                continue
            tmp = jaccard_similarity(uncover_labels, train_labels_list[idx])
            if tmp > max_jc and visited[idx] == 0:
                max_jc = tmp
                max_union_list = []
            if tmp == max_jc and visited[idx] == 0:
                max_union_list.append(idx)

        # Extract all samples with the highest jaccard score
        retrieving_train_ids = []
        max_jc = 0
        for idx in max_union_list:
            tmp = jaccard_similarity(cur_labels, train_labels_list[idx])
            if tmp > max_jc and visited[idx] == 0:
                max_jc = tmp
                retrieving_train_ids = []
            if tmp == max_jc and visited[idx] == 0:
                retrieving_train_ids.append(idx)
        
        
        # Retrieve top-1 from the extracted samples
        retrieving_train_utterances = [train_utterances[idx] for idx in max_union_list] 
        if args.sim_method == 'rouge':
            retrived_id = rouge_retriver(retrieving_train_utterances, max_union_list, query_uterrance, cached_sim_scores)
        elif args.sim_method == 'sbert':
            retrived_id = sbert_retriver(retrieving_train_utterances, max_union_list, query_uterrance, cached_sim_scores)

        # Set visited to covered predicates
        visited[retrived_id] = 1
        uncover_labels = uncover_labels - train_labels_list[retrived_id]

        # Reset covered predicates
        if len(uncover_labels) == 0: 
            uncover_labels = cur_labels
        result.append(retrived_id)
    return result


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Process ...')
    parser.add_argument('--num_sample', type=str, default="all")
    parser.add_argument('--predicate_retriever', type=str, default="oracle", help=" choose from {oracle, nopredicate, robertapair, wordngram}")
    parser.add_argument('--sim_method', type=str, default="rouge", help=" choose from {rouge, sbert}")
    parser.add_argument('--top_k_demos', type=int, default=20)
    parser.add_argument('--path_folder_data', type=str, default='data/top/low_resource/reminder/')
    parser.add_argument('--data_prefix', type=str, default='reminder_25spis-')
    parser.add_argument('--data_type', type=str, default='dev')

    args, unknown = parser.parse_known_args()

    train_utterances, train_labels_list, dev_utterances, dev_labels_list, train_entity_levels, label_set, train_golds = \
        load_data(args) 

    if args.sim_method == 'sbert':
        # Corpus with example sentences
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        corpus_embeddings = embedder.encode(train_utterances, convert_to_tensor=True)

    print("Ranking ...")
    result = []
    for idx, query in enumerate(tqdm(dev_utterances[:])):
        tmp = {}
        tmp["q_input"] = query
        tmp["q_id"] = idx
        tmp["candidates"] = []
        avoiding_indexes = []
        if args.data_type == 'train':
            avoiding_indexes = [idx]
        top_k_ids = find_samples(args, dev_labels_list[idx], dev_utterances[idx], train_labels_list, 
                                train_utterances, no_predicate=True, avoiding_indexes=avoiding_indexes)
        for id in top_k_ids:
            tmp["candidates"].append([train_utterances[id], train_golds[id], train_entity_levels[id]])
        result.append(tmp)
        tmp["candidate_ids"] = top_k_ids

    write_json(result, f"{args.path_folder_data}/{args.predicate_retriever}_{args.num_sample}_{args.sim_method}.{args.data_type}.json")
