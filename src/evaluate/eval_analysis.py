from argparse import Namespace
import json
import torch, os, sys
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from seqeval.metrics.sequence_labeling import get_entities
import pandas as pd 
from collections import Counter


sys.path.append("../")
sys.path.append("src/")
from utils import get_node_info

def slot_f1_score(pred_entities, true_entities):
    pred_entities = set(pred_entities)
    true_entities = set(true_entities)
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0
    return p, r, score

def stats_entities(pred_entities, true_entities, topk=10):
    n_sample =len(set([e.split(":")[0] for e in true_entities]))
    pred_entities = set(pred_entities) 
    true_entities = set(true_entities)
    correct_pred = true_entities & pred_entities
    wrong_entities =[":".join(e.split(":")[1:3]) for e in pred_entities.difference(correct_pred)]
    true_entities =[":".join(e.split(":")[1:3]) for e in true_entities]

    wrong_counter = Counter(wrong_entities).most_common(topk)
    true_counter = Counter(true_entities)
    # wrong_entities_rate = [(e[0], e[1]/ true_counter[e[0]] if true_counter[e[0]] > 0 else 0) for e in wrong_counter ]
    wrong_entities_rate = [(e[0], e[1]/ n_sample ) for e in wrong_counter ]
    return wrong_counter, wrong_entities_rate

def parse_output(_querier, output_str, sentence):
    from tree import Tree
    try: 
        # sentence = " ".join(output_str.split("\n")[0].replace("Step 1: ", "").strip().split(" ")[1:-1])
        cot_steps = _querier.format_output(output_str, "@#@#@$")
        pred_lg = _querier.restore_full_logical_form(cot_steps, sentence)
        return Tree(pred_lg)
    except:
        return None

def _load_meta_info(file_prediction):
    from query_llm import querier_class_map
    from tree import Tree
    
    print(f"- Eval {file_prediction}!")
    folder_prediction = file_prediction[:-len(file_prediction.split("/")[-1])] 
    
    train_arg_path = f"{folder_prediction}/training_args.bin"
    query_arg_path = f"{folder_prediction}/config.json"
    if os.path.exists(train_arg_path):
        infor = file_prediction.split("/")[-2].split("_")
        if "atis_flatten" in file_prediction:
            data_name = infor[0]+"_"+infor[1]
            querier_class_name = infor[5]
            llm_model = infor[2]
            top_k = int(infor[6].replace("shot",""))
        else:
            data_name = infor[0] 
            querier_class_name = infor[4]
            llm_model = infor[1]
            top_k = int(infor[5].replace("shot",""))
            
        d_type = file_prediction.split("/")[-1].split("_")[1]
        if d_type == 'valid': d_type = "dev"
        
        # prediction
        all_data = json.load(open(file_prediction))
            
    elif os.path.exists(query_arg_path):
        query_args = json.load(open(query_arg_path))
        querier_class_name = query_args['query_type']
        data_name = query_args["path_folder_data"].split("/")[5]
        d_type = query_args['data_type']
        all_data_tmp = json.load(open(file_prediction))
        all_data = {'detail_pred': [None]*len(all_data_tmp)}
        llm_model = query_args['model'].split("/")[-1]
        top_k = query_args['top_k_demos']
        for e in all_data_tmp:
            if 'q_output_raw' in e:
                all_data['detail_pred'][e['id']-1] = ["\n".join(e['q_output_raw'].split("\n\n")[top_k].split("\n")[1:])] if e['q_output_raw'] != "Err" else [""]
            elif 'q_output' in e:
                # if "\n" not in e['q_output']:
                all_data['detail_pred'][e['id']-1] = [e['q_output']]
                # else:
                #     all_data['detail_pred'][e['id']-1] = ["\n".join(e['q_output'].split("\n\n")[top_k].split("\n")[1:])]
                
        
    querier_class_name = querier_class_name.replace("cotv2", 'cot') 
    querier_class = querier_class_map.get(querier_class_name)
    querier = querier_class(Namespace(path_folder_data=None), None, None)
    
    # gold_data
    gold_data_raw = json.load(open(f"./data/{data_name}/mrc-ner.{d_type}"))
    gold_data = [e['org_label'] for e in gold_data_raw ]
    sentence_data = [e['context'] for e in gold_data_raw ]
    
    # if 'Meta-Llama-3' in file_prediction:
    #     from transformers import AutoTokenizer
    #     tokenizer = AutoTokenizer.from_pretrained(folder_prediction, trust_remote_code=True,use_fast=False)
    #     gold_data = [e.replace("]", "] ").replace("  ", " ").strip() for e in tokenizer.batch_decode(tokenizer(gold_data, padding='longest',truncation='longest_first')['input_ids'], skip_special_tokens=True, cleanup_tokenization_spaces=False)]
    #     sentence_data = tokenizer.batch_decode(tokenizer(sentence_data, padding='longest',truncation='longest_first')['input_ids'], skip_special_tokens=True, cleanup_tokenization_spaces=False)
    
    # parsing prediction process  
    pred_schemas, gold_schemas = [], []
    pred_full_lg, gold_full_lg = [], []
    all_pred_node_infors, all_gold_node_infors = [], []
    depth_levels = []
    for idx, e in enumerate(all_data['detail_pred']): 
        pred_parsed_tree = parse_output(querier, e[0], sentence_data[idx])
        pred_full_lg.append(str(pred_parsed_tree) if pred_parsed_tree is not None else f"Err {idx}")
        all_pred_node_infors += [f"{idx}:{e[2]}:{e[0]}" for e in get_node_info(pred_parsed_tree)] if pred_parsed_tree is not None else []
        pred_schemas.append(querier.get_lg_tree(pred_parsed_tree) if pred_parsed_tree is not None else f"Err {idx}")
        
        gold_parsed_tree = Tree(gold_data[idx])
        depth_levels.append(gold_parsed_tree.root.get_depth_level(cur_level=-2))
        gold_full_lg.append(str(gold_parsed_tree))
        all_gold_node_infors += [f"{idx}:{e[2]}:{e[0]}" for e in get_node_info(gold_parsed_tree)]  if gold_parsed_tree is not None else []
        gold_schemas.append(querier.get_lg_tree(gold_parsed_tree)) 
        # gold_parsed_tree2 = parse_output(querier, e[1])
        # gold_schemas2.append(querier.get_lg_tree(gold_parsed_tree2) if gold_parsed_tree2 is not None else f"Err gold {idx}")
        
    return querier, data_name, d_type, all_data, top_k, llm_model, sentence_data, gold_data,\
        pred_schemas, gold_schemas, pred_full_lg, gold_full_lg, all_pred_node_infors, all_gold_node_infors, depth_levels

def depth_analysis(depth_levels, metric_results):
    analysis_result = {}
    analysis_return = {}
    for d, correct in zip(depth_levels, metric_results):
        if d  not in analysis_result:
            analysis_result[d] = []
            analysis_return[d] = {}
        analysis_result[d].append(correct)
        
    for d, v in analysis_result.items():
        analysis_return[d]['count'] = len(v)
        analysis_return[d]['correct'] = sum(v)
        analysis_return[d]['performance'] = sum(v) / len(v) *100
        
    return analysis_return
lst_entities = None
def get_eval_logic_schema(file_prediction):
    querier, data_name, d_type, all_data, top_k, llm_model, sentence_data, gold_data, \
        pred_schemas, gold_schemas, pred_full_lg, gold_full_lg, all_pred_node_infors, all_gold_node_infors, depth_levels = _load_meta_info(file_prediction)
    
    sem_frame_count_true = sum([a == b for a,b in zip (pred_schemas, gold_schemas)])
    em_values = [a == b for a,b in zip (pred_full_lg, gold_full_lg)]
    em_count_true = sum(em_values)
    slot_p, slot_r, slot_f1 = slot_f1_score(all_pred_node_infors, all_gold_node_infors)
    
    wrong_entities, wrong_entities_rate = stats_entities(all_pred_node_infors, all_gold_node_infors, topk=100)
    # cached one system for applying to other system 
    global lst_entities
    topk=10
    if lst_entities == None:
        lst_entities = [e[0] for e in wrong_entities_rate][:topk]
        wrong_entities, wrong_entities_rate = wrong_entities[:topk], wrong_entities_rate[:topk]
    else:
        wrong_entities, wrong_entities_rate = [e for e in wrong_entities if e[0] in lst_entities], [e for e in wrong_entities_rate if e[0] in lst_entities]
        
        
    wrong_entities_rate = [ ( e[0].replace("_", "\_") , e[1]) for e in wrong_entities_rate]
    print(" ".join([f'({e[0]}, {e[1]*100})' for e in wrong_entities_rate]))
    print(','.join([e[0] for e in wrong_entities_rate]))
    
    # print(json.dumps(list(zip(wrong_entities, wrong_entities_rate)), indent=2))
    
    em_depth_analysis = depth_analysis(depth_levels, em_values)
    # print(json.dumps(sorted(em_depth_analysis.items()), indent=2))
    eval_metrics = {
        'data_name': data_name,
        'd_type': d_type,
        'k_demonstrations': top_k,
        "file_prediction": file_prediction, 
        "querier_class": querier.__class__.__name__,
        "llm_model": llm_model,
        "sem_frame_count_true": sem_frame_count_true, 
        "em_count_true": em_count_true,
        'sem_frame_acc': sem_frame_count_true / len(pred_schemas),
        'em': em_count_true / len(gold_full_lg),
        "slot_p": slot_p, 
        "slot_r": slot_r, 
        "slot_f1": slot_f1, 
    }
    json.dump({
        'eval_metrics': eval_metrics,
        'detail_pred': [{"idx":i, 
                         "pred_logic": pred_full_lg[i], 
                         "gold_logic": gold_full_lg[i], 
                         'correct': pred_full_lg[i]==gold_full_lg[i]
                         } for i in range(len(gold_full_lg))]},
               open(file_prediction.replace('.json', '_compare_2_gold.json'), "wt"), indent=2)
    return eval_metrics

    
def get_eval_metrics():
    
    path_result_files = [
        # # # top 
        './finetuned_llm/atis_flatten_Llama-2-7b-hf_ep3_lrs-cosine3e-4_cotv2dfs_0shot_r32_L1024_seed42_rc_0shot_secomb/result_valid_step-195.json',
        './data/top/CoT/test/out/gpt-4o_cotdfs_robertapair_all_top_10_sim_rouge_chunk_1:1_semcomb_False_openai-data/result.json',
         
        # atis   
        # './finetuned_llm/atis_flatten_Llama-2-7b-hf_ep3_lrs-linear3e-4_base_0shot_r32/result_valid_step-111.json',
        './data/atis_flatten/dev/out/gpt-4o-mini_cotdfs_robertapair_all_top_10_sim_rouge_chunk_1:1_semcomb_False_openai-data/result.json',
        
        # # # snips 
    ]
     
    
    results = [get_eval_logic_schema(path_result) for path_result in path_result_files]
    df = pd.DataFrame(data=results)
    print(df)
    df.to_csv("./finetuned_llm/result_analysis.csv", sep=",")
    
def compare_system(file_name1, *lst_sys):
    path_folder_out = './finetuned_llm' 
    path_file_out  = f"{path_folder_out}/{file_name1}.json"
    
    results= {'shared_wrong_count': None, 'wrong_preds': {}}
    shared_wrong_idx = None
    for sys_out in lst_sys:
        all_infors = _load_meta_info(sys_out)
        id1 = sys_out.split("/")[-2]
        pred_full_lg, gold_full_lg = all_infors[10], all_infors[11]
        depth_levels = all_infors[12]
        em_true = [a == b for a,b in zip (pred_full_lg, gold_full_lg)]
        wrong_preds_idx = [idx for idx in range(len(gold_full_lg)) if not em_true[idx]]
        if shared_wrong_idx == None:
            shared_wrong_idx = set(wrong_preds_idx)
        else:
            shared_wrong_idx = shared_wrong_idx.intersection(set(wrong_preds_idx))
        results[sys_out] = {
            'em': sum(em_true) / len(pred_full_lg),
            'em_count': sum(em_true) ,
            'total_pred': len(gold_full_lg),
            'wrong_count': len(wrong_preds_idx)
        } 
        for idx in wrong_preds_idx:
            if idx not in results['wrong_preds']:
                results['wrong_preds'][idx] = {
                    'gold': gold_full_lg[idx]  
                }
            results['wrong_preds'][idx][id1] = pred_full_lg[idx]
    results['shared_wrong_count'] = len(shared_wrong_idx)
    
    json.dump(results, open(path_file_out, 'wt'), indent=2)
    return  

    
if __name__=="__main__": 
    get_eval_metrics()