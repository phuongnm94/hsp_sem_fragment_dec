
import argparse
import glob
import json
import re
import traceback 
from datasets import load_dataset
import datasets
from lightning import seed_everything
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments 
from trl import setup_chat_format
from peft import LoraConfig, AutoPeftModelForCausalLM
from trl import SFTTrainer
from transformers import pipeline
from tqdm import tqdm
from query_llm import LLMCoTBottomUpQuery, LLMCoTQuery, LLMMergingCoTQuery, LLMDfsCoTQuery, LLMBfsCoTQuery, LLMBaseQuery
from torch.utils.data import DataLoader
from transformers.trainer_utils import EvalLoopOutput
import random, os
import numpy as np
from datasets import Dataset

def set_random_seed(seed: int):
    """set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_raw_data_zeroshot_prompting(path_raw, path_data_out):
    system_message = """You are an text to Hierarchical Semantic parser. Users will ask you questions in English and you will generate a Hierarchical Semantic form based on that question. """
    raw_dat = json.load(open(path_raw))
    processed_data = []
    for sample in raw_dat:
        processed_data.append(json.dumps({
            "messages": [
            {"role": "system", "content": system_message },
            {"role": "user", "content": sample["context"]},
            {"role": "assistant", "content": sample["org_label"]}
            ]
        } ))
    with open(path_data_out, 'wt', encoding='utf-8') as f:
        f.write("\n".join(processed_data))
    return processed_data

def process_raw_data_fewshot_prompting(path_raw, candidates_path, path_data_out, topk=3):
    
    def few_shot_fotmat(candidates):
        prompting = "\n"
        for e in candidates:
            prompting = prompting + e[0] + "\n" + e[1] + "\n" 
        return prompting
    
    system_message = """You are an text to Hierarchical Semantic parser. Users will ask you questions in English and you will generate a Hierarchical Semantic form based on that question. """
    
    system_message += "For example:"
    all_candidates = json.load(open(candidates_path))
    
    raw_dat = json.load(open(path_raw))
    processed_data = []
    for i_s, sample in enumerate(raw_dat):
        cur_system_message = system_message + few_shot_fotmat(all_candidates[i_s]['candidates'][:topk])
        processed_data.append(json.dumps({
            "messages": [
            {"role": "system", "content": cur_system_message },
            {"role": "user", "content": sample["context"]},
            {"role": "assistant", "content": sample["org_label"]}
            ]
        } ))
    with open(path_data_out, 'wt', encoding='utf-8') as f:
        f.write("\n".join(processed_data))
    return processed_data

def process_raw_data_cotv2_prompting(path_raw, candidates_path, path_data_out, topk=3, query_class=LLMCoTQuery):
    def cot_format(sample_info):
        prompting = query_class.query_generate(sample_info, top_k_demos=topk)
        samples_inout_steps = [e.split("\n") for e in prompting.replace("Input: ", "").split("\n\n")[:-1]]
        prompt_conversation = []
        for e in samples_inout_steps:
            prompt_conversation.append({"role": "user", "content": e[0]})
            prompt_conversation.append({"role": "assistant", "content": "\n".join(e[1:])})
        return  prompt_conversation
    
    system_message = """You are an text to Hierarchical Semantic parser. Users will ask you questions in English and you will generate a Hierarchical Semantic form based on that question. """
    system_message += "For example:"
    all_candidates = json.load(open(candidates_path))
    
    raw_dat = json.load(open(path_raw))
    processed_data = []
    for i_s, sample in enumerate(raw_dat):
        cur_system_message = system_message
        gold_label = query_class.build_output(sample["context"], sample["org_label"])[0]
        processed_data.append(json.dumps({
            "messages": [{"role": "system", "content": cur_system_message }]
                            + cot_format(all_candidates[i_s])
                            +[
                                {"role": "user", "content": sample["context"]},
                                {"role": "assistant", "content": gold_label}
                            ]
        } ))
    with open(path_data_out, 'wt', encoding='utf-8') as f:
        f.write("\n".join(processed_data))
    return processed_data

def process_raw_data_cotv3_prompting(path_raw, candidates_path, path_data_out, topk=3, query_class=LLMCoTQuery):
    def cot_format(sample_info):
        prompting = query_class.query_generate(sample_info, top_k_demos=topk, add_final_step=True)
        samples_inout_steps = [e.split("\n") for e in prompting.replace("Input: ", "").split("\n\n")[:-1]]
        prompt_conversation = []
        for e in samples_inout_steps:
            prompt_conversation.append({"role": "user", "content": e[0]})
            prompt_conversation.append({"role": "assistant", "content": "\n".join(e[1:])})
        return  prompt_conversation
    
    system_message = """You are an text to Hierarchical Semantic parser. Users will ask you questions in English and you will generate a Hierarchical Semantic form based on that question. """
    system_message += "For example:"
    all_candidates = json.load(open(candidates_path))
    
    raw_dat = json.load(open(path_raw))
    processed_data = []
    for i_s, sample in enumerate(raw_dat):
        cur_system_message = system_message
        gold_label = query_class.build_output(sample["context"], sample["org_label"], add_final_step=True)[0]
        processed_data.append(json.dumps({
            "messages": [{"role": "system", "content": cur_system_message }]
                            + cot_format(all_candidates[i_s])
                            +[
                                {"role": "user", "content": sample["context"]},
                                {"role": "assistant", "content": gold_label}
                            ]
        } ))
    with open(path_data_out, 'wt', encoding='utf-8') as f:
        f.write("\n".join(processed_data))
    return processed_data

def process_raw_data_cot_prompting(path_raw, candidates_path, path_data_out, topk=3):
    def cot_format(sample_info):
        prompting = LLMCoTQuery.query_generate(sample_info, top_k_demos=topk)
        return "\n"+ "\n".join(prompting.split("\n")[:-2]).replace("Input: ", "")
    
    system_message = """You are an text to Hierarchical Semantic parser. Users will ask you questions in English and you will generate a Hierarchical Semantic form based on that question. """
    
    system_message += "For example:"
    all_candidates = json.load(open(candidates_path))
    
    raw_dat = json.load(open(path_raw))
    processed_data = []
    for i_s, sample in enumerate(raw_dat):
        cur_system_message = system_message + cot_format(all_candidates[i_s])
        gold_label = LLMCoTQuery.build_output(sample["context"], sample["org_label"])[0]
        processed_data.append(json.dumps({
            "messages": [
            {"role": "system", "content": cur_system_message },
            {"role": "user", "content": sample["context"]},
            {"role": "assistant", "content": gold_label}
            ]
        } ))
    with open(path_data_out, 'wt', encoding='utf-8') as f:
        f.write("\n".join(processed_data))
    return processed_data

def test_ft_model(ft_model_path, data_path, file_result_out=None, prompting_type="cotv2"):

    ft_model_id = ft_model_path
    peft_model_id = ft_model_id
    tokenizer = AutoTokenizer.from_pretrained(f"{ft_model_id}/")
    tokenizer.padding_side = 'left' 

    # Load Model with PEFT adapter
    model = AutoPeftModelForCausalLM.from_pretrained(
        peft_model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )
    # load into pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)


    def evaluate_2(samples):
        prompt = [pipe.tokenizer.apply_chat_template(
            sample["messages"][:-1], tokenize=False, add_generation_prompt=True) for sample in samples]
        outputs = pipe(prompt, max_new_tokens=256, do_sample=False,
                    eos_token_id=pipe.tokenizer.eos_token_id,
                    pad_token_id=pipe.tokenizer.pad_token_id,
                    batch_size=2)
        predicted_answers = [output[0]['generated_text']
                            [len(prompt[i]):].strip() for i, output in enumerate(tqdm(outputs))]
        gold_answers = [sample["messages"][-1]['content'] for sample in samples]
        input_sents = [sample["messages"][-2]['content'] for sample in samples]
        
        opts_cot_querier = argparse.Namespace(batch_size=8, cur_chunk=1, 
                                              data_type='prediction', 
                                              model='ft-llm', 
                                              num_chunk=1, 
                                              num_samples='all', 
                                              path_folder_data=None,
                                              predicate_retriever='robertapair', 
                                              query_type='cot', 
                                              sim_method='rouge', 
                                              top_k_demos=3)
        llm_querier = LLMCoTQuery(opts_cot_querier, model, tokenizer)
        
        def output_str_to_steps(sent, out_str):
            if "cot" in prompting_type:
                steps = [re.sub(r'Step \d+:', "", e.strip()).strip() for e in out_str.split("\n")] 
            else:
                logic_str_to_steps = llm_querier.build_output(sent, re.sub(r' {2,}', ' ', out_str.replace("]", "] ").strip()))[1]
                steps = logic_str_to_steps
            
            return steps
        
        all_gold_lg = []
        all_pred_lg = []
        for i, pred_steps in enumerate(predicted_answers):
            try:
                gold_lg =  llm_querier.restore_full_logical_form(output_str_to_steps(input_sents[i], gold_answers[i]), input_sents[i])
                pred_lg =  llm_querier.restore_full_logical_form(output_str_to_steps(input_sents[i], pred_steps), input_sents[i])
            except Exception as e:
                traceback.print_exc()
                print(e)
                pred_lg = ""
                
            all_gold_lg.append(gold_lg)
            all_pred_lg.append(pred_lg)
        correct_preds = [e == all_pred_lg[i]
                        for i, e in enumerate(all_gold_lg)]
        return list(zip(correct_preds, predicted_answers, gold_answers, prompt, all_pred_lg, all_gold_lg))


    eval_dataset = load_dataset(
        "json", data_files=data_path, split="train")

    # number_of_eval_samples = 1000
    # all_s = [s for s in tqdm(eval_dataset.shuffle().select(
    #     range(number_of_eval_samples)))]

    all_s = [s for s in eval_dataset]

    output_preds = evaluate_2(all_s[:])
    success_rate = [e[0] for e in output_preds]
    log_results = [{
        'correct': e[0],
        'q_output_steps': e[1],
        'gold_label_steps': e[2],
        'q_input': e[3],
        'pred_logic': e[4],
        'gold_logic': e[5],
    } for e in output_preds]

    if file_result_out is None:
        file_result_out = f'{ft_model_path}/predicted_results.json'
        
    json.dump(log_results, open(f'{file_result_out}', 'wt', encoding='utf-8'), indent=2)

    # compute accuracy
    accuracy = sum(success_rate)/len(success_rate)
    print(f"Accuracy: {accuracy*100:.2f}%")
    json.dump({'acc': accuracy,
               'wrong_pred': [e for e in log_results if not e['correct']]
               }, open(f'{file_result_out.replace(".json", ".performance.json")}', 'wt', encoding='utf-8'), indent=2)
    
    # free the memory again
    del model
    del pipe
    torch.cuda.empty_cache()
    
def formatting_prompts_func(samples):
    prompt_texts = [tokenizer.apply_chat_template(
             sample[:-1], tokenize=False, add_generation_prompt=True) for sample in samples["messages"]]
    
    print("=="*50)
    print(prompt_texts[-1])
    print("=="*50)
    return prompt_texts

def split_label(sample):
    tokenized_lb = tokenizer.encode(sample['messages'][-1]['content'], padding='max_length',max_length=300, truncation=True)
    sample['labels'] = tokenized_lb 
    return sample
 
class LLMSpTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_process_args = argparse.Namespace(
            packing=False,
            dataset_text_field=None,
            max_seq_length=kwargs.get('max_seq_length', None),
            formatting_func=formatting_prompts_func,
            num_of_sequences=kwargs.get('num_of_sequences', 1024),
            chars_per_token=kwargs.get('chars_per_token', 3.6),
            remove_unused_columns=kwargs.get('args').remove_unused_columns if kwargs.get('args') is not None else True,
            dataset_kwargs=kwargs.get('dataset_kwargs', {})
        )
        self.eval_dataset = self._process_raw_data(kwargs.get('eval_dataset', None))  
        print("len(eval dataset) = ",  len(self.eval_dataset))
    
    def _process_raw_data(self, dataset):
        dataset2 = dataset.map(split_label)
        dataset = self._prepare_dataset(
                dataset=dataset,
                tokenizer=self.tokenizer,
                packing=False,
                dataset_text_field=None,
                max_seq_length=self.data_process_args.max_seq_length,
                formatting_func=self.data_process_args.formatting_func,
                num_of_sequences=self.data_process_args.num_of_sequences,
                chars_per_token=self.data_process_args.chars_per_token,
                remove_unused_columns=self.data_process_args.remove_unused_columns,
                **self.data_process_args.dataset_kwargs, 
            )
        dataset = dataset.add_column('labels', dataset2['labels']) 
        return dataset 
    
    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        if eval_dataset is not None and "input_ids" not in eval_dataset.column_names and "labels" not in eval_dataset.column_names:
            # this is raw data which need to preprocess
            eval_dataset = self._process_raw_data(eval_dataset)
            
        return super().get_eval_dataloader(eval_dataset)
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only= None,
        ignore_keys = None,
        metric_key_prefix="eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        # if self.state.global_step % 400 != 0:
        #     return EvalLoopOutput(predictions=[], label_ids=[], metrics={f"{metric_key_prefix}_em": 82  }, num_samples=len(dataloader))
        
        model = self.model
        model = model.to(dtype=torch.bfloat16)
        
        model.eval()
            
        # losses/preds/labels on CPU (final containers)
        all_preds = []
        all_labels = []
        all_raw_decoded = []
         
        def post_process(str_out):
            try:
                gen_text = str_out.split("assistant\n")[-1].split("<|im_end|>")[0].strip()
            except:
                gen_text = "error"
            return gen_text
        
        # Main evaluation loop
        with torch.no_grad():
            for i_step, inputs in enumerate(tqdm(dataloader)):
                inputs = self._prepare_inputs(inputs)
                gen_kwargs = {'max_new_tokens': 300, 
                              'do_sample': False, 
                              'eos_token_id': self.tokenizer.eos_token_id, 
                              'pad_token_id': self.tokenizer.pad_token_id,
                              "temperature": 0.1,
                              }
                generated_tokens = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **gen_kwargs,
                )
                labels = inputs.pop("labels")
                str_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                
                raw_decoded = [e for e in self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)]
                str_decoded = [post_process(e) for e in raw_decoded]
                all_preds += str_decoded
                all_labels += str_labels 
                all_raw_decoded += raw_decoded
                
                if i_step % 1000 == 999:
                    _acc = sum([all_labels[i] == all_preds[i] for i in range(len(all_preds))]) / len(all_preds)
                    metrics = { f"{metric_key_prefix}_em": _acc  }
                    json.dump({"metrics": metrics, 
                            "detail_pred": list(zip(all_preds, all_labels, all_raw_decoded))}, 
                            open(f"{self.args.output_dir}/result_{metric_key_prefix}_step-{self.state.global_step}.json", "wt"), indent=1)
                    
                    
        num_samples = len(dataloader)
          
        acc = sum([all_labels[i] == all_preds[i] for i in range(len(all_labels))]) / len(all_labels)
        metrics = { f"{metric_key_prefix}_em": acc  }
        
        json.dump({"metrics": metrics, 
                   "detail_pred": list(zip(all_preds, all_labels, all_raw_decoded))}, 
                   open(f"{self.args.output_dir}/result_{metric_key_prefix}_step-{self.state.global_step}.json", "wt"), indent=1)
        
        # free the memory again
        del model
        torch.cuda.empty_cache()
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

def gen_idx_valid_set(total = 500, max_n = 4000, path_rand_indexes = None):
    if os.path.exists(path_rand_indexes):
        return json.load(open(path_rand_indexes))
    else:
        indexes = [i for i in range(max_n)]
        random.shuffle(indexes)
        json.dump(indexes[:total], open(path_rand_indexes, 'wt'))
        return indexes[:total]
    
def load_or_gen_dataset(args):
    # convert data 
    if args.re_gen_data:
        for d_type in ['dev', 'train', 'test']:
            if args.prompting_type == 'cot':
                process_raw_data_cot_prompting(f'/home/phuongnm/SP_LLMs/data/top/mrc-ner.{d_type}',
                                            candidates_path=f'/home/phuongnm/SP_LLMs/data/top/CoT/{d_type}/robertapair_all_rouge.{d_type}.json', 
                                            path_data_out=f'/home/phuongnm/SP_LLMs/data/top/top.{d_type}.{args.prompting_type}.{args.kshot}-shot.jsonl',
                                            topk=args.kshot)
                
            elif args.prompting_type == 'cotv2':
                process_raw_data_cotv2_prompting(f'/home/phuongnm/SP_LLMs/data/top/mrc-ner.{d_type}',
                                            candidates_path=f'/home/phuongnm/SP_LLMs/data/top/CoT/{d_type}/robertapair_all_rouge.{d_type}.json', 
                                            path_data_out=f'/home/phuongnm/SP_LLMs/data/top/top.{d_type}.{args.prompting_type}.{args.kshot}-shot.jsonl',
                                            topk=args.kshot)
            
            elif args.prompting_type == 'fewshot':
                process_raw_data_fewshot_prompting(f'/home/phuongnm/SP_LLMs/data/top/mrc-ner.{d_type}',
                                            candidates_path=f'/home/phuongnm/SP_LLMs/data/top/CoT/{d_type}/robertapair_all_rouge.{d_type}.json', 
                                            path_data_out=f'/home/phuongnm/SP_LLMs/data/top/top.{d_type}.{args.prompting_type}.{args.kshot}-shot.jsonl',
                                            topk=args.kshot)
            elif args.prompting_type == 'zeroshot':
                process_raw_data_zeroshot_prompting(f'/home/phuongnm/SP_LLMs/data/top/mrc-ner.{d_type}',
                                        path_data_out=f'/home/phuongnm/SP_LLMs/data/top/top.{d_type}.{args.prompting_type}.{args.kshot}-shot.jsonl') 
                
    data_path_format = f"/home/phuongnm/SP_LLMs/data/top/top.train.{args.prompting_type}.{args.kshot}-shot.jsonl"
        
    # Load jsonl data from disk
    dataset = datasets.load_dataset("json", data_files=data_path_format, split="train", 
                            cache_dir="/".join(data_path_format.split("/")[:-1]))
    valid_dataset = datasets.load_dataset("json", data_files=data_path_format.replace(".train.", ".dev."), split="train", 
                                    cache_dir="/".join(data_path_format.split("/")[:-1]))
    test_dataset = datasets.load_dataset("json", data_files=data_path_format.replace(".train.", ".test."), split="train",
                                cache_dir="/".join(data_path_format.split("/")[:-1]))
    
    valid_indexes = gen_idx_valid_set(500, len(valid_dataset), f"/home/phuongnm/SP_LLMs/data/top/random_indexes_valid.json")
    valid_dataset_2 = valid_dataset.select(valid_indexes)
    
    return dataset, valid_dataset, valid_dataset_2, test_dataset
    
def load_or_gen_combine_dataset(args):
    # convert data 
    if args.re_gen_data:
        if args.sem_comb:
            data_gen_func = process_raw_data_cotv3_prompting
        else:
            data_gen_func = process_raw_data_cotv2_prompting
            
        for d_type in ['dev', 'train', 'test']:
            print(f"Generating {d_type} data ...")
        
            if args.prompting_type == 'cot':
                process_raw_data_cot_prompting(f'{args.data_dir}/mrc-ner.{d_type}',
                                            candidates_path=f'{args.data_dir}/robertapair_all_rouge.{d_type}.json', 
                                            path_data_out=f'{args.data_dir}/top.{d_type}.{args.prompting_type}.{args.kshot}-shot.jsonl',
                                            topk=args.kshot)
                
            elif args.prompting_type == 'cotv2':
                data_gen_func(f'{args.data_dir}/mrc-ner.{d_type}',
                                            candidates_path=f'{args.data_dir}/robertapair_all_rouge.{d_type}.json', 
                                            path_data_out=f'{args.data_dir}/top.{d_type}.{args.prompting_type}.{args.kshot}-shot.jsonl',
                                            topk=args.kshot,
                                            query_class=LLMCoTQuery)
            elif args.prompting_type == 'cotv2bottomup':
                data_gen_func(f'{args.data_dir}/mrc-ner.{d_type}',
                                            candidates_path=f'{args.data_dir}/robertapair_all_rouge.{d_type}.json', 
                                            path_data_out=f'{args.data_dir}/top.{d_type}.{args.prompting_type}.{args.kshot}-shot.jsonl',
                                            topk=args.kshot,
                                            query_class=LLMCoTBottomUpQuery)
            elif args.prompting_type == 'cotv2merging':
                data_gen_func(f'{args.data_dir}/mrc-ner.{d_type}',
                                            candidates_path=f'{args.data_dir}/robertapair_all_rouge.{d_type}.json', 
                                            path_data_out=f'{args.data_dir}/top.{d_type}.{args.prompting_type}.{args.kshot}-shot.jsonl',
                                            topk=args.kshot,
                                            query_class=LLMMergingCoTQuery)
            
            elif args.prompting_type == 'cotv2dfs':
                data_gen_func(f'{args.data_dir}/mrc-ner.{d_type}',
                                            candidates_path=f'{args.data_dir}/robertapair_all_rouge.{d_type}.json', 
                                            path_data_out=f'{args.data_dir}/top.{d_type}.{args.prompting_type}.{args.kshot}-shot.jsonl',
                                            topk=args.kshot,
                                            query_class=LLMDfsCoTQuery)
                
            elif args.prompting_type == 'cotv2bfs':
                data_gen_func(f'{args.data_dir}/mrc-ner.{d_type}',
                                            candidates_path=f'{args.data_dir}/robertapair_all_rouge.{d_type}.json', 
                                            path_data_out=f'{args.data_dir}/top.{d_type}.{args.prompting_type}.{args.kshot}-shot.jsonl',
                                            topk=args.kshot,
                                            query_class=LLMBfsCoTQuery)
                
            elif args.prompting_type == 'base':
                data_gen_func(f'{args.data_dir}/mrc-ner.{d_type}',
                                            candidates_path=f'{args.data_dir}/robertapair_all_rouge.{d_type}.json', 
                                            path_data_out=f'{args.data_dir}/top.{d_type}.{args.prompting_type}.{args.kshot}-shot.jsonl',
                                            topk=args.kshot,
                                            query_class=LLMBaseQuery)
                
            elif args.prompting_type == 'fewshot':
                process_raw_data_fewshot_prompting(f'{args.data_dir}/mrc-ner.{d_type}',
                                            candidates_path=f'{args.data_dir}/robertapair_all_rouge.{d_type}.json', 
                                            path_data_out=f'{args.data_dir}/top.{d_type}.{args.prompting_type}.{args.kshot}-shot.jsonl',
                                            topk=args.kshot)
            elif args.prompting_type == 'zeroshot':
                process_raw_data_zeroshot_prompting(f'{args.data_dir}/mrc-ner.{d_type}',
                                        path_data_out=f'{args.data_dir}/top.{d_type}.{args.prompting_type}.{args.kshot}-shot.jsonl') 
                
       
    def cus_load_dataset(path_f):
        lines = []
        with open(path_f) as f:
            for l in f.readlines ():
                try:
                    jobj = json.loads(l.strip() )
                    lines.append(jobj)
                except Exception as e:
                    pass
            _dataset = Dataset.from_list(lines)
        return _dataset
        
 
    data_path_format = f"{args.data_dir}/top.train.{args.prompting_type}.{args.kshot}-shot.jsonl"
        
    # Load jsonl data from disk
    dataset = cus_load_dataset(data_path_format)
    valid_dataset = cus_load_dataset(data_path_format.replace(".train.", ".dev."))
    test_dataset = cus_load_dataset(data_path_format.replace(".train.", ".test."))
    
    # dataset = datasets.load_dataset("json", data_files=data_path_format, split="train", 
    #                         cache_dir="/".join(data_path_format.split("/")[:-1]))
    # valid_dataset = datasets.load_dataset("json", data_files=data_path_format.replace(".train.", ".dev."), split="train", 
    #                                 cache_dir="/".join(data_path_format.split("/")[:-1]))
    # test_dataset = datasets.load_dataset("json", data_files=data_path_format.replace(".train.", ".test."), split="train",
    #                             cache_dir="/".join(data_path_format.split("/")[:-1]))
    
    valid_indexes = gen_idx_valid_set(500, len(valid_dataset), f"{args.data_dir}/random_indexes_valid.json")
    valid_dataset_2 = valid_dataset.select(valid_indexes)
    
    return dataset, valid_dataset, valid_dataset_2, test_dataset

if __name__=='__main__':    
    
    parser = argparse.ArgumentParser(description='Process ...')
    parser.add_argument('--re_gen_data', action="store_true", help='re generate data', default=False)
    parser.add_argument('--seed', type=int, default=42, help='random seed value')
    parser.add_argument('--no_data_packing', action="store_true", help='re generate data', default=False)
    parser.add_argument('--do_train', action="store_true", help='fine tuning a LLM model with LoRA', default=False)
    parser.add_argument('--do_eval_test', action="store_true", help='eval on test set', default=False)
    parser.add_argument('--do_eval_dev', action="store_true", help='eval on dev set', default=False)
    parser.add_argument('--ft_model_path', type=str, default=None, help='fintuned model path') 
    parser.add_argument('--ft_model_id', type=str, default=None, help='fintuned model id for saving after train it')
    parser.add_argument('--prompting_type', type=str, default='cot', help='prompting style in {cot, fewshot, zeroshot}')
    parser.add_argument('--base_model_id', type=str, default='meta-llama/Llama-2-7b-hf', help='base llm model id')
    parser.add_argument('--data_dir', type=str, default='./data/', help='data dir')
    parser.add_argument('--output_dir', type=str, default='./finetuned_llm/', help='ouput dir')
    parser.add_argument('--epoch', type=int, default=3, help='training epoch')
    parser.add_argument('--lr_scheduler', type=str, default='constant', help='learning rate scheduler')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate value')
    parser.add_argument('--kshot', type=int, default=3, help='k shot examples for llm')
    parser.add_argument('--lora_r', type=int, default=32, help='lora rank')
    parser.add_argument('--sem_comb', action="store_true", help='semantic combination', default=False)
    parser.add_argument('--max_seq_length', type=int, default=None, help='max seq length')
    
    args, unknown = parser.parse_known_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.prompting_type == 'zeroshot':
        args.kshot = 0
        
    set_random_seed(args.seed)
    print(args)
        
    dataset, valid_dataset, valid_dataset_2, test_dataset =  load_or_gen_combine_dataset(args) # load_or_gen_combine_dataset(args) #  

    
    # Load model and tokenizer
    model_id = args.base_model_id # "codellama/CodeLlama-7b-hf" # or `mistralai/Mistral-7B-v0.1`
    if args.do_train:
        tensor_data_type = torch.bfloat16  
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=tensor_data_type
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=tensor_data_type,
            quantization_config=bnb_config
        )
        if 'Mistral' in model_id:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True,use_fast=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
    else:
        tensor_data_type = torch.float32 # for reduce the miss matching of ouputs of batch inference
        ft_model_path = f"{args.output_dir}/{args.ft_model_id}" if args.ft_model_path is None else args.ft_model_path
        tokenizer = AutoTokenizer.from_pretrained(ft_model_path)
        model = AutoPeftModelForCausalLM.from_pretrained(
            ft_model_path,
            device_map="auto",
            torch_dtype=tensor_data_type
        )
        
    tokenizer.padding_side = 'left' # to prevent warnings

    # # set chat template to OAI chatML, remove if you start from a fine-tuned model
    model, tokenizer = setup_chat_format(model, tokenizer)
    
    # training config 
    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
            lora_alpha=128,
            lora_dropout=0.05,
            r=args.lora_r,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM", 
    )

    ft_model_id = args.ft_model_id 
    output_dir=f'{args.output_dir}/{ft_model_id}' if args.ft_model_path is None else args.ft_model_path
    training_args = TrainingArguments(
        output_dir=output_dir,                  # directory to save and repository id
        num_train_epochs=args.epoch,                     # number of training epochs
        per_device_train_batch_size=4,          # batch size per device during training
        per_device_eval_batch_size=16,          # batch size per device during training
        gradient_accumulation_steps=4,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        save_total_limit=1,
        optim="adamw_torch_fused",              # use fused adamw optimizer
        
        eval_delay=10,                       # log every 10 steps
        logging_steps=50,                       # log every 10 steps
        eval_steps=0.1,
        save_steps=0.1,
        load_best_model_at_end=False,
        metric_for_best_model='em',
        greater_is_better=True,
        eval_strategy='steps',
        save_strategy="steps", 
        
        learning_rate=args.lr,                     # learning rate, based on QLoRA paper
        bf16=True,                              # use bfloat16 precision
        tf32=True,                              # use tf32 precision
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
        lr_scheduler_type=args.lr_scheduler,           # use constant learning rate scheduler
        push_to_hub=False,                      # push model to hub ##########################
        group_by_length=True,
        report_to="tensorboard",                # report metrics to tensorboard
    ) 

    trainer = LLMSpTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=valid_dataset_2,
        neftune_noise_alpha=5,
        peft_config=peft_config,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        packing=not args.no_data_packing,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False, # No need to add additional separator token
        }
    )
    
    config_file = f"{output_dir}/run_config.json"
    all_params = {'args': vars(args), 'training_args': dict([(k, str(v))for k, v in vars(training_args).items() ])}
    if args.do_train:
        print("training .... ")
        json.dump(all_params, open(config_file.replace(".json", "_tmp.json"), "wt"), indent=2)
        
        pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        pytorch_total_params = sum(p.numel() for p in model.parameters() )
        print(f"pytorch_total_trainable_params={pytorch_total_trainable_params}, pytorch_total_params = {pytorch_total_params}" )
        
        # start training, the model will be automatically saved to the hub and the output directory
        trainer.train(resume_from_checkpoint=True if len(glob.glob(f'{output_dir}/checkpoint-*')) > 0 else None)

        # save model 
        trainer.save_model()
        json.dump(all_params, open(config_file, "wt"), indent=2)
        os.remove(config_file.replace(".json", "_tmp.json"))
    else:
        if not os.path.exists(config_file):
            json.dump(all_params, open(config_file, "wt"), indent=2)
        
    if args.do_eval_dev:
        print("eval dev .... ")
        result = trainer.evaluate(valid_dataset, metric_key_prefix='valid')
        print(f"Valid result = {result}")
        
    if args.do_eval_test:
        print("eval test .... ")
        result = trainer.evaluate(test_dataset, metric_key_prefix='test')
        print(f"Test result = {result}")
        
