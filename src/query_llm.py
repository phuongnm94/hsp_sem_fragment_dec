
import json
import logging
import os
import pickle
import sys
from dotenv import dotenv_values
import torch
import argparse
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM
from utils import *
import numpy as np
from torch.utils.data import DataLoader
import traceback
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class LLMBaseQuery:
    def __init__(self, args, model, tokenizer) -> None:
        self.args, self.model, self.tokenizer = args, model, tokenizer
        if args.path_folder_data is not None:
            self.run_id = f"{args.model.split('/')[-1]}_{args.query_type}_{args.predicate_retriever}_{args.num_samples}_top_{args.top_k_demos}_sim_{args.sim_method}_chunk_{args.cur_chunk}:{args.num_chunk}_semcomb_{args.sem_comb}"
            self.output_folder = f"{args.path_folder_data}/out/{self.run_id}"
            if args.prepare_openai_query:
                self.output_folder = f"{self.output_folder}_openai-data"
                
            os.makedirs(self.output_folder, exist_ok=True)
            
            if not os.path.exists(f'{self.output_folder}/config.json'):
                json.dump(vars(args), open(f'{self.output_folder}/config.json', 'wt'), indent=2)
            else: 
                json.dump(vars(args), open(f'{self.output_folder}/config_new.json', 'a'), indent=2)
                
            logging.basicConfig(filename=f'{self.output_folder}/run.log', filemode='a', level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
            logging.info(self.args)

    @classmethod
    def get_lg_tree(cls, logic_form):
        if isinstance(logic_form, str):
            tree = Tree(logic_form).root
        elif isinstance(logic_form, Tree):
            tree = logic_form.root
            
        def _remove_word(cur_node):
            for idx in range(len(cur_node.children) - 1, -1, -1):
                if type(cur_node.children[idx]) == Token:
                    cur_node.children.pop(idx)
                else:
                    _remove_word(cur_node.children[idx])
        _remove_word(tree)
        return str(tree)
    
    def format_output(self, output_text, input_text, args=None):
        cleaned_output_text = output_text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").replace(input_text, "").strip()

        output_sample = cleaned_output_text.strip().split("\n")[0].replace("Output: ", "").strip()
        return output_sample
    
    # def format_output(self, output_text, input_text, args=None):
    #     cleaned_output_text = output_text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").replace(input_text, "").strip()

    #     output_sample = cleaned_output_text.split("\n\n")[0].strip()
    #     return output_sample
    
    @classmethod
    def build_output(cls, input_text, logic_form):
        return logic_form, logic_form

    def check_output(self, gold_lg, pred_lg): 
        return gold_lg.strip() == pred_lg.strip()

    def restore_full_logical_form(self, lg, input_sent=None):
        return lg.strip()
    
    def evaluate(self, output, gold_data):
        count_true = 0
        output_dict = dict([(e['id']-1, e) for e in output])
        for i_sample in range(len(gold_data)):
            gold_lg = gold_data[i_sample].replace('"" "" "" ""', "").replace('"[', "[").replace(']"', "]")
            try:
                pred_lg =  self.restore_full_logical_form(output_dict[i_sample]['q_output'])
            except Exception as e:
                traceback.print_exc()
                print(e)
                pred_lg = ""
                
            is_correct_prediction = self.check_output(gold_lg, pred_lg)
            if is_correct_prediction: 
                count_true += 1
            else:
                print(i_sample, gold_lg,"\t\t", pred_lg)
        acc = count_true/ len(output)
        msg = f"Acc = {acc}"
        logging.info(msg)
        print(msg)
        return acc

    @classmethod
    def query_generate(cls, input_samples, top_k_demos, *args, **kwargs):
        str_query = ""
        
        # support example 
        for candidate in input_samples['candidates'][:top_k_demos]:
            input_text = f"Input: {candidate[0]}\n"
            input_text += f"Output: {candidate[1]}\n\n"
            str_query += input_text
        
        # curent query 
        str_query +=  f"""Input: {input_samples['q_input']}\nOutput: """ 
        return str_query
    
    def query_generate_with_support_samples(self, input_samples, top_k_demos, max_add_tokens=50, *args, **kwargs):
        MAX_SEQ_LENGTH = self.tokenizer.max_len_single_sentence
        tmp_top_k_demos = top_k_demos
        input_text = self.query_generate(input_samples, tmp_top_k_demos, *args, **kwargs)
        current_length = self.tokenizer(input_text, return_tensors="pt")["input_ids"].shape[-1]
        logic_length_expectation = current_length // tmp_top_k_demos
        while MAX_SEQ_LENGTH < current_length + logic_length_expectation + max_add_tokens:
            tmp_top_k_demos = tmp_top_k_demos - 1
            input_text = self.query_generate(input_samples, tmp_top_k_demos, *args, **kwargs)
            current_length = self.tokenizer(input_text, return_tensors="pt")["input_ids"].shape[-1]
            logic_length_expectation = current_length // tmp_top_k_demos
        return input_text, logic_length_expectation + max_add_tokens, tmp_top_k_demos, current_length

    def single_query(self, input_text, max_new_tokens=300, temperature=0.0):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature)
        output_text = self.tokenizer.decode(outputs[0])
        return output_text

    def batch_query(self, input_text, max_new_tokens=300, temperature=0.0):
        if self.args.force_batch_infer:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left' # to prevent warnings
            inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation='longest_first')
        else:
            inputs = self.tokenizer(input_text, return_tensors="pt", padding=False)
        inputs = inputs.to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=False)
        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output_text

    def load_data(self):
        data_folder = args.path_folder_data
        with open (f'{data_folder}/golds_{args.data_type}_{args.num_samples}.tsv') as f:
            gold_data = [l.strip() for l in f.readlines()]
        all_candidate_examples = json.load(open(f'{data_folder}/{args.predicate_retriever}_{args.num_samples}_{args.sim_method}.{args.data_type}.json'))
        
        # filter query ids
        lst = range(len(all_candidate_examples))
        query_ids = set(np.array_split(lst, args.num_chunk)[args.cur_chunk-1])
        
        if os.path.exists(f'{self.output_folder}/result.json'):
            result_out = json.load(open(f'{self.output_folder}/result.json') )
            for e in result_out:
                if (e['id']-1) in query_ids:
                    query_ids.remove((e['id']-1))
        else:
            result_out = []
        print(list(query_ids)[:10])
        print(len(query_ids))
        
        return all_candidate_examples, gold_data, query_ids, result_out, None
        
    def prepare_query(self, prompt_msg = None):
        args = self.args
        all_candidate_examples, gold_data, query_ids, result_out, missing_predicate = self.load_data()
        
        utterances = []
        prepared_data = []
        
        for i_check, support_samples in enumerate(tqdm(all_candidate_examples)):
            utterances.append(support_samples['q_input'])
            if i_check not in query_ids:
                continue
            
            input_text = self.query_generate(support_samples, args.top_k_demos,  add_final_step=args.sem_comb)
            gold_lg = gold_data[i_check].replace("\" \" \" \"", "").replace('"[', "[").replace(']"', "]")

            prompt_msg = "" if prompt_msg is None else prompt_msg
            prepared_data.append({
                'id': i_check,
                'text_input': prompt_msg + input_text,
                'gold_lg': gold_lg,
                'func_name': f'{self}.{self.batch_query.__name__}',
                'num_demonstrations': args.top_k_demos
            }) 
                     
        return prepared_data
    
    def prepare_openai_query(self):
        if not os.path.exists(f'{self.output_folder}/tracking_submission.json'):
            query_data = self.prepare_query()
            all_data = []
            sys_msg = "You are an expert in hierarchical semantic parsing. Please follow my example to generate the \"Output\" for my most recent \"Input\" in the same format as the example, without any explanation."
            for sample in query_data:
                sample_data = {
                    "custom_id": str(sample['id']),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                            "model": self.args.model,
                            "messages": [
                                    {
                                            "role": "system",
                                            "content": sys_msg,
                                    },
                                    {
                                            "role": "user",
                                            "content": sample['text_input']
                                    }
                            ],
                            "temperature": 0.1,
                            "max_tokens": 300
                    }
                }
                all_data.append(json.dumps(sample_data))
            
            json.dump(query_data, open(f'{self.output_folder}/query_data.json', 'wt'), indent=1, ensure_ascii=False)
            with  open(f'{self.output_folder}/data_openai.jsonl', 'wt') as f:
                f.write("\n".join(all_data)) 
                
            from openai import OpenAI

            config = dotenv_values("./src/.env")
            KEY=config['OPENAI_KEY'] 
            client = OpenAI(api_key=KEY)
                
            batch_input_file = client.files.create(
                file=open(f'{self.output_folder}/data_openai.jsonl', "rb"),
                purpose="batch"
            )
            batch_obj = client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": f"{self.run_id}"
                }
            )
            print(batch_obj)
            json.dump(
                dict([(str(k), str(v)) for k, v in batch_obj.__dict__.items()]), 
                open(f'{self.output_folder}/tracking_submission.json', "wt"), 
                indent=1)
        else:
            if not os.path.exists(f'{self.output_folder}/result_submission.jsonl'):
                from openai import OpenAI

                config = dotenv_values("./src/.env")
                KEY=config['OPENAI_KEY'] 
                client = OpenAI(api_key=KEY)
                submission_info = json.load(open(f'{self.output_folder}/tracking_submission.json', 'rt'))
                batch_obj = client.batches.retrieve(submission_info['id'])
                print(batch_obj)
                if batch_obj.output_file_id is not None:
                    file_response = client.files.content(batch_obj.output_file_id)
                    with open(f'{self.output_folder}/result_submission.jsonl', 'wt') as f:
                        f.write(file_response.text)
                    print(f"Check result in: {self.output_folder}/result_submission.jsonl")

                if batch_obj.error_file_id is not None:
                    file_response = client.files.content(batch_obj.error_file_id)
                    with open(f'{self.output_folder}/err_log_submission.log', 'wt') as f:
                        f.write(file_response.text)
            else:      
                print(f"Check result in: {self.output_folder}/result_submission.jsonl")
                
            self.conbine_openai_result()
                
    def conbine_openai_result(self):
        file_result  = f'{self.output_folder}/result_submission.jsonl'
        if os.path.exists(file_result):
            print(f'- conbine_openai_result => {self.output_folder}/result.json')
            combined_results = []
            with open(f'{self.output_folder}/query_data.json') as f:
                input_data = json.load(f)
            with open(f'{self.output_folder}/result_submission.jsonl') as f:
                shuffle_results = [json.loads(e.strip()) for e in f.readlines()]
                results = [None]*len(shuffle_results)
                for e in shuffle_results:
                    results[int(e['custom_id'])] = e
            for input_info, result_openai in zip(input_data, results):
                combined_results.append({
                    "id": input_info['id']+1,
                    "q_input": input_info['text_input'],
                    "q_output_raw": input_info['text_input']+ " " +result_openai['response']['body']['choices'][0]['message']['content'],
                    "gold_label": input_info['gold_lg'],
                    "func_name": input_info['func_name'],
                    "max_new_tokens": 300,
                    "num_demonstrations": input_info['num_demonstrations']
                })
            json.dump(combined_results, open(f'{self.output_folder}/result.json', 'wt'), indent=1)
        
    def generate_all_query(self):
        args = self.args
        all_candidate_examples, gold_data, query_ids, result_out, missing_predicate = self.load_data()
        
        utterances = []
        
        q_input_group_by_len = {}
        for i_check, support_samples in enumerate(tqdm(all_candidate_examples)):
            utterances.append(support_samples['q_input'])
            if i_check not in query_ids:
                continue
            str_query, max_new_tokens, num_demonstrations, q_sub_tok_len = self.query_generate_with_support_samples(support_samples, 
                                                                                                                    args.top_k_demos, 
                                                                                                                    50, 
                                                                                                                    missing_predicate[i_check] if missing_predicate is not None else None,
                                                                                                                    add_final_step=args.sem_comb)
            if self.args.force_batch_infer:
                if int(i_check//50) not in q_input_group_by_len:
                    q_input_group_by_len[int(i_check//50)] = []
                    
                q_input_group_by_len[int(i_check//50)].append([str_query, max_new_tokens, num_demonstrations, q_sub_tok_len, i_check])
            else:
                if q_sub_tok_len not in q_input_group_by_len:
                    q_input_group_by_len[q_sub_tok_len] = []
                    
                q_input_group_by_len[q_sub_tok_len].append([str_query, max_new_tokens, num_demonstrations, q_sub_tok_len, i_check])
                    
        for q_len, gr_info in tqdm(q_input_group_by_len.items()):
            q_outputs = []
            
            try:
                # try to infer with llm
                for batch_info in tqdm(DataLoader(gr_info, batch_size=args.batch_size, shuffle=False)):
                    batch_queries = batch_info[0]
                    batch_max_new_tokens = max(batch_info[1])
                    batch_output =  self.batch_query(batch_queries, batch_max_new_tokens)    # TODO CHECK
                    q_outputs += batch_output
            except Exception as e:
                traceback.print_exc()
                print(e)
                continue  # if exception by out of CUDA MEMORY
                
            for idx, q_output in enumerate(q_outputs):
                gr_info[idx].append(q_output)
                
            for idx, (str_query, max_new_tokens, num_demonstrations, q_sub_tok_len, i_check, q_output) in enumerate(gr_info):
                try:
                    output_raw = self.format_output(q_output, str_query, args)
                    pred_lg = self.restore_full_logical_form(output_raw, input_sent=utterances[i_check])
                except Exception as e:
                    traceback.print_exc()
                    print(e)
                    pred_lg = ""
                gold_lg = gold_data[i_check].replace("\" \" \" \"", "").replace('"[', "[").replace(']"', "]")

                result_out.append({
                    "id": i_check+1,
                    "q_input": str_query,
                    "q_output": pred_lg,
                    "q_output_raw": q_output,
                    "gold_label": gold_lg,
                    "correct": self.check_output(pred_lg.lower(), gold_lg.lower()),
                    'func_name': f'{self}.{self.batch_query.__name__}',
                    'max_new_tokens': max_new_tokens,
                    'num_demonstrations': num_demonstrations
                })
            
                logging.info(q_output)
                logging.info("=="*20)
            json.dump(result_out, open(f'{self.output_folder}/result.json', 'wt'), indent=2)
            
        # json.dump(result_out, open(f'{self.output_folder}/result.json', 'wt'), indent=2)
        # self.evaluate(json.load(open(f'{self.output_folder}/result.json')), gold_data=gold_data)

class LLMCoTQuery(LLMBaseQuery):
        
    def format_output(self, output_text, input_text, args=None):
        cleaned_output_text = output_text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").replace(input_text, "").strip()
        if args is not None and args.model == 'google/flan-ul2':
            cleaned_output_text = cleaned_output_text.replace('Step ', '\nStep ')
        output_sample = cleaned_output_text.split("\n\n")[0]
        predicted_steps = output_sample.split("\n")
        result = []
        pattern = r"Step \d+: "  # Regex pattern to match "Step [number]: "
        for step in predicted_steps:
            tmp = re.sub(pattern, "", step).strip()
            result.append(tmp)
        return result

    def restore_full_logical_form(self, COT_steps, input_sent=None):
        for e in COT_steps:
            if "Semantic combination" in e:
                return e.replace("Semantic combination: ", "")
            
        OPEN_BRACKET= "["
        CLOSE_BRACKET= "]"
        if len(COT_steps) == 0:
            return ""
        
        org_steps = COT_steps
        
        # normalize steps
        COT_steps = []
        first_intent = org_steps[0].split(" ")[0]
        COT_steps.append(f"{first_intent} {input_sent} ]")
        for e in org_steps[1:]:
            e = e.replace(", ]", " ]").replace("?", "").replace("'s", " 's ")
            m = re.search(r'(\w*\s*\.) \]', e)
            if m is not None and input_sent.endswith(m.group(1)):
                e = e.replace(m.group(1), m.group(1)[:-1]) # only remove "." in if that is end of sentence, e.g. skip L.A. 
            e = re.sub(" {2,}", " ", e.strip())
            COT_steps.append(e)
        
        def find_sub_list(rsl, rl):
            sl = [item.lower() for item in rsl]
            l = [item.lower() for item in rl]
            results=[]
            sll=len(sl)
            for ind in (i for i,e in enumerate(l) if e==sl[0]):
                if l[ind:ind+sll]==sl:
                    results.append((ind,ind+sll-1))
            return results
            
        
        root_tokens = COT_steps[0].split(" ")
        for step in COT_steps[1:]:
            
            # Extract info at this step
            cur_tokens = step.split(" ")
            label_token = cur_tokens[0]
            close_backet_token = cur_tokens[-1]
            span_tokens = cur_tokens[1:-1] 

            # Check valid sub-logics
            if not (label_token.startswith(OPEN_BRACKET) and close_backet_token == CLOSE_BRACKET):
                raise ValueError(f'"{step}" is an invalid sub logic.')

            span_indices = find_sub_list(span_tokens, root_tokens)
            if len(span_indices) == 0:
                raise ValueError(f'Do not find the span of step "{step}" in the utterance {root_tokens}.')
            
            # Insert labels to the utterance
            for x in reversed(span_indices[:1]):
                root_tokens.insert(x[1]+1, close_backet_token)
                root_tokens.insert(x[0], label_token)

        return " ".join(root_tokens)

    @classmethod
    def get_node_info_from_lg(cls, logic_form):
        return get_node_info(Tree(logic_form))
    
    @classmethod
    def build_output(cls, input_text, logic_form, add_final_step=False):
        node_info = cls.get_node_info_from_lg(logic_form)
        label_types = [item[2] for item in node_info]
        entity_levels = [item[1] for item in node_info]
        start_positions = [int(item[0].split(";")[0]) for item in node_info]
        end_positions = [int(item[0].split(";")[1]) for item in node_info]

        tokens = input_text.split(" ")
        result = []
        result_list = []
        idx = 1
        for label_type, ent_level, start_pos, end_pos in zip(label_types, entity_levels, start_positions, end_positions):
            tmp = f"Step {idx}: " + f"[{label_type} " + " ".join(tokens[start_pos:end_pos+1]) + " ]"
            result.append(tmp)
            result_list.append(f"[{label_type} " + " ".join(tokens[start_pos:end_pos+1]) + " ]")
            idx += 1
        if add_final_step:
            result.append(f"Semantic combination: {logic_form}")
            result_list.append(f"Semantic combination: {logic_form}")
        return "\n".join(result), result_list
    
    @classmethod
    def query_generate(cls, input_samples, top_k_demos, *args, **kwargs):
        str_query = ""
        
        # support example 
        for candidate in input_samples['candidates'][:top_k_demos]:
            input_text = f"Input: {candidate[0]}\n"
            input_text += cls.build_output(candidate[0], candidate[1], kwargs.get('add_final_step', False))[0]
            input_text += f"\n\n"
            str_query += input_text
        
        # curent query 
        str_query +=  f"""Input: {input_samples['q_input']}\nStep 1:""" 
        return str_query
    
class LLMCoTBottomUpQuery(LLMCoTQuery):
         
    @classmethod
    def get_node_info_from_lg(cls, logic_form):
        def _list_nonterminals_node(cur_node, node_info, node_level):
            for child in cur_node.children:
                if type(child) != Root and type(child) != Token:
                    _list_nonterminals_node(child, node_info, node_level+1)
            
            for child in cur_node.children:
                if type(child) != Root and type(child) != Token:
                    try:
                        tmp = get_span(child)
                        tmp = str(tmp[0]) + ";" + str(tmp[1]-1)
                        node_info.append((tmp, node_level, child.label, cur_node.label))
                    except Exception as e:
                        print(f'[Err] get span of  {child}')
            return node_info

        tree = Tree(logic_form)
        node_info = _list_nonterminals_node(tree.root, [], 1)
        return node_info
 
    def restore_full_logical_form(self, COT_steps, input_sent=None):
        all_steps = COT_steps + []
        all_steps.reverse()
        return super().restore_full_logical_form(all_steps, input_sent)
    
class LLMBfsCoTQuery(LLMCoTQuery):
    @classmethod
    def get_node_info_from_lg(cls, logic_form):
        
        def _list_nonterminals_node(queue_nodes, checking_idx):
            while True:
                if checking_idx >= len(queue_nodes):
                    break 
                node = queue_nodes[checking_idx]
                for child in node.children:
                    if type(child) != Root and type(child) != Token:
                        queue_nodes.append(child)
                checking_idx += 1
            return queue_nodes
        
        tree = Tree(logic_form)
        queue_nodes = [tree.root]
        all_nodes = _list_nonterminals_node(queue_nodes, 0)
        node_info = []
        for node in all_nodes[1:]:
            tmp = get_span(node)
            tmp = str(tmp[0]) + ";" + str(tmp[1]-1) 
            node_info.append([tmp, 1, node.label, None, 0]) 
            
        return node_info
    
class LLMDfsCoTQuery(LLMCoTQuery):
    
    @classmethod
    def get_node_info_from_lg(cls, logic_form):
        
        def _list_nonterminals_node(cur_node, node_info, node_level):
            for child in cur_node.children:
                if type(child) != Root and type(child) != Token:
                    try:
                        tmp = get_span(child)
                        tmp = str(tmp[0]) + ";" + str(tmp[1]-1) 
                        node_info.append([tmp, node_level, child.label, cur_node.label])
                        _list_nonterminals_node(child, node_info, node_level+1)
                    except Exception as e:
                        print(f'[Err] get span of  {child}')
            return node_info

        node_info = _list_nonterminals_node(Tree(logic_form).root, [], 1)
        return node_info

class LLMMergingCoTQuery(LLMCoTQuery):
    
    @classmethod
    def get_node_info_from_lg(cls, logic_form):
        
        def _list_nonterminals_node(cur_node, node_info, node_level, parent_index=None):
            for child in cur_node.children:
                if type(child) != Root and type(child) != Token:
                    try:
                        tmp = get_span(child)
                        tmp = str(tmp[0]) + ";" + str(tmp[1]-1)
                        if parent_index != tmp:
                            node_info.append([tmp, node_level, child.label, cur_node.label])
                        else:
                            node_info[-1][2] = f"{node_info[-1][2]}/{child.label}"
                        _list_nonterminals_node(child, node_info, node_level+1, tmp)
                    except Exception as e:
                        print(f'[Err] get span of  {child}')
                        
            return node_info

        tree = Tree(logic_form)
        node_info = _list_nonterminals_node(tree.root, [], 1)
 
        return node_info
    
    def restore_full_logical_form(self, COT_steps, input_sent=None):
        all_steps = COT_steps + []
        fixed_all_steps = []
        for idx in range(len(all_steps)):
            if "Semantic combination" in all_steps[idx]:
                return all_steps[idx].replace("Semantic combination: ", "")
            
            tokens = all_steps[idx].split(" ")
            logic_token = tokens[0]
            span_str = " ".join(tokens[1:-1])
            
            sub_logics = logic_token[1:].split("/")
            for sub_logic in sub_logics:
                fixed_all_steps.append(f"[{sub_logic} {span_str} ]")
            
        return super().restore_full_logical_form(fixed_all_steps, input_sent)

class LLMv3CoTQuery(LLMCoTQuery):
        
    def format_output(self, output_text, input_text, args=None):
        cleaned_output_text = output_text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split("### Parse following sentence:\n\nInput:")[1].strip()
        if args is not None and args.model == 'google/flan-ul2':
            cleaned_output_text = cleaned_output_text.replace('Step ', '\nStep ')
        predicted_steps = cleaned_output_text.strip().split("\n")[1:]
        result = []
        pattern = r"Step \d+: "  # Regex pattern to match "Step [number]: "
        for step in predicted_steps:
            tmp = re.sub(pattern, "", step).strip()
            result.append(tmp)
            if "Step" not in step:
                break
        return result
    
    @classmethod
    def query_generate(cls, input_samples, top_k_demos, *args, **kwargs):
        str_query = f"### You are a hierarchical  semantic parser, given {top_k_demos} examples here:\n\n"
        
        # support example 
        for candidate in input_samples['candidates'][:top_k_demos]:
            input_text = f"Input: {candidate[0]}\n"
            input_text += cls.build_output(candidate[0], candidate[1])[0]
            input_text += f"\n\n"
            str_query += input_text
        
        # curent query 
        str_query +=  f"""### Parse following sentence:\n\nInput: {input_samples['q_input']}\nStep 1:""" 
        return str_query
    
class LLMCoTPredicateQuery(LLMCoTQuery):
    def load_data(self):
        all_candidate_examples, gold_data, finished_query_ids, result_out, _ = super().load_data()      
        with open(f'{args.path_folder_data}/{args.data_type}.sent_predicate') as f:
            missing_predicates = [[predicate.strip() for predicate in l.strip().split("[sep]")[1].strip().split(" ")] for l in f.readlines()]

        return all_candidate_examples, gold_data, finished_query_ids, result_out, missing_predicates
    
    def reorder_list_predicates(self, label_types, missing_predicates):
        grammar = {'ROOT':{}}
        freq_score = {}
        for labels in label_types:
            # compute freq score by lb, in the case the node prob is break 
            for lb in set(labels):
                if lb not in freq_score:
                    freq_score[lb] = 0
                else:
                    freq_score[lb] += 1
            
            # compute node grammar probs 
            labels = [e for e in labels if e in set(labels)]
            
            cur_node = 'ROOT'
            for idx, lb in enumerate(labels):
                if lb not in grammar[cur_node]:
                    grammar[cur_node][lb] = 0
                grammar[cur_node][lb] += 1
                
                # new probs
                cur_node = lb
                if cur_node not in grammar:
                    grammar[cur_node] = {}

        missing_predicates = set(missing_predicates)
        new_ordered_predicates = []
        cur_node = 'ROOT'
        while len(missing_predicates)>0:
            next_node_scores = grammar.get(cur_node, {})
            next_node_scores = list([(k, v) for k, v in next_node_scores.items()])
            next_node_scores.sort(key=lambda x: x[1], reverse=True)
            
            # found the best node
            found_node = False 
            for next_node, score in next_node_scores:
                if next_node in missing_predicates:
                    # choose the node having best score
                    new_ordered_predicates.append(next_node)
                    missing_predicates.remove(next_node)
                    found_node = True
                    cur_node = next_node
                    break 
            
            if not found_node:
                score_by_freq = [(check_node, freq_score.get(check_node, 0)) for check_node in missing_predicates]
                score_by_freq.sort(key=lambda x: x[1], reverse=True)
            
                new_ordered_predicates.append(score_by_freq[0][0])
                missing_predicates.remove(score_by_freq[0][0])
                cur_node = score_by_freq[0][0]
                
        return new_ordered_predicates 
                
    def build_output(self, input_text, logic_form, missing_predicates):
        node_info = get_node_info(Tree(logic_form))
        label_types = [item[2] for item in node_info]
        entity_levels = [item[1] for item in node_info]
        start_positions = [int(item[0].split(";")[0]) for item in node_info]
        end_positions = [int(item[0].split(";")[1]) for item in node_info]

        tokens = input_text.split(" ")
        
        # remove dupplicated in instruction sentence
        no_dupplicated_label_types =[]
        for i in range(len(label_types)):
            if label_types[i] not in no_dupplicated_label_types:
                no_dupplicated_label_types.append(label_types[i])
        result = [f"Extract the content of {', '.join(no_dupplicated_label_types)} from input sentence."]
        result_list = []
        idx = 1
        for label_type, ent_level, start_pos, end_pos in zip(label_types, entity_levels, start_positions, end_positions):
            tmp = f"Step {idx}: " + f"[{label_type} " + " ".join(tokens[start_pos:end_pos+1]) + " ]"
            result.append(tmp)
            result_list.append(f"[{label_type} " + " ".join(tokens[start_pos:end_pos+1]) + " ]")
            idx += 1
        return "\n".join(result), result_list
    
    def query_generate(self, input_samples, top_k_demos, missing_predicates, **kwargs):
        str_query = ""
        
        # missing_predicates => will be changed the order of predicates 
        supported_label_types = []
        for candidate in input_samples['candidates'][:top_k_demos]:
            node_info = get_node_info(Tree(candidate[1]))
            supported_label_types.append([item[2] for item in node_info])
            
        missing_predicates = self.reorder_list_predicates(supported_label_types, missing_predicates)

        # support example 
        for candidate in input_samples['candidates'][:top_k_demos]:
            input_text = f"Input: {candidate[0]}\n"
            input_text += self.build_output(candidate[0], candidate[1], missing_predicates)[0]
            input_text += f"\n\n"
            str_query += input_text
        
        # curent query 
        str_query +=  f"""Input: {input_samples['q_input']}\nExtract the content of {', '.join(missing_predicates)} from input sentence.\nStep 1:""" 
        return str_query
  
class LLMCoTModQuery(LLMCoTQuery):
    def load_data(self):
        all_candidate_examples, gold_data, finished_query_ids, result_out, _ = super().load_data()
        
        #  
        result_out = json.load(open(f'./data/top/CoT/dev/out/Llama-2-70b-hf_cotmap_robertapair_all_top_10_sim_rouge_chunk_1:1/result.json') )
        result_out = dict([(e['id']-1, e) for e in result_out])
                  
        with open('./data/top/CoT/dev/dev.sent_predicate') as f:
            predicates = [[predicate.strip() for predicate in l.strip().split("[sep]")[1].strip().split(" ")] for l in f.readlines()]

        stats = {True: 0, False: 0}
        missing_predicates = {}
        for q_id in result_out:
            cur_predicates = predicates[q_id]
            missing=set()
            for pred_ in cur_predicates:
                if f'[{pred_} ' not in result_out[q_id]['q_output']:
                    missing.add(pred_)
            if len(missing) > 0:
                stats[result_out[q_id]['correct']] +=1
                missing = list(missing)
                missing.sort()
                missing_predicates[q_id] = missing
            else:
                # remove q_id if that query is not be missing predicate 
                finished_query_ids.remove(q_id)
        print(stats)
        finished_query_ids = [e for e in finished_query_ids if e in missing_predicates]
        return all_candidate_examples, gold_data, finished_query_ids, missing_predicates
    
    def build_output(self, input_text, logic_form, missing_predicates):
        node_info = get_node_info(Tree(logic_form))
        label_types = [item[2] for item in node_info]
        entity_levels = [item[1] for item in node_info]
        start_positions = [int(item[0].split(";")[0]) for item in node_info]
        end_positions = [int(item[0].split(";")[1]) for item in node_info]

        tokens = input_text.split(" ")
        result = [f"Extract the content of {', '.join(missing_predicates)} from input sentence."]
        result_list = []
        idx = 1
         
        for label_type, ent_level, start_pos, end_pos in zip(label_types, entity_levels, start_positions, end_positions):
            # only extract the missing labels
            if label_type not in missing_predicates:
                continue 
            
            tmp = f"Step {idx}: " + f"[{label_type} " + " ".join(tokens[start_pos:end_pos+1]) + " ]"
            result.append(tmp)
            result_list.append(f"[{label_type} " + " ".join(tokens[start_pos:end_pos+1]) + " ]")
            idx += 1
        return "\n".join(result), result_list
    
    def query_generate(self, input_samples, top_k_demos, missing_predicates, **kwargs):
        str_query = ""
        
        # support example 
        for candidate in input_samples['candidates'][:top_k_demos]:
            input_text = f"Input: {candidate[0]}\n"
            input_text += self.build_output(candidate[0], candidate[1], missing_predicates)[0]
            input_text += f"\n\n"
            str_query += input_text
        
        # curent query 
        str_query +=  f"""Input: {input_samples['q_input']}\nExtract the content of {', '.join(missing_predicates)} from input sentence.\nStep 1:""" 
        return str_query
    
    
class LLMCoTMapQuery(LLMCoTQuery):
        
    def format_output(self, output_text, input_text, args=None):
        cleaned_output_text = output_text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").replace(input_text, "").strip()
        if args is not None and args.model == 'google/flan-ul2':
            cleaned_output_text = cleaned_output_text.replace('Step ', '\nStep ')
        output_sample = cleaned_output_text.split("\n\n")[0]
        predicted_steps = output_sample.split("\n")
        result = []
        pattern = r".* contains: "  # Regex pattern to match "Step [number]: "
        for step in predicted_steps:
            tmp = re.sub(pattern, "", step).strip()
            result.append(tmp)
        return result

    def build_output(self, input_text, logic_form):
        node_info = get_node_info(Tree(logic_form))
        label_types = [item[2] for item in node_info]
        entity_levels = [item[1] for item in node_info]
        start_positions = [int(item[0].split(";")[0]) for item in node_info]
        end_positions = [int(item[0].split(";")[1]) for item in node_info]
        parent_nodes = [item[3] for item in node_info]

        tokens = input_text.split(" ")
        result = []
        result_list = []
        idx = 1
        for label_type, ent_level, start_pos, end_pos, parent_node in zip(label_types, entity_levels, start_positions, end_positions, parent_nodes):
            tmp = f"{parent_node} contains: " + f"[{label_type} " + " ".join(tokens[start_pos:end_pos+1]) + " ]"
            result.append(tmp)
            result_list.append(f"[{label_type} " + " ".join(tokens[start_pos:end_pos+1]) + " ]")
            idx += 1
        return "\n".join(result), result_list
    
    def query_generate(self, input_samples, top_k_demos, *args, **kwargs):
        str_query = ""
        
        # support example 
        for candidate in input_samples['candidates'][:top_k_demos]:
            input_text = f"Input: {candidate[0]}\n"
            input_text += self.build_output(candidate[0], candidate[1])[0]
            input_text += f"\n\n"
            str_query += input_text
        
        # curent query 
        str_query +=  f"""Input: {input_samples['q_input']}\nROOT contains:""" 
        return str_query
    
    
# build querier 
querier_class_map = {
    "cot": LLMCoTQuery,
    "cotBottomup": LLMCoTBottomUpQuery,
    "cotbottomup": LLMCoTBottomUpQuery,
    "cotmerging": LLMMergingCoTQuery,
    "cotdfs": LLMDfsCoTQuery,
    "cotbfs": LLMBfsCoTQuery,
    "cotv3": LLMv3CoTQuery,
    "base": LLMBaseQuery,
    "zeroshot": LLMBaseQuery,
    "cotmap": LLMCoTMapQuery,
    "cotmod": LLMCoTModQuery,
    "cotQ": LLMCoTPredicateQuery,
    "cotQv2": LLMCoTPredicateQuery
}

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process ...')
    parser.add_argument('--model', type=str, default="meta-llama/Llama-2-70b-hf")
    parser.add_argument('--peft_lora_path', type=str, default=None)
    parser.add_argument('--predicate_retriever', type=str, default="oracle")
    parser.add_argument('--num_samples', type=str, default="all")
    parser.add_argument('--cur_chunk', type=int, default=1)
    parser.add_argument('--num_chunk', type=int, default=4)
    parser.add_argument('--top_k_demos', type=int, default=10)
    parser.add_argument('--path_folder_data', type=str, default='data/top/low_resource/reminder/')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--query_type', type=str, default="cot", help='choose from {cot, base, cotmap, cotmod}')
    parser.add_argument('--data_type', type=str, default="dev")
    parser.add_argument('--force_batch_infer', action="store_true", help='llm batch infer', default=False)
    parser.add_argument('--sim_method', type=str, default='rouge', help='choose from {sbert, rouge}')
    parser.add_argument('--sem_comb', action="store_true", help='semantic combination', default=False)
    parser.add_argument('--prepare_openai_query', action="store_true", help='generate openai query', default=False)
    
    args, unknown = parser.parse_known_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    if args.prepare_openai_query:
        querier_class = querier_class_map.get(args.query_type, querier_class_map['base'])
        querier:LLMBaseQuery = querier_class(args, None, None)

        querier.prepare_openai_query( )
        sys.exit() 

    print("Loading model ...")
    if args.peft_lora_path is not None:
        lora_config = LoraConfig.from_pretrained(args.peft_lora_path) 
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = LlamaForCausalLM.from_pretrained(lora_config.base_model_name_or_path, quantization_config=bnb_config, device_map={"":0})
        tokenizer = AutoTokenizer.from_pretrained(lora_config.base_model_name_or_path)
        model = get_peft_model(model, lora_config)
        model = PeftModel.from_pretrained(model, args.peft_lora_path) # load lora weights
        model.eval()
        args.model = lora_config.base_model_name_or_path
    else:
        if args.model == "google/flan-t5-xxl":
            tokenizer = T5Tokenizer.from_pretrained(args.model)
            model = T5ForConditionalGeneration.from_pretrained(args.model, device_map="auto", torch_dtype=torch.float16)
        elif args.model == "google/flan-ul2":
            tokenizer = AutoTokenizer.from_pretrained(f"google/flan-ul2")
            model = T5ForConditionalGeneration.from_pretrained(f"google/flan-ul2", device_map="auto", torch_dtype=torch.float16)
        elif args.model == "tiiuae/falcon-40b-instruct":
            tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-40b-instruct")
            model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b-instruct", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
        elif args.model == "huggyllama/llama-65b":
            tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-65b")
            model = LlamaForCausalLM.from_pretrained("huggyllama/llama-65b", device_map="auto", load_in_8bit=True)
        elif args.model == "huggyllama/llama-7b":
            tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
            model = LlamaForCausalLM.from_pretrained("huggyllama/llama-7b", device_map="auto")
        elif args.model == "meta-llama/Llama-2-7b-hf":
            tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
            model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="auto")
        elif args.model == "huggyllama/llama-13b":
            tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-13b")
            model = LlamaForCausalLM.from_pretrained("huggyllama/llama-13b", device_map="auto")
        elif args.model == "meta-llama/Llama-2-70b-hf":
            tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")
            model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf", device_map="auto", load_in_8bit=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", load_in_8bit=True)
        model.eval()
    
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    querier_class = querier_class_map.get(args.query_type, querier_class_map['base'])
    querier:LLMBaseQuery = querier_class(args, model, tokenizer)

    querier.generate_all_query()
