import argparse
import glob
import json
import logging
from pytorch_lightning import Trainer

import torch
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModel, RobertaPreTrainedModel
from transformers import get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from torch.optim import SGD, AdamW
from pytorch_lightning.callbacks import ModelCheckpoint
import pickle
import os, re
import random
import numpy as np
from pytorch_lightning import seed_everything
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from tree import Tree
from utils import get_list_labels

def set_random_seed(seed: int):
    """set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_random_seed(0)


class RelevantDocModel(RobertaPreTrainedModel):
    def __init__(
        self, 
        config,
        dropout
    ): 
        # init model 
        super().__init__(config)
        self.config = config

        self.roberta = AutoModel.from_config(config=self.config)  # Load pretrained bert
        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.classifier = nn.Linear(in_features=self.config.hidden_size, out_features=2, bias=True)

        # init pretrained language model - RoBERTa 
        # froze 10 layers, only train 2 last layer 
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
        for i in range(10):
            for param in self.roberta.encoder.layer[i].parameters():
                param.requires_grad = False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], description="Model", add_help=False)
        parser.add_argument("--dropout", default=0.1, type=float, help="dropout value")
        return parser

    def forward(self, **input_text_pair_ids):
        outputs = self.roberta(**input_text_pair_ids)
        h_cls = outputs[1]  # [CLS]
        return self.classifier(self.dropout(h_cls))
 
 
class RelevantPredicateClassifier(pl.LightningModule):
    def __init__(
        self, 
        args: argparse.Namespace,
        data_train_size=None
    ):
        """Initialize."""
        super().__init__() 

        format = '%(asctime)s - %(name)s - %(message)s'
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
            self.args = args
            logging.basicConfig(format=format, filename=os.path.join(self.args.log_dir, "run.log"), level=logging.INFO)
        else:
            args = argparse.Namespace(**args)
            self.args = args
            logging.basicConfig(format=format, filename=os.path.join(self.args.log_dir, "run.log"), level=logging.INFO)

        logging.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args))
        self.result_logger = logging.getLogger(__name__)
        self.result_logger.setLevel(logging.INFO)
        self.result_logger.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args))
    
        self.ignore_index = args.ignore_index

        # init model 
        # Load config from pretrained name or path 
        self.config = AutoConfig.from_pretrained(args.model_name_or_path)  # Load pretrained bert
        self.model = RelevantDocModel.from_pretrained(args.model_name_or_path, dropout=self.args.dropout)

        self.optimizer = self.args.optimizer
        self.data_train_size = data_train_size
        self.loss_function = F.cross_entropy
        
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.predict_step_outputs = []


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], description="Model", add_help=False)
        parser.add_argument("--dropout", default=0.1, type=float, help="dropout value")
        parser.add_argument("--final_div_factor", type=float, default=1e4,
                            help="final div factor of linear decay scheduler")
        parser.add_argument("--lr_scheduler", type=str, default="onecycle", ) 
        parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
        parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps used for scheduler.")
        parser.add_argument("--accumulate_grad_batches", default=1, type=int, help="accumulate_grad_batches.")

        parser.add_argument("--optimizer", choices=["adamw", "sgd", "adam"], default="adam",
                            help="loss type")
        parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--weight_decay", default=0.0001, type=float, help="Weight decay if we apply some.")

        return parser


    def test_step(self, batch, batch_idx):
        result = self._eval_step(batch, batch_idx)
        self.log(f"test/loss",result['loss'], prog_bar=True)
        self.test_step_outputs.append(result)
    
    def training_step(self, batch, batch_idx, return_y_hat=False, add_log=True):
        model_inputs, labels, sent_ids, predicate_ids  = batch
        y_hat = self.model(**model_inputs)

        loss = self.loss_function(y_hat, labels)
        if add_log:
            self.log('train/loss', loss, prog_bar=True)

        if return_y_hat:
            return loss, y_hat
        return loss
         

    def predict_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        result = self._eval_step(batch, batch_idx)
        self.log(f"dev/loss",result['loss'], prog_bar=True)
        self.validation_step_outputs.append(result)
        
    def _eval_step(self, batch, batch_idx):
        model_inputs, labels, sent_ids, predicate_ids  = batch
        loss, y_hat = self.training_step(batch, batch_idx, return_y_hat=True, add_log=False)
        
        result = {'loss': loss.detach().cpu(), 'y_hat': y_hat.detach().cpu(), 'labels': labels.detach().cpu(), 'sent_ids': sent_ids, 'predicate_ids': predicate_ids}
        return result

    @staticmethod
    def group_by_qid(sent_ids, predicate_ids, relevants, scores, topk=10):
        results = {}
        for i, question_id in enumerate(sent_ids):
            if question_id not in results:
                results[question_id] = {'predicate_ids': [], 'question_id': question_id, 'confidence_scores':[], 'rank':[], 'topk': []}
            if bool(relevants[i]):
                results[question_id]['predicate_ids'].append(predicate_ids[i])
                results[question_id]['confidence_scores'].append(scores[i].item())

            results[question_id]['rank'].append((predicate_ids[i], scores[i].item()))

        for question_id in results:
            results[question_id]['rank'].sort(key=lambda x: x[1], reverse=True)
            idx = 0
            while len(set(results[question_id]['topk'])) < topk and len(results[question_id]['rank']) > idx:
                if results[question_id]['rank'][idx][0] not in results[question_id]['topk']:
                    results[question_id]['topk'].append(results[question_id]['rank'][idx][0])
                idx += 1 

        return results
    
    def on_test_epoch_end(self):
        result = self._eval_on_epoch_end(self.test_step_outputs, main_prediction_enss=None, topk=20)
        self.log_dict(dict([(f"test/{k}", v) for k, v in result.items() if isinstance(v, (int, float))]), prog_bar=True, sync_dist=True)
        self.test_step_outputs.clear()
        self.result_logger.info(f"{dict([(k, v) for k, v in result.items() if isinstance(v, (int, float))])}")
    
    def on_validation_epoch_end(self):
        result = self._eval_on_epoch_end(self.validation_step_outputs, main_prediction_enss=None)
        self.log_dict(dict([(f"dev/{k}", v) for k, v in result.items() if isinstance(v, (int, float))]), prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()
        self.result_logger.info(f"{dict([(k, v) for k, v in result.items() if isinstance(v, (int, float))])}")

    def _eval_on_epoch_end(self, batch_parts, main_prediction_enss=None, topk=30):
        miss_q, main_prediction = main_prediction_enss if main_prediction_enss is not None else (None, None)
        
        # aggregrate values 
        def aggregrate_val(batch_parts):
            scores = torch.cat([torch.nn.Softmax(dim=1)(batch_output['y_hat'])[:,1] for batch_output in batch_parts],  dim=0)
            predictions = torch.cat([torch.argmax(batch_output['y_hat'], dim=1) for batch_output in batch_parts],  dim=0)
            labels = torch.cat([batch_output['labels']  for batch_output in batch_parts],  dim=0)
            sent_ids, predicate_ids = [], []
            for batch_output in batch_parts:
                sent_ids += batch_output['sent_ids']
                predicate_ids += batch_output['predicate_ids']

            gr_pred = self.group_by_qid(sent_ids, predicate_ids, predictions, scores, topk=topk)
            gr_gold = self.group_by_qid(sent_ids, predicate_ids, labels, scores)
            return gr_pred, gr_gold
        
        gr_pred, gr_gold =  aggregrate_val(batch_parts)
        main_gr_pred = None
        if main_prediction is not None:
            main_gr_pred, main_gr_gold =  aggregrate_val(main_prediction)
            for k in list(gr_pred.keys()):
                if k not in miss_q:
                    gr_pred[k] = main_gr_pred[k]
            
            # add top 1 ranking
            for k in list(gr_pred.keys()):
                if len(gr_pred[k]['predicate_ids']) == 0:
                    gr_pred[k]['predicate_ids'].append( main_gr_pred[k]['rank'][0][0]) 
                    gr_pred[k]['confidence_scores'].append( main_gr_pred[k]['rank'][0][1]) 

        def f1(p, r):
            if p + r == 0:
                return 0
            return (2*p*r)/(p + r)

        for q_id, gold_info in gr_gold.items():
            gold_predicate_ids = list(set(gold_info['predicate_ids']))
            
            # 
            # collect all c_id prediction and corresponding scores 
            pred_predicate_ids = []
            pred_c_scores = []
            for c_id, c_id_score in zip(gr_pred[q_id]['predicate_ids'], gr_pred[q_id]['confidence_scores']):

                if c_id not in pred_predicate_ids:
                    pred_predicate_ids.append(c_id)
                    pred_c_scores.append(c_id_score)
            gold_info['pred_predicate_ids'] = pred_predicate_ids
            gold_info['pred_c_scores'] = pred_c_scores


            # 
            # compute the correction 
            count_true = 0 
            for c_id in pred_predicate_ids:
                if c_id in gold_predicate_ids:
                    count_true += 1
            gold_info['retrieved'] = count_true 
            gold_info['p'] = count_true / len(pred_predicate_ids) if len(pred_predicate_ids) > 0 else 0.0
            gold_info['r'] = count_true / len(gold_predicate_ids)if len(gold_predicate_ids) > 0 else 0.0
            gold_info['f1'] = f1(gold_info['p'], gold_info['r'])

            # 
            # statistic top k 
            pred_predicate_ids_topk = gr_pred[q_id]['topk'] if main_gr_pred is None else main_gr_pred[q_id]['topk']
            count_true_topk = 0 
            for c_id in pred_predicate_ids_topk:
                if c_id in gold_predicate_ids:
                    count_true_topk += 1
            r = count_true_topk / len(gold_predicate_ids)if len(gold_predicate_ids) > 0 else 0.0
            gold_info['pred_predicate_ids_topk'] = pred_predicate_ids_topk
            gold_info['r-topk'] =  r

        retrieved = sum([e['retrieved'] for k, e in gr_gold.items()])
        return_results = {'retrieved': retrieved} 

        for metric in ['p', 'r', 'f1', 'r-topk']:
            _values = [e[metric] for k, e in gr_gold.items()]
            avg = sum(_values) / len(_values)
            return_results[f'{metric}'] = avg
             
        self.result_logger.info(f"total_q = {len(gr_gold.keys())}" )

        # collect miss query
        missed_q = [k for k, v in gr_pred.items() if len(v['predicate_ids']) == 0]
        return_results['miss_q'] = missed_q
        return_results['detail_pred'] = gr_gold
        self.result_logger.info(f"Miss_querry = {missed_q}" ) 

        return return_results



    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.optimizer == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters,
                              betas=(0.9, 0.98),  # according to RoBERTa paper
                              lr=self.args.lr,
                              eps=self.args.adam_epsilon,)
        elif self.optimizer == "adam":
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                          lr=self.args.lr,
                                          eps=self.args.adam_epsilon,
                                          weight_decay=self.args.weight_decay)
        else:
            optimizer = SGD(optimizer_grouped_parameters, lr=self.args.lr, momentum=0.9)
        num_gpus = len(self.args.gpus)
        t_total = int((self.data_train_size // (self.args.accumulate_grad_batches * num_gpus) + 1) * self.args.max_epochs)
        if self.args.lr_scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.args.lr, pct_start=float(self.args.warmup_steps/t_total),
                final_div_factor=self.args.final_div_factor,
                total_steps=t_total, anneal_strategy='linear'
            )
        elif self.args.lr_scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)
        elif self.args.lr_scheduler == "polydecay":
            if self.args.lr_mini == -1:
                lr_mini = self.args.lr / 5
            else:
                lr_mini = self.args.lr_mini
            scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, self.args.warmup_steps, t_total, lr_end=lr_mini)
        else:
            raise ValueError
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

class SemDataPreprocessor:
    def __init__(self, tokenizer, max_seq_length) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    @staticmethod
    def aggregate_data(raw_data_path_x, raw_data_path_y, predicate_vocab=None):
        with open(raw_data_path_y) as f:
            gold_y = [l.strip() for l in f.readlines()]
            sample_predicates = [[str(e)[2:-1] for e in re.findall(r'\( [^ ]+ ', lg,  re.DOTALL)] for lg in gold_y]
        with open(raw_data_path_x) as f:
            input_sents = [l.strip() for l in f.readlines()]
        
        # generate order vocab
        if predicate_vocab is None:
            predicate_vocab = set('[none]')
            for e in sample_predicates:
                predicate_vocab = predicate_vocab.union(set(e))
        predicate_vocab = list(predicate_vocab)
        predicate_vocab.sort()

        # pair predicate and sentence data 
        samples = []
        for i_sent, input_sent in enumerate(input_sents):
            for predicate in predicate_vocab:
                # gold label of pair 
                gold_predicates = sample_predicates[i_sent]
                if len(gold_predicates) == 0:
                    gold_predicates = gold_predicates+ ['[none]']
                    
                samples.append({'sent_id': i_sent, 
                                'sent_content': input_sent, 
                                'predicate_id': predicate, 
                                'predicate_content':predicate.replace("_", " "), 
                                'label': predicate in gold_predicates})
        return samples, predicate_vocab
    
    def __call__(self, mini_batch):
        max_seq_length = min(self.max_seq_length, self.tokenizer.model_max_length)

        predicate_ids = [e['predicate_id'] for e in mini_batch]
        predicate_texts = [e['predicate_content'] for e in mini_batch]
        sent_ids = [e['sent_id'] for e in mini_batch]
        sent_texts = [e['sent_content'] for e in mini_batch]
        input_data = self.tokenizer(predicate_texts, sent_texts, padding='longest', 
                                    max_length=max_seq_length, truncation=True, return_tensors='pt')

        labels = torch.LongTensor([e['label'] for e in mini_batch])

        return (input_data, labels, sent_ids, predicate_ids)

class HierachicalSemDataPreprocessor:
    def __init__(self, tokenizer, max_seq_length) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    @staticmethod
    def aggregate_data(raw_data_path, vocab_path):
        with open(raw_data_path) as f:
            data = [l.strip().split("	") for l in f.readlines()]            
            output_sents = [e[2] for e in data]
            input_sents = [e[1] for e in data]
        
        # generate order vocab
        predicate_vocab = json.load(open(vocab_path))
        predicate_vocab = list(predicate_vocab.keys())+ ['[none]']
        predicate_vocab.sort()

        sample_predicates = []
        for i_s, sample in enumerate(output_sents):
            cur_predicates = [ ]
            for predicate in predicate_vocab:
                if f'[{predicate} ' in sample:
                    cur_predicates.append(predicate)
            sample_predicates.append(cur_predicates)
        
        # pair predicate and sentence data 
        samples = []
        for i_sent, input_sent in enumerate(input_sents):
            for predicate in predicate_vocab:
                # gold label of pair 
                gold_predicates = sample_predicates[i_sent]
                if len(gold_predicates) == 0:
                    gold_predicates = gold_predicates + ['[none]']
                    
                samples.append({'sent_id': i_sent, 
                                'sent_content': input_sent, 
                                'predicate_id': predicate, 
                                'predicate_content':predicate.replace("_", " ").replace(":", " : "), 
                                'label': predicate in gold_predicates})
        return samples, predicate_vocab
    
    @staticmethod
    def aggregate_data_from_json(data_path, predicate_vocab=None, args=None):
        data = json.load(open(data_path))
        if 'train' in data_path and args.rate_train!= 1.0:
            random.shuffle(data)      
            data = data[:int(args.rate_train*len(data))]     
             
        output_sents = [e['org_label'] for e in data]
        input_sents = [e['context'] for e in data]
        

        sample_predicates = []
        for i_s, sample in enumerate(output_sents):            
            sample_predicates.append(get_list_labels(sample))
            
        if predicate_vocab is None:
            # generate order vocab
            predicate_vocab = set(['[none]'])
            predicate_vocab = list(predicate_vocab.union(*sample_predicates))
            predicate_vocab.sort()
        
        # pair predicate and sentence data 
        samples = []
        for i_sent, input_sent in enumerate(input_sents):
            # gold label of pair 
            gold_predicates = sample_predicates[i_sent]
            if len(gold_predicates) == 0:
                gold_predicates = gold_predicates + ['[none]']
                
            for predicate in predicate_vocab:
                samples.append({'sent_id': i_sent, 
                                'sent_content': input_sent, 
                                'predicate_id': predicate, 
                                'predicate_content':predicate.replace("_", " ").replace(":", " : "), 
                                'label': predicate in gold_predicates})
        return samples, predicate_vocab
    
    
    def __call__(self, mini_batch):
        max_seq_length = min(self.max_seq_length, self.tokenizer.model_max_length)

        predicate_ids = [e['predicate_id'] for e in mini_batch]
        predicate_texts = [e['predicate_content'] for e in mini_batch]
        sent_ids = [e['sent_id'] for e in mini_batch]
        sent_texts = [e['sent_content'] for e in mini_batch]
        input_data = self.tokenizer(predicate_texts, sent_texts, padding='longest', 
                                    max_length=max_seq_length, truncation=True, return_tensors='pt')

        labels = torch.LongTensor([e['label'] for e in mini_batch])

        return (input_data, labels, sent_ids, predicate_ids)
if __name__=="__main__":

    # training+model args
    parser = argparse.ArgumentParser(description="Training Args")
    parser = RelevantPredicateClassifier.add_model_specific_args(parser)
        
    # parser.add_argument("--data_dir", type=str, required=True, help="data dir")
    parser.add_argument("--log_dir", type=str, default=".", help="log dir")
    parser.add_argument("--rate_train", type=float, default=1.0, help="training samples rate")
    parser.add_argument("--max_keep_ckpt", default=1, type=int, help="the number of keeping ckpt max.")
    parser.add_argument("--pretrained_checkpoint", default=None, type=str, help="pretrained checkpoint path")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--model_name_or_path", type=str, default='roberta-large',  help="pretrained model name or path")
    parser.add_argument("--ignore_index",  type=int, default=-100)
    parser.add_argument("--max_epochs", default=5, type=int, help="Max training epochs.")
    parser.add_argument("--max_seq_length",  type=int, default=512, help="Max seq length for truncating.")
    parser.add_argument("--no_train", action="store_true", default=False, help="Do not training.")
    parser.add_argument("--no_test", action="store_true", default=False, help="Do not test.")
    parser.add_argument("--no_dev", action="store_true", default=False, help="Do not dev at last.")
    parser.add_argument("--gpus", nargs='+', default=[0], type=int, help="Id of gpus for training")
    parser.add_argument("--ckpt_steps", default=1000, type=int, help="number of training steps for each checkpoint.")
    parser.add_argument("--file_output_id", default="allEnss", type=str, help="Id of submission")
    parser.add_argument("--civi_code_path", default="data/parsed_civil_code/en_civil_code.json", type=str, help="civil code path")
    parser.add_argument("--main_enss_path", default="settings/bert-base-japanese-whole-word-masking_5ckpt_150-newE5Seq512L2e-5/datout/test_{}_5_80_0015.txt", type=str, help="Id of submission")

    opts = parser.parse_args()
    if opts.pretrained_checkpoint is not None and not opts.pretrained_checkpoint.endswith(".ckpt"):
        path_pretrained_folder = opts.pretrained_checkpoint
        opts.pretrained_checkpoint = glob.glob(f"{opts.pretrained_checkpoint}/*.ckpt")[0]
        print(f"Found checkpoint - {opts.pretrained_checkpoint}")

    # load pretrained_checkpoint if it is set 
    if opts.pretrained_checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(path_pretrained_folder, use_fast=True, 
                                                  config=AutoConfig.from_pretrained(path_pretrained_folder))
        model = RelevantPredicateClassifier.load_from_checkpoint(opts.pretrained_checkpoint, args=opts)
        max_seq_length=model.args.max_seq_length
    else:
        config = AutoConfig.from_pretrained(opts.model_name_or_path)
        config.save_pretrained(opts.log_dir)
        tokenizer = AutoTokenizer.from_pretrained(opts.model_name_or_path, use_fast=True, max_seq_length=opts.max_seq_length)
        tokenizer.save_pretrained(opts.log_dir)
        max_seq_length=opts.max_seq_length

    #
    # Data loader  
    coliee_data_preprocessor =HierachicalSemDataPreprocessor(tokenizer, max_seq_length=max_seq_length)
    raw_data_path = f'{opts.log_dir}/mrc-ner.train'
    
    train_samples, predicate_vocab = HierachicalSemDataPreprocessor.aggregate_data_from_json(raw_data_path, args=opts)
    train_loader = DataLoader(train_samples, batch_size=opts.batch_size, collate_fn=coliee_data_preprocessor, shuffle=True)
    
    dev_samples, _ = HierachicalSemDataPreprocessor.aggregate_data_from_json(raw_data_path.replace('train', 'dev'), predicate_vocab=predicate_vocab)
    dev_loader = DataLoader(dev_samples, batch_size=opts.batch_size, collate_fn=coliee_data_preprocessor, shuffle=False)
    
    test_samples, _ = HierachicalSemDataPreprocessor.aggregate_data_from_json(raw_data_path.replace('train', 'test'), predicate_vocab=predicate_vocab)
    test_loader = DataLoader(test_samples, batch_size=opts.batch_size, collate_fn=coliee_data_preprocessor, shuffle=False)

    # model 
    if not opts.pretrained_checkpoint: 
        model = RelevantPredicateClassifier(opts, data_train_size=len(train_loader))
    else:
        model.data_train_size = len(train_loader)
    
    # trainer
    checkpoint_callback = ModelCheckpoint(dirpath=opts.log_dir, save_top_k=opts.max_keep_ckpt, 
                                          auto_insert_metric_name=True, mode="max", monitor="dev/r", 
                                          )
    trainer = Trainer(max_epochs=opts.max_epochs, 
                      accelerator='gpu' if len(opts.gpus) > 0 else 'cpu', 
                      devices=opts.gpus, 
                      callbacks=[checkpoint_callback], 
                      default_root_dir=opts.log_dir, 
                      accumulate_grad_batches=2,
                      val_check_interval=0.1
                      )

    if not opts.no_train:
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=dev_loader)

    # predict 
    for data_loader, file_out, samples in zip ([ dev_loader, test_loader, train_loader], 
                                      [ f'{opts.log_dir}/dev.sent_predicate',
                                        f'{opts.log_dir}/test.sent_predicate',  
                                        f'{opts.log_dir}/train.sent_predicate'],
                                      [dev_samples,test_samples,  train_samples]):
        data_type = file_out.split("/")[-1].split(".")[0]
        all_input_sent = dict(list(set([(e['sent_id'], e['sent_content']) for e in samples])))
        num_sent = len(all_input_sent)
        all_input_sent = [all_input_sent[idx] for idx in range(num_sent)]
        
        
        if not os.path.exists(file_out+'_prediction.pkl'):
            cpkt_path = checkpoint_callback.best_model_path if opts.pretrained_checkpoint is None else opts.pretrained_checkpoint
            predictions = trainer.predict(model, data_loader, ckpt_path=cpkt_path)
            pickle.dump(predictions, open(file_out+'_prediction.pkl', 'wb'))
        else:
            predictions = pickle.load(open(file_out+'_prediction.pkl', 'rb'))
            
        cur_checkpoint_ret = model._eval_on_epoch_end(predictions)
        print(cur_checkpoint_ret["f1"])

        dict_pred = dict([(i_sent, pred['pred_predicate_ids']) for i_sent, pred in cur_checkpoint_ret['detail_pred'].items()])
        all_pred = [dict_pred[i] for i in range(len(dict_pred))]
        prediction_text = []
        for i_sent, pred_predicates in enumerate(all_pred):
            input_x = all_input_sent[i_sent]
            output_y = " ".join(pred_predicates)
            prediction_text.append(f'{input_x} [sep] {output_y}')
        with open (file_out, 'wt') as f:
            f.write("\n".join(prediction_text))
