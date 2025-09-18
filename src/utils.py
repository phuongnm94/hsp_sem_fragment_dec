from argparse import Namespace
from glob import glob
import os
import json
import re
from sklearn.feature_extraction import _stop_words
import torch
from nltk.stem import PorterStemmer

import csv

def save_list_to_tsv(file_path, data):
    with open(file_path, 'w', newline='') as tsv_file:
        writer = csv.writer(tsv_file, delimiter='\t')
        for item in data:
            writer.writerow([item])

def load_tensor(path, cpu=False):
    if cpu:
        return torch.load(path, map_location=torch.device('cpu'))
    return torch.load(path)

def standardize_data(row):
    row = re.sub(r"[\.,\?]+$-", "", row)
    row = row.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ").replace("?", " ") \
        .replace("(", " ").replace(")", " ") \
        .replace("<", " ").replace(">", " ")
    row = row.strip().lower()
    return row

def remove_stop_words(tokens):
    tokenized_doc = []
    for token in tokens:
        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)
    return tokenized_doc

ps = PorterStemmer()
def stem_word(tokens):
    tokenized_doc = []
    for token in tokens:
        tokenized_doc.append(ps.stem(token))
    return tokenized_doc

def tokenize_text(text, standard_data=True, clean= False, stem=False):
    if standard_data:
        tmp = standardize_data(text)
    else:
        tmp = text
    tmp = tmp.split()
    if clean:
        tmp = remove_stop_words(tmp)
    if stem:
        tmp = stem_word(tmp)
    return tmp

def get_file_paths(dir_path):
    return glob(os.path.join(dir_path, "*"), recursive = True)


def read_lines(path):
    with open(path, 'r') as file1:
        lines = file1.readlines()
    return lines

def write_txt_file(content, path):
    with open(path, 'w') as f:
        f.write(content)


def read_json_file(path):
    with open(path, 'r') as json_file:
        json_data = json.load(json_file)
    return json_data

def write_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        text_data = json.dumps(data, indent=4, ensure_ascii=False)
        f.write(text_data)
    return 

def lines_to_paragraphs(lines: list):
    paragraphs = {}
    paragraph = ''
    key = 'intro'
    for line in lines:
        if not (line == 'Summary:' or line.strip('[').strip(']').strip().isnumeric()):
            paragraph = paragraph + line + '\n'
        else:
            if key in paragraphs.keys():
                para = paragraphs.get(key)
                para.append(paragraph)
                paragraphs.update({key: para})
            else:
                paragraphs.update({key:[paragraph]})
            key = line
            paragraph = ''

    if key not in paragraphs.keys():
        paragraphs.update({key:[paragraph]})

    if not paragraphs:
        for line in lines:
            if not line.strip('[').strip()[0].isdigit():
                paragraph = paragraph + ' ' + line
            else:
                if key == 'intro':
                    paragraphs.update({key: [paragraph]})
                paragraphs.update({line.split(']')[0] + ']': [']'.join(line.split(']')[1:])]})
    return paragraphs

def only_english(paragraphs: dict):
    # check intro where the english version is, is it the first one or the second one of the paragraphs?
    # but only if there are multiple options for the paratgraph
    freen = '[English language version follows French language version]'
    freen2 = '[La version anglaise vient à la suite de la version française]'
    enfre = '[French language version follows English language version]'
    enfre2 = '[La version française vient à la suite de la version anglaise]'

    if enfre in paragraphs.get('intro')[0] or enfre2 in paragraphs.get('intro')[0]:
        for key, value in paragraphs.items():
            if len(value) > 1:
                paragraphs.update({key: [value[0]]})
    elif freen in paragraphs.get('intro')[0] or freen2 in paragraphs.get('intro')[0]:
        for key, value in paragraphs.items():
            if len(value) > 1:
                paragraphs.update({key: [value[1]]})
    return paragraphs


def only_string_in_dict(paragraphs: dict):
    for key, value in paragraphs.items():
        paragraphs.update({key: value[0]})
    return paragraphs




#!/usr/bin/env python

import math
import numpy as np
from multiprocessing import Pool, cpu_count

"""
All of these algorithms have been taken from the paper:
Trotmam et al, Improvements to BM25 and Language Models Examined
Here we implement all the BM25 variations mentioned. 
"""


class BM25:
    def __init__(self, corpus, tokenizer=None):
        self.corpus_size = len(corpus)
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer
        self.q_freq_cache = {}

        if tokenizer:
            corpus = self._tokenize_corpus(corpus)

        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus):
        print("Loading BM25 model ...")
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word]+=1
                except KeyError:
                    nd[word] = 1

        self.avgdl = num_doc / self.corpus_size
        return nd

    def _tokenize_corpus(self, corpus):
        pool = Pool(cpu_count())
        tokenized_corpus = pool.map(self.tokenizer, corpus)
        return tokenized_corpus

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        raise NotImplementedError()

    def get_top_n(self, query, documents, n=5):

        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]


class BM25Okapi(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query:
        :return:
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            if q in self.q_freq_cache:
                q_freq = self.q_freq_cache[q]
            else:
                q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
                self.q_freq_cache[q] = q_freq
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score.tolist()


class BM25Plus(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=1):
        # Algorithm specific parameters
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = math.log((self.corpus_size + 1) / freq)
            self.idf[word] = idf

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            if q in self.q_freq_cache:
                q_freq = self.q_freq_cache[q]
            else:
                q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
                self.q_freq_cache[q] = q_freq
            score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
                                               (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
                                               (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
        return score.tolist()

#=========================== Tree Utils ===============================

from tree import Tree, Root, Token

def list_nonterminals(cur_node, node_info, node_level):
    for child in cur_node.children:
        if type(child) != Root and type(child) != Token:
            try:
                tmp = get_span(child)
                tmp = str(tmp[0]) + ";" + str(tmp[1]-1)
                node_info.append((tmp, node_level, child.label, cur_node.label))
            except Exception as e:
                print(f'[Err] get span of  {child}')
    for child in cur_node.children:
        if type(child) != Root and type(child) != Token:
            list_nonterminals(child, node_info, node_level+1)
    return node_info

def get_span(node):
    return node.get_token_span()

def get_node_info(tree):
    node_info = list_nonterminals(tree.root, [], 1)
    return node_info

def get_list_labels(output_text):
    tree = Tree(output_text)
    return set([item[2] for item in get_node_info(tree)])

def get_local_structures(cur_node, cur_LS, node_info, node_level):
    for child in cur_node.children:
        if type(child) != Root and type(child) != Token:
            cur_LS.append("->" + child.label)
        else:
            node_info.append(cur_LS)

    for child in cur_node.children:
        if type(child) != Root and type(child) != Token:
            get_local_structures(child, node_info, node_level+1)
    return node_info


def get_all_local_structures(tree):
    all_local_structures = all_local_structures(tree.root, "ROOT", [], 1)
    return all_local_structures

def update_pos(start_positions, end_positions, cursor):
    new_start_positions = [start+1 if start >= cursor else start for start in start_positions ]
    new_end_positions = [end+1 if end >= cursor else end for end in end_positions ]
    return new_start_positions, new_end_positions


def extract_sentences_from_parsed_tree(parsed_tree):
    sentences = []
    sentences.append(parsed_tree["context"])
    start_positions, end_positions, entities = [], [], []
    for label in parsed_tree["label"]:
        start_positions.append(int(label[0].split(";")[0]))
        end_positions.append(int(label[0].split(";")[1]))
        entities.append(label[2])
    
    cur_words = parsed_tree["context"].split()
    for idx in range(len(start_positions)):
        # Update context and postions
        cur_words.insert(end_positions[idx]+1, f"]")
        cur_words.insert(start_positions[idx], f"[{entities[idx]}")
        start_positions, end_positions = update_pos(start_positions, end_positions, start_positions[idx])
        start_positions, end_positions = update_pos(start_positions, end_positions, end_positions[idx]+1)
        sentences.append(" ".join(cur_words))
    return sentences

def get_list_non_terminal_nodes(output_text):
    gold_tree = Tree(output_text)
    return [item[2] for item in get_node_info(gold_tree)]


def separate_steps_in_full_tree(input_text, output_text):
    gold_tree = Tree(output_text)
    label = get_node_info(gold_tree)
    return extract_sentences_from_parsed_tree({"context": input_text, "label": label})

def format_output_cot(output_text, input_text):
    cleaned_output_text = output_text.replace("<s>", "").replace("<pad> ", "").replace("</s>", "").replace(input_text, "")
    output_sample = cleaned_output_text.split("\n\n")[0]
    predicted_text = output_sample.split("\n")[-1].replace("Output:", "").strip()
    return predicted_text

# input_text = """Input: are there any Christmas parties this week
# Step 1: [IN:GET_EVENT are there any Christmas  parties this week ]
# Step 2: [IN:GET_EVENT are there any [SL:DATE_TIME Christmas ] parties this week ]
# Step 3: [IN:GET_EVENT are there any [SL:DATE_TIME Christmas ] [SL:CATEGORY_EVENT parties ] this week ]
# Output: [IN:GET_EVENT are there any [SL:DATE_TIME Christmas ] [SL:CATEGORY_EVENT parties ] [SL:DATE_TIME this week ] ]

# Input: Are there any Christmas parties
# Step 1:"""

# output_text = """<s>Input: are there any Christmas parties this week
# Step 1: [IN:GET_EVENT are there any Christmas  parties this week ]
# Step 2: [IN:GET_EVENT are there any [SL:DATE_TIME Christmas ] parties this week ]
# Step 3: [IN:GET_EVENT are there any [SL:DATE_TIME Christmas ] [SL:CATEGORY_EVENT parties ] this week ]
# Output: [IN:GET_EVENT are there any [SL:DATE_TIME Christmas ] [SL:CATEGORY_EVENT parties ] [SL:DATE_TIME this week ] ]

# Input: Are there any Christmas parties
# Step 1:  [IN:GET_EVENT are there any Christmas  parties ]
# Step 2:  [IN:GET_EVENT are there any [SL:DATE_TIME Christmas ] parties ]
# Step 3:  [IN:GET_EVENT are there any [SL:DATE_TIME Christmas ] [SL:CATEGORY_EVENT parties ] ]
# Output: [IN:GET_EVENT are there any [SL:DATE_TIME Christmas ] [SL:CATEGORY_EVENT parties ] ]

# Input: Are there any Christmas parties this week
# Step 1:  [IN:GET_EVENT are there any Christmas  parties this week ]
# Step 2:  [IN:GET_EVENT are there any [SL:DATE_TIME Christmas ] parties this week ]
# Step 3:  [IN:GET_EVENT are there any [SL:DATE_TIME Christmas ] [SL:CATEGORY_EVENT parties ] this week ]
# Output: [IN:GET_EVENT are there any [SL:DATE_TIME Christmas ] [SL:CATEGORY_EVENT parties ] [SL:DATE_TIME this week"""

if __name__=="__main__":
    from query_llm import LLMMergingCoTQuery, LLMCoTQuery, LLMDfsCoTQuery, LLMBfsCoTQuery, LLMCoTBottomUpQuery
    logic_form = "[IN:GET_EVENT are there any [SL:DATE_TIME [SL:CATEGORY_EVENT [SL:CATEGORY_EVENT2 parties ] [SL:CATEGORY_EVENT3 parties ] ] ] [SL:CATEGORY_EVENT [SL:DATE_TIME parties this1 week1 ] ] [SL:DATE_TIME this week ] ]"
    input_text = "are there any parties parties parties this1 week1 this week"
    
    querier = LLMCoTBottomUpQuery(Namespace(path_folder_data=None), None, None)
    
    print(querier.build_output( input_text, logic_form, add_final_step=True)[0])
    cots = querier.build_output(input_text, logic_form, add_final_step=True)[1] 
    print( querier.restore_full_logical_form(cots, input_text))
    print(querier.get_lg_tree(logic_form))