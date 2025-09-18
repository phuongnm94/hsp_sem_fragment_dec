#!/usr/bin/bash
#

#PBS -N snips_exp
#PBS -j oe -l ngpus=1
#PBS -q GPU-1 
#PBS -o pbs_run_hsp_sem_fragment_dec.log
#PBS -e pbs_run_hsp_sem_fragment_dec.err.log

source ~/miniconda3/etc/profile.d/conda.sh

cd ~/hsp_sem_fragment_dec
conda activate ./env_spllm/
ROOT_DIR=$(pwd)

DATA_DIR="${ROOT_DIR}/data/snips"
DATA_PREDICATE_DIR="${DATA_DIR}/predicate_retrieve"
PREDICATE_RETRIEVER="robertapair"
SIM_METHOD="rouge"

# ===========================
# Step 1: Train a predicate retriever to generate predicate name (semantic fragment id)
mkdir -p ${DATA_PREDICATE_DIR} && cd $DATA_PREDICATE_DIR && ln -s ../mrc-ner.train; ln -s ../mrc-ner.test; ln -s ../mrc-ner.dev
cd $ROOT_DIR && python  ${ROOT_DIR}/src/predicate_retrievier.py \
    --log_dir  ${DATA_DIR}/predicate_retrieve   \
    --max_epochs 3 --lr 5e-6 \
    # --no_train --pretrained_checkpoint ${DATA_DIR}/predicate_retrieve

## ===========================
## Step 2: Generate demonstration data using CombineSF with ROUGE
for d_type in dev test train ;
do
    ln -s  ${DATA_PREDICATE_DIR}/${d_type}.sent_predicate ${DATA_DIR}/${d_type}.sent_predicate

    # # data generate
    python ${ROOT_DIR}/src/demo_generator.py \
    --path_folder_data ${DATA_DIR} \
    --predicate_retriever ${PREDICATE_RETRIEVER} \
    --data_prefix "" \
    --data_type ${d_type} \
    --sim_method ${SIM_METHOD} 

    mkdir -p ${DATA_DIR}/${d_type} 
    ln -s  ${DATA_DIR}/golds_${d_type}_all.tsv ${DATA_DIR}/${d_type}/golds_${d_type}_all.tsv
    ln -s  ${DATA_PREDICATE_DIR}/${d_type}.sent_predicate ${DATA_DIR}/${d_type}/${d_type}.sent_predicate
    ln -s  ${DATA_DIR}/${PREDICATE_RETRIEVER}_all_${SIM_METHOD}.${d_type}.json ${DATA_DIR}/${d_type}/${PREDICATE_RETRIEVER}_all_${SIM_METHOD}.${d_type}.json

done

## ===========================
## Step 3: Infer with LLMs 
for d_type in dev test ;
do 
    for topk in 10 ;
    do
        python ${ROOT_DIR}/src/query_llm.py \
        --model  "meta-llama/Llama-2-7b-hf" \
        --predicate_retriever robertapair \
        --num_chunk  1 \
        --path_folder_data "${DATA_DIR}/dev" \
        --top_k_demos $topk --batch_size 4  \
        --data_type "dev" \
        --query_type "cotdfs" \
        --sim_method "rouge"  \
        2>&1 | tee ${DATA_DIR}/run_server.log 
    done
done 
