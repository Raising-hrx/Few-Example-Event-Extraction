import json
import copy
import numpy as np
import os
import sys
import random
from tqdm import tqdm
import argparse


import torch
from transformers import BertTokenizer, BertForMaskedLM
from transformers import AlbertTokenizer, AlbertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM

import spacy
spacy_nlp = spacy.load('en_core_web_sm')

def get_noun_phrases(spacy_nlp, tokens):
    doc = spacy.tokens.doc.Doc(spacy_nlp.vocab, words=tokens)
    for name, proc in spacy_nlp.pipeline:
        doc = proc(doc)
    assert [i.text for i in doc] == tokens
    
    noun_phrases = []
    for np in doc.noun_chunks:
        noun_phrases.append({
            'position': [np.start,np.end],
            'text': np.text,
            'head_text': np.root.text,
            'head_lemma': np.root.lemma_,
        })
    return noun_phrases

def parse_phrase(sapcy_nlp, phrase):
    doc = spacy_nlp(phrase)
    for token in doc:
        if token.dep_ == 'ROOT':
            break

    return token.lemma_, token.text

# PLM 
def load_filling_model(filling_model_name):
    if 'roberta' in filling_model_name:
        # roberta-large # roberta-base
        filling_tokenizer = RobertaTokenizer.from_pretrained(filling_model_name) 
        filling_model = RobertaForMaskedLM.from_pretrained(filling_model_name)  
    elif 'albert' in filling_model_name:
        # albert-base-v2  # albert-xxlarge-v2
        filling_tokenizer = AlbertTokenizer.from_pretrained(filling_model_name) 
        filling_model = AlbertForMaskedLM.from_pretrained(filling_model_name)
    elif 'bert' in filling_model_name:
       # bert-base-uncased  bert-large-uncased bert-large-uncased-whole-word-masking  
        filling_tokenizer = BertTokenizer.from_pretrained(filling_model_name) 
        filling_model = BertForMaskedLM.from_pretrained(filling_model_name)
    else:
        raise NotImplementedError
    return filling_tokenizer, filling_model

def replace_sentences_LM_pos(tokens, replace_position, trace_positions = [], topk = 10):
    """
    use PLM with single mask for mask filling
    replace_position e.g., [ [0,1],[3,5],[5,7],[8,9] ]
    """
    new_tokens = copy.deepcopy(tokens)
    new_trace_positions = copy.deepcopy(trace_positions)
    
    replace_scores = []
    for ri, pos in enumerate(replace_position):
        r_start, r_end =  pos
        assert r_end > r_start

        input_tokens = new_tokens[:r_start] + [filling_tokenizer.mask_token] + [' ']*(r_end-r_start-1) + new_tokens[r_end:]

        input_ids = filling_tokenizer(' '.join(input_tokens), return_tensors='pt')['input_ids']
        with torch.no_grad():
            logits = filling_model(input_ids.to(device)).logits # [bs,token_len,vocab_len]
            logits = logits[0]  
            logits = logits.cpu()

        masked_indexs = (input_ids[0] == filling_tokenizer.mask_token_id).nonzero(as_tuple=False).item() # only one mask

        probs = logits[masked_indexs].softmax(dim=0)
        values, predictions = probs.sort(descending = True)
        
        select_index = random.randint(0,topk) 
        word = filling_tokenizer.decode([predictions[select_index]])
        score = values[select_index]
        
        if len(filling_tokenizer.encode(word, add_special_tokens=False)) == 0:
            new_tokens = new_tokens
        else:
            new_tokens = new_tokens[:r_start] + [word] + ['']*(r_end-r_start-1) + new_tokens[r_end:]

        replace_scores.append(float(score))
        
    for nt,t in zip(new_trace_positions,trace_positions):       
        assert new_tokens[nt[0]:nt[1]] == tokens[t[0]:t[1]]  # check traced position tokens
    assert len(new_tokens) == len(tokens)

    return new_tokens, replace_scores

# KB
def load_probase(parobase_path):
    pb_data = []
    with open(parobase_path,'r') as f:
        for line in f.readlines():
            pb_data.append(line.strip().split('\t'))
    pb_data = pb_data[:-1]

    entity2concept = {}
    concept2entity = {}

    for con, ent, fre in pb_data:
        ent = ent.lower()
        con = con.lower()

        if ent not in entity2concept.keys():
            entity2concept[ent] = []
        if con not in concept2entity.keys():
            concept2entity[con] = []

        entity2concept[ent].append([con,fre])
        concept2entity[con].append([ent,fre])
    
    return entity2concept, concept2entity

def replace_sentences_PB(tokens,replace_nps,trace_positions = [],topk_concept = 3, topk_entity = 15, min_fre = 0):
    """
    use probase to replace noun phrase
    """
    new_tokens = copy.deepcopy(tokens)
    new_trace_positions = copy.deepcopy(trace_positions)
    
    replace_scores = []
    for ri, np in enumerate(replace_nps):
        r_start, r_end =  np['position']
        assert r_end > r_start

        np_head_lemma = np['head_lemma'].lower()
        np_text = np['text']
        
        tmp_concepts_list = entity2concept.get(np_head_lemma,[])[:topk_concept]
        concepts_list = []
        for con, fre in tmp_concepts_list:
            if int(fre) > min_fre:
                concepts_list.append(con)
        
        entity_list = []
        for con in concepts_list:
            for ent, fre in concept2entity[con][:topk_entity//topk_concept]:
                if int(fre) > min_fre:
                    entity_list.append(ent) 

        if not entity_list:
            entity_list = [np_text]
        
        select_index = random.randint(0,len(entity_list)-1)
        word = entity_list[select_index]
        score = 1.0 
        
        new_tokens = new_tokens[:r_start] + [word] + ['']*(r_end-r_start-1) + new_tokens[r_end:]
        replace_scores.append(float(score))
        
        if word != np_text:
            print(f'----PB----- {np_head_lemma} ==> {"/".join(concepts_list)} ==> {word} ')
        
    for nt,t in zip(new_trace_positions,trace_positions):       
        assert new_tokens[nt[0]:nt[1]] == tokens[t[0]:t[1]]  # check traced position tokens
        
    assert len(tokens) == len(new_tokens)

    return new_tokens, replace_scores


# post-processing
def delect_empty(data_item):
    tokens = data_item['tokens']
    failed = False

    empty_idx = []
    for idx,t in enumerate(tokens):
        if len(tokenizer.encode(t,add_special_tokens=False)) == 0:
            empty_idx.append(idx)

    new_tokens = [t for idx,t in enumerate(tokens) if idx not in empty_idx]

    def change_idx(pos,empty_idx):
        pre_num = len([idx for idx in empty_idx if idx < pos])
        return pos - pre_num

        
    for ent in data_item['entity_mentions']:
        ent['start'] = change_idx(ent['start'],empty_idx)
        ent['end'] = change_idx(ent['end'],empty_idx)

    for eve in data_item['event_mentions']:
        eve['trigger']['start'] = change_idx(eve['trigger']['start'],empty_idx)
        eve['trigger']['end'] = change_idx(eve['trigger']['end'],empty_idx)

        assert eve['trigger']['end'] > eve['trigger']['start'],eve['trigger']

        acceptable_triggers = []
        acceptable_triggers.append(" ".join(new_tokens[eve['trigger']['start']:eve['trigger']['end']]))
        acceptable_triggers.append("".join(new_tokens[eve['trigger']['start']:eve['trigger']['end']]))
        acceptable_triggers += [at.lower() for at in acceptable_triggers]
        if  eve['trigger']['text'] not in acceptable_triggers:
            failed = True

        for arg in eve['arguments']:
            orig_arg = ''.join(tokens[arg['start']:arg['end']])
            
            arg['start'] = change_idx(arg['start'],empty_idx)
            arg['end'] = change_idx(arg['end'],empty_idx)
            
            new_arg = ''.join(new_tokens[arg['start']:arg['end']])
            
            if new_arg != orig_arg:
                failed = True
            if arg['end'] <= arg['start']:
                failed = True

    data_item['tokens'] = new_tokens
    
    return data_item, failed

def split_multi_token(data_item):
    tokens = data_item['tokens']
    failed = False
    
    new_tokens = tokens

    def change_idx(pos,add_pos,add_len):
        if pos <= add_pos:
            return pos
        else:
            return pos + add_len
    
    while True:
        for idx,t in enumerate(new_tokens):
            if len(t.split()) == 1:
                continue
            else:
                assert len(t.split()) > 1
                split_tokens = t.split()
                add_len = len(split_tokens) - 1
                new_new_tokens = new_tokens[:idx] + split_tokens + new_tokens[idx+1:]
                
                for ent in data_item['entity_mentions']:
                    ent['start'] = change_idx(ent['start'],idx,add_len)
                    ent['end'] = change_idx(ent['end'],idx,add_len)

                for eve in data_item['event_mentions']:
                    eve['trigger']['start'] = change_idx(eve['trigger']['start'],idx,add_len)
                    eve['trigger']['end'] = change_idx(eve['trigger']['end'],idx,add_len)

                    assert eve['trigger']['end'] > eve['trigger']['start'],eve['trigger']
                    acceptable_triggers = []
                    acceptable_triggers.append(" ".join(new_new_tokens[eve['trigger']['start']:eve['trigger']['end']]))
                    acceptable_triggers.append("".join(new_new_tokens[eve['trigger']['start']:eve['trigger']['end']]))
                    acceptable_triggers += [at.lower() for at in acceptable_triggers]
                    if  eve['trigger']['text'] not in acceptable_triggers:
                        failed = True

                    for arg in eve['arguments']:
                        orig_arg = ' '.join(new_tokens[arg['start']:arg['end']])

                        arg['start'] = change_idx(arg['start'],idx,add_len)
                        arg['end'] = change_idx(arg['end'],idx,add_len)

                        new_arg = ' '.join(new_new_tokens[arg['start']:arg['end']])

                        if new_arg != orig_arg:
                            failed = True
                        if arg['end'] <= arg['start']:
                            failed = True
                new_tokens = new_new_tokens

                break
                
        if all([len(t.split())==1 for t in new_tokens]):
            break
                
    data_item['tokens'] = new_tokens

    return data_item, failed


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--src_files", default=[], type=str, nargs='+', required=False,)
    parser.add_argument("--target_file", default='', type=str, required=False,)

    parser.add_argument("--p_nps", default=0.75, type=float, required=False,)
    parser.add_argument("--p_args", default=0.75, type=float, required=False,)

    parser.add_argument("--aug_num_per_sentence", default=10, type=int, required=False,)
    
    # KB
    parser.add_argument("--topk_concept", default=3, type=int, required=False,)
    parser.add_argument("--topk_entity", default=15, type=int, required=False,)
    parser.add_argument("--min_fre", default=20, type=int, required=False,)

    # PLM
    parser.add_argument("--topk_plm", default=10, type=int, required=False,)

    parser.add_argument("--use_KB", default=0, type=int, required=False,)
    parser.add_argument("--use_PLM", default=0, type=int, required=False,)
    parser.add_argument("--KB_PLM_ratio", default=0.5, type=float, required=False,)

    parser.add_argument("--show", default=False, action="store_true", required=False,)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    return args



args = parse_args()
src_files = args.src_files
target_file = args.target_file

# each noun phrase (no overlap with triggers or arguments) is replaced with a probability of p_nps
p_nps = args.p_nps 
# each argument is replaced with a probability of p_args
p_args= args.p_args 
show = args.show

aug_num_per_sentence = args.aug_num_per_sentence

topk_concept = args.topk_concept
topk_entity = args.topk_entity
min_fre = args.min_fre

topk_plm = args.topk_plm

use_KB = bool(args.use_KB)
use_PLM = bool(args.use_PLM)
KB_PLM_ratio = args.KB_PLM_ratio

assert len(src_files) > 0 and target_file != ''

print('saving args')
with open(target_file.replace('.json','.args.json'), 'w') as f:
    json.dump(vars(args),f,indent=4)

# os.environ["CUDA_VISIBLE_DEVICES"]="3"

print('loading model')
filling_model_name = 'bert-large-cased'
print(filling_model_name)
filling_tokenizer, filling_model = load_filling_model(filling_model_name)
filling_model = filling_model.eval()

device = 'cuda'
filling_model = filling_model.to(device)


print('loading probase')
parobase_path = '<path to /probase/data-concept-instance-relations.txt>'
entity2concept, concept2entity = load_probase(parobase_path)


print('loading data')
all_datas = []
for file in src_files:
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            all_datas.append(json.loads(line))
print(len(all_datas))


new_datas = []
for data_item in tqdm(all_datas):
    data_nps = get_noun_phrases(spacy_nlp, data_item['tokens'])
    
    for ii in range(aug_num_per_sentence):
        show = True

        # collect the positions of triggers and arguments
        tri_positions = []
        arg_positions = []
        for eve in data_item['event_mentions']:
            tri_positions.append([eve['trigger']['start'],eve['trigger']['end']])
            for arg in eve['arguments']:
                arg_positions.append([arg['start'],arg['end']])

        tri_set = []
        for kp in tri_positions:
            tri_set += list(range(kp[0],kp[1]))
        tri_set = set(tri_set)
        
        tri_arg_set = []
        for kp in tri_positions+arg_positions:
            tri_arg_set += list(range(kp[0],kp[1]))
        tri_arg_set = set(tri_arg_set)

        # sample the nps to be replaced
        selected_nps = []
        copy_nps = copy.deepcopy(data_nps)
        for np in copy_nps:
            start,end = np['position']
            if len(set(tri_arg_set).intersection(set(range(start,end)))) == 0:
                if random.uniform(0,1) < p_nps:
                    selected_nps.append(np)

        # sample the arguments to be replaced
        selected_args = []
        for arg_pos in arg_positions:
            start,end = arg_pos
            if len(set(tri_set).intersection(set(range(start,end)))) == 0: 
                if random.uniform(0,1) < p_args:  
                    
                    arg_text = ' '.join(data_item['tokens'][start:end])
                    arg_head_text, arg_head_lemma = parse_phrase(spacy_nlp, arg_text)
                    selected_args.append({
                            'position': [start,end],
                            'text': arg_text,
                            'head_text': arg_head_text,
                            'head_lemma': arg_head_lemma,
                        })
    
        selected_nps = selected_nps + selected_args
        
        replace_KB = [] 
        replace_PLM = []
        if use_KB and use_PLM:
            for nps in selected_nps:
                if random.uniform(0,1) < KB_PLM_ratio:
                    replace_KB.append(nps)
                else:
                    replace_PLM.append(nps)
            
        elif use_KB:
            replace_KB = selected_nps
            
        elif use_PLM:
            replace_PLM = selected_nps
            
        else:
            raise NotImplementedError
        
        # start replcing
        new_tokens = copy.deepcopy(data_item['tokens'])
        print('==\t\t', ' '.join(new_tokens)) if show else None
        print('==\t\t', ' | '.join(new_tokens)) if show else None
        
        # KB
        new_tokens, _ = replace_sentences_PB(new_tokens,
                                            replace_nps=replace_KB,
                                            trace_positions = tri_positions,
                                            topk_concept = topk_concept, 
                                            topk_entity = topk_entity, 
                                            min_fre = min_fre) 

        
        replace_KB_pos = [[np['position'][0],np['position'][1]] for np in replace_KB]
        print('====KB\t\t', ' | '.join(new_tokens),replace_KB_pos) if show else None
        
        # PLM
        replace_PLM_pos = [[np['position'][0],np['position'][1]] for np in replace_PLM]
    
        new_tokens, _ = replace_sentences_LM_pos(new_tokens,
                                                replace_position=replace_PLM_pos,
                                                trace_positions = tri_positions,
                                                topk=topk_plm)
        print('======PLM\t\t', ' | '.join(new_tokens), replace_PLM_pos) if show else None
        
        new_tokens[0] = new_tokens[0].capitalize()

        new_item = copy.deepcopy(data_item)
        new_item['tokens'] = new_tokens

        new_datas.append(new_item)
        print('=========\t\t', ' '.join(new_tokens)) if show else None
        print()  if show else None

with open(target_file, 'w', encoding='utf-8') as f:
    for new_item in new_datas:
        f.writelines(json.dumps(new_item)+'\n')

print('post processing')
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(filling_model_name)

print('post processing: delect empty token')
with open(target_file,'r',encoding='utf-8') as f:
    datas = [json.loads(line) for line in f.readlines()]
    
new_datas = []
for data_item in datas:
    new_data_item,failed = delect_empty(data_item)
    if not failed:
        new_datas.append(new_data_item)
    else:
        print('delect_empty failed')

with open(target_file, 'w', encoding='utf-8') as f:
    for new_item in new_datas:
        f.writelines(json.dumps(new_item)+'\n')

print('post processing: split multi token')
with open(target_file,'r',encoding='utf-8') as f:
    datas = [json.loads(line) for line in f.readlines()]
    
new_datas = []
for data_item in datas:
    new_data_item,failed = split_multi_token(data_item)
    if not failed:
        new_datas.append(new_data_item)
    else:
        print('split_multi_token failed')

with open(target_file, 'w', encoding='utf-8') as f:
    for new_item in new_datas:
        f.writelines(json.dumps(new_item)+'\n')

print('finish',target_file)
