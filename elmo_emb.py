
import math
import multiprocessing as mtp
import time
import logging
import shutil
import os.path
import numpy as np
import pickle
from allennlp.data.dataset import Batch
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper, ELMoTokenCharactersIndexer
from allennlp.data import Token,Vocabulary,Instance
from allennlp.data.fields import TextField
import torch
options_file = "../data/elmo_option.json"
weight_file  = '../data/elmo_weights.hdf5'

def elmo(ll):
    for k in ll:
        sen_list=w[k]
        count+=1
        sen_s=[]
        for s in sen_list:
            sen_s.append(s.split())
        elmo=Elmo(options_filw,weight_file,1) 
        instances=[]
        indexer=ELMoTokenCharactersIndexer()
        for sen in sen_s:
            tokens=[Token(token) for token in sen]
            field = TextField(tokens,{'character_ids': indexer})
            instance = Instance({'elmo':field})
            instances.append(instance)
        dataset = Batch(instances)
        voca=Vocabulary()
        dataset.index_instances(voca)

        dic={'elmo':{'num_tokens':15}}
        character_ids=dataset.as_tensor_dict(dic)['elmo']['character_ids']
        character_ids=character_ids
        sth = elmo(character_ids)['elmo_representations']
        sth = list(torch.chunk(result,result.shape[0],0))
        re[k] = sth


def multiCombine(dicc_list):

   
    tasks = []
    total = len(dicc_list)
    proc_count = mtp.cpu_count()-10
    block = int(math.ceil(total*1.0/proc_count))
    box = []
    print( ' - Process:',proc_count)
    print(' - Block size:',block)
    for i in range(proc_count):
        left = int(i*block)
        if (i+1)*block > total:
            right = int(total)
        else:
            right = int((i+1)*block)
        box.append(dicc_list[left:right])
    tgtFunc = elmo
    count = mtp.Value('i', 0)
    lock = mtp.Lock()

    print('elmo generation start')
    duration = time.time()
    for i in range(proc_count):
        proc = mtp.Process(target=tgtFunc, args=(box[i]))
        proc.start()
        tasks.append(proc)

    for proc in tasks:
        proc.join()

    duration = time.time() - duration
    print('Combining duration:', int(duration), 's')

with open('../data/msrvtt_captions.pkl','rb') as f:
    w=pickle.load(f)
re={}
kk = [k for k in w.keys()]
multiCombine(kk)
with open('msrvtt_elmo_captions_MUl.pkl','wb') as m:
    pickle.dump(re)
'''
count=0
for k in w.keys():
    sen_list=w[k]
    count+=1
    sen_s=[]
    for s in sen_list:
        sen_s.append(s.split())
    if count%5==0:
        print(count)
    elmo_pre=elmo(sen_s)
    result = elmo_pre[0].data
    result = list(torch.chunk(result,result.shape[0],0))
    re[k] = result

with open('./data/elmo_msrvtt_captions.pkl','wb') as m:
    pickle.dump(re)
m.close()
'''
