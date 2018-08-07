import pickle
from allennlp.data.dataset import Batch
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper, ELMoTokenCharactersIndexer
from allennlp.data import Token,Vocabulary,Instance
from allennlp.data.fields import TextField
import torch
options_file = "./elmo/models/elmo_option.json"
weight_file = './elmo/models/elmo_weights.hdf5'
def elmo(sentences):
    elmo=Elmo(options_file,weight_file,1,dropout=0)
    instances=[]
    elmo=elmo.cuda()
    indexer=ELMoTokenCharactersIndexer()
    for sen in sentences:
        tokens=[Token(token) for token in sen]
        field = TextField(tokens,{'character_ids': indexer})
        instance = Instance({'elmo':field})
        instances.append(instance)
    dataset = Batch(instances)
    voca=Vocabulary()
    dataset.index_instances(voca)

    dic={'elmo':{'num_tokens':15}}
    character_ids=dataset.as_tensor_dict(dic)['elmo']['character_ids']
    character_ids=character_ids.cuda()
    return elmo(character_ids)['elmo_representations']
with open('./data/msrvtt_captions.pkl','rb') as f:
    w=pickle.load(f)
re={}
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
