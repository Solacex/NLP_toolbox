import scipy.io
#from allennlp.modules.elmo import Elmo,batch_to_ids
import json
import torch
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
import torch
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
from allennlp.commands import elmo
with open('./tv16_test.json','r') as f:
    loaded=json.loads(f.read())
sen=[]
sen_ids=[]
video_ids=[]
vectors=[]
i=0
keys=[]
for se in loaded['sentences']:
    sen.append(se['caption'].split(' '))
    keys.append(str(se['sen_id']).zfill(6)+str(se['video_id']))
    video_ids.append(int(se['video_id']))
    sen_ids.append(se['sen_id'])
    i+=1
    if i%500==0:
        #break
        print(i)
emb_list=[]
elmo = elmo.ElmoEmbedder(options_file, weight_file, 1)
llen=int(len(sen)/500)
length=len(sen)
print(length)
for e in range(llen):
    start=e*500
    end=start+500
    if end>length:
        end=length
    print('starting:',start,end)
    sen_list=sen[start:end]
    #character_ids = batch_to_ids(sen_list)
    embeddings = elmo.embed_sentences(sen_list)
    for emb in embeddings:
        emb_list.append(emb.mean(0))
result=dict(zip(keys,emb_list))
print(emb_list[0].shape)
import h5py
import numpy as np
wt=h5py.File('avg_elmo_tv16_test.hdf5','w')
wt['sen_id']=sen_ids
print(type(emb_list),type(emb_list[0]))
wt['video_id']=video_ids
grp=wt.create_group('elmo')
for ct in range(length):
#emb=np.asarray(emb_list)
    grp.create_dataset(str(sen_ids[ct]),data=emb_list[ct])
    if ct%500==0:
        print(ct)
#print(emb.shape,emb.dtype)
##wt.create_dataset('elmo',data=emb)
#emb=emb.astype(np.float64)
#wt['elmo']=emb
wt.close()
'''
import numpy as np
result={'sen_id':sen_ids,'video_id':video_ids,'sen_vec':np.asarray(emb_list),'caption':sen}
scipy.io.savemat('elmo_train.mat',result,format='4')
mid=map(list,zip(sen_ids,video_ids,emb_list,sen))
final_list=[]
for item in mid:
    new_dict=dict(zip(['sen_id','video_id','sen_vec','caption'],item))
    final_list.append(new_dict)

jre=json.dumps(final_list)
with open('ttestjson.json','w',encoding='utf8') as q:
    q.write(jre)
'''
