from sklearn.externals import joblib
import numpy as np
import pickle
with open('./tv17_bog_dic.pkl','rb') as m:
    bog = pickle.load(m)
with open('../data/tv17_captions_test.pkl','rb') as f:
    w = pickle.load(f)


keys=list(bog.keys())
keys=np.array(keys)
length_dic=len(keys)
count=0
repre={}


for k in w.keys():
    sen_list=w[k]
    count+=1
    len_cu =len(sen_list)
    tmp=np.zeros((len_cu, length_dic))
    i = 0
    for s in sen_list:
        tmp_word = s.split()
        for word in tmp_word:
            tmp[i,np.argwhere(keys==word)]+=1
    np.save('../data/bog/'+k+'.npy',tmp)
    if count%500==0:
        print(count)

print(len(repre))
#joblib.dump(repre,'bog_msrvtt_caption.pkl')
