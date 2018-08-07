import pickle
with open('./bog_dic.pkl','rb') as m:
    bog = pickle.load(m)
with open('./data/msrvtt_captions.pkl','rb') as f:
    w = pickle.load(f)


index_bog={}
cnt = 0
for k in bog.keys():
    index_bog[k]=cnt
    cnt+=1
count=0
repre={}
length_dic=len(bog)

for k in w.keys():
    sen_list=w[k]
    count+=1
    len_cu =len(sen_list)
    tmp=np.zeros((len_cu, length_dic))
    i = 0
    for s in sen_list:
        tmp_word = s.split()
            for word in tmp_word:
                tmp[i,index_bog[word]]+=1
    repre[k] = tmp
    if count%5==0:
        print(count)

print(len(repre))
with open('./bog_msrvtt_caption.pkl','wb') as m:
    pickle.dump(repre,m)
m.close()
