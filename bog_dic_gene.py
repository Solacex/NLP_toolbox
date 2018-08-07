import pickle

with open('./data/msrvtt_captions.pkl','rb') as f:
    w=pickle.load(f)
re={}
count=0
for k in w.keys():
    sen_list=w[k]
    count+=1
    sen_s=[]
    for s in sen_list:
        tmp = s.split()
        for kk in emp:
            re[kk] = re.get(kk,0) +1

    if count%5==0:
        print(count)

print(len(re))
with open('./bog_dic.pkl','wb') as m:
    pickle.dump(re)
m.close()
