import pickle

with open('../data/msrvtt_captions.pkl','rb') as f:
    w=pickle.load(f)
re={}
count=0
for k in w.keys():
    sen_list=w[k]
    count+=1
    sen_s=[]
    for s in sen_list:
        s.replace(',','').replace('.','').replace('?','')
        tmp = s.split(' ')
        for kk in tmp:
            re[kk] = re.get(kk,0) +1

    if count%500==0:
        print(count)

print(len(re))
rmlist=[]
for k in re.keys():
    if re[k]<5:
        rmlist.append(k)
        print(k)
print(len(rmlist))
for rm in rmlist:
    re.pop(rm)
with open('./bog_dic.pkl','wb') as m:
    pickle.dump(re,m)
m.close()
