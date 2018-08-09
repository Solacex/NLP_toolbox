import re
import array
import os
import numpy as np
import sys
import pickle
file_dir='/projects/D2DCRC/lg/guangrui/word2vec/flickr/vec500flickr30m'

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    return string.strip().lower().split()

class BigFile:

    def __init__(self, datadir):
        self.nr_of_images, self.ndims = map(int, open(os.path.join(datadir,'shape.txt')).readline().split())
        id_file = os.path.join(datadir, "id.txt")
        self.names = open(id_file).read().strip().split()
        print(len(self.names),self.nr_of_images)
        assert(len(self.names) == self.nr_of_images)
        self.name2index = dict(zip(self.names, range(self.nr_of_images)))
        self.binary_file = os.path.join(datadir, "feature.bin")
        print ("[%s] %dx%d instances loaded from %s" % (self.__class__.__name__, self.nr_of_images, self.ndims, datadir))


    def read(self, requested, isname=True):
        requested = set(requested)
        if isname:
            index_name_array = [(self.name2index[x], x) for x in requested if x in self.name2index]
        else:
            assert(min(requested)>=0)
            assert(max(requested)<len(self.names))
            index_name_array = [(x, self.names[x]) for x in requested]
        if len(index_name_array) == 0:
            return [], []

        index_name_array.sort(key=lambda v:v[0])
        sorted_index = [x[0] for x in index_name_array]

        nr_of_images = len(index_name_array)
        vecs = [None] * nr_of_images
        offset = np.float32(1).nbytes * self.ndims

        res = array.array('f')
        fr = open(self.binary_file, 'rb')
        fr.seek(index_name_array[0][0] * offset)
        res.fromfile(fr, self.ndims)
        previous = index_name_array[0][0]

        for next in sorted_index[1:]:
            move = (next-1-previous) * offset
            #print next, move
            fr.seek(move, 1)
            res.fromfile(fr, self.ndims)
            previous = next

        fr.close()

        return [x[1] for x in index_name_array], [ res[i*self.ndims:(i+1)*self.ndims].tolist() for i in range(nr_of_images) ]


    def read_one(self, name):
        renamed, vectors = self.read([name])
        return vectors[0]

    def shape(self):
        return [self.nr_of_images, self.ndims]


class Text2Vec:

    def __init__(self, datafile, ndims = 0, L1_normalize = 0, L2_normalize = 0):
#        printStatus(INFO + '.' + self.__class__.__name__, 'initializing ...')
        self.datafile = datafile
        self.ndims = ndims
        self.L1_normalize = L1_normalize
        self.L2_normalize = L2_normalize


        assert type(L1_normalize) == int
        assert type(L2_normalize) == int
        assert (L1_normalize + L2_normalize) <= 1

    def embedding(self, query):
        vec = self.mapping(query)
        if vec is not None:
            vec = np.array(vec)
        return vec

    def do_L1_norm(self, vec):
        L1_norm = np.linalg.norm(vec, 1)
        return 1.0 * np.array(vec) / L1_norm

    def do_L2_norm(self, vec):
        L2_norm = np.linalg.norm(vec, 2)
        return 1.0 * np.array(vec) / L2_norm

class AveWord2Vec(Text2Vec):

    # datafile: the path of pre-trained word2vec data
    def __init__(self, datafile, ndims = 0, L1_normalize = 0, L2_normalize = 0):
        Text2Vec.__init__(self, datafile, ndims, L1_normalize, L2_normalize)
        self.word2vec = BigFile(datafile)
        if ndims != 0:
            assert self.word2vec.ndims == self.ndims, "feat dimension is not match %d != %d" % (self.word2vec.ndims, self.ndims)
        else:
            self.ndims = self.word2vec.ndims


    def preprocess(self, query, clear):
        if clear:
            words = clean_str(query)
        else:
            words = query.strip().split()
        return words

    def mapping(self, query, clear = True):
        words = self.preprocess(query, clear)

        #print query, '->', words
        renamed, vectors = self.word2vec.read(words)
        renamed2vec = dict(zip(renamed, vectors))

        if len(renamed) != len(words):
            vectors = []
        #    dic={}
            for word in words:
                if word in renamed2vec:
                    vectors.append(renamed2vec[word])
             #       dic[word]=renamed2vec[word]

        if len(vectors)>0:
            vec = np.array(vectors).mean(axis=0)
            vec=np.array(vectors)
            if self.L1_normalize:
                return self.do_L1_norm(vec)
            if self.L2_normalize:
                return self.do_L2_norm(vec)
            return vec#,dic
        else:
            return None

w2v_encoder=AveWord2Vec(file_dir)
with open('../data/tv17_captions_test.pkl','rb') as f:
    ori = pickle.load(f)
result={}
for k  in ori.keys():
    tmp_list=ori[k]
    for sen in tmp_list:
        if k in result:
            result[k].append(w2v_encoder.mapping(sen))
        else:
            result[k]=[w2v_encoder.mapping(sen)]
    np.save('../data/w2v/'+k+'.npy',result[k])
#with open('msrvtt_word2vec.pkl','wb') as f:
#    pickle.dump(result,f)
