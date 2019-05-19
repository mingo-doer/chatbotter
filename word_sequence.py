import numpy as np

class WordSequence(object):
    # 填充标签
    PAD_TAG = '<pad>'
    # 未知标签
    UNK_TAG = '<unk>'
    # 开始标签
    START_TAG = '<s>'
    # 结束标签
    END_TAG = '</s>'

    PAD = 0
    UNK = 1
    START = 2
    END = 3

    def __init__(self):
        #初始化基本的字典dict
        self.dict = {
            WordSequence.PAD_TAG: WordSequence.PAD,
            WordSequence.UNK_TAG: WordSequence.UNK,
            WordSequence.START_TAG: WordSequence.START,
            WordSequence.END_TAG: WordSequence.END,
        }
        # 是否训练
        self.fited = False
    # 字转换为index
    def to_index(self, word):
        assert self.fited, "WordSequence 尚未进行 fit 操作"
        if word in self.dict:
            return self.dict[word]
        return WordSequence.UNK
    # index转换为字
    def to_word(self, index):
        assert self.fited, "WordSequence 尚未进行 fit 操作"
        # 从键值对里取数据，从key ，value中取
        for k, v in self.dict.items():
            if v == index:
                return k
        return WordSequence.UNK_TAG

    def size(self):

        assert self.fited, "WordSequence 尚未进行 fit 操作"
        return len(self.dict) + 1

    def __len__(self):
        return self.size()

    #拟合 （self，句子，最小出现次数，最大出现次数，最大的特征数）
    # 训练字典
    def fit(self, sentences, min_count=5, max_count=None, max_features=None):
        # 已经被训练了就提示
        assert not self.fited, 'WordSequence 只能fit一次'
        # 进行统计
        count = {}
        for sentence in sentences:
            arr = list(sentence)
            for a in arr:
                # 如果没有被统计
                if a not in count:
                    count[a] = 0
                count[a] += 1
        # 统计比最小的大的，最大的小的
        if min_count is not None:
            count = {k: v for k, v in count.items() if v >= min_count}
        if max_count is not None:
            count = {k: v for k, v in count.items() if v <= max_count}
        self.dict = {
            WordSequence.PAD_TAG: WordSequence.PAD,
            WordSequence.UNK_TAG: WordSequence.UNK,
            WordSequence.START_TAG: WordSequence.START,
            WordSequence.END_TAG: WordSequence.END,
        }
        # 用来判断变量的变量类型
        if isinstance(max_features, int):

            count = sorted(list(count.items()), key=lambda x:x[1])
            if max_features is not None and len(count) > max_features:
                count = count[-int(max_features):]
            for w, _ in count:
                self.dict[w] = len(self.dict)
        else:

            for w in sorted(count.keys()):
                self.dict[w] = len(self.dict)
        self.fited = True

    # 句子和向量的转换
    def transform(self, sentence, max_len=None):
        assert self.fited, "WordSequence 尚未进行 fit 操作"

        if max_len is not None:
            r = [self.PAD] * max_len
        else:
            r = [self.PAD] * len(sentence)
        # 枚举类型 举句子
        for index, a in enumerate(sentence):
            if max_len is not None and index >= len(r):
                break
            r[index] = self.to_index(a)

        return np.array(r)
    # 向量转句子
    def inverse_transform(self, indices,
                          ignore_pad=False, ignore_unk=False,
                          ignore_start=False, igonre_end=False):
        ret = []
        for i in indices:
            word = self.to_word(i)
            if word == WordSequence.PAD_TAG and ignore_pad:
                continue
            if word == WordSequence.UNK_TAG and ignore_unk:
                continue
            if word == WordSequence.START_TAG and ignore_start:
                continue
            if word == WordSequence.END_TAG and igonre_end:
                continue
            ret.append(word)

        return ret

def test():

    ws = WordSequence()
    ws.fit([
        ['你', '好', '啊'],
        ['你', '好', '哦'],
    ])

    indice = ws.transform(['我', '们', '好','吗'])
    print(indice)

    back = ws.inverse_transform(indice)
    print(back)

if __name__ == '__main__':
    test()