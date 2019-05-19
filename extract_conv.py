import re
import sys
import pickle
from tqdm import tqdm
# 将句子与标点分隔开
def make_split(line):
    if re.match(r'.*([，…?!\.,!？])$', ''.join(line)):
        return  []
    return [', ']
# 判断是不是一个好句子
def good_line(line):
    if len(re.findall(r'[a-zA-Z0-9]', ''.join(line)))>2:
        return False
    return True
# 进行一些正则表达

def regular(sen):
    # 找. 最少出现三次，最多100词，当这样的条件出现时，替换为···，对应的是sen的这个句子
    sen = re.sub(r'\.{3,100}', '…', sen)
    sen = re.sub(r'…{1,100}', '', sen)
    sen = re.sub(r'[,]{1,100}', '，', sen)
    sen = re.sub(r'[\.]{1,100}', '。', sen)
    sen = re.sub(r'[\?]{1,100}', '？', sen)
    sen = re.sub(r'[!]{1,100}', '！', sen)
    # 小黄鸭部分
    sen=re.sub(r'[=]{1,100}','',sen)
    sen = re.sub(r'[。]{1,100}', '。', sen)
    sen = re.sub(r'[□]{1,100}', '', sen)
    sen = re.sub(r'[~]{1,100}', '', sen)
    sen = re.sub(r'[(]{1,100}', '', sen)
    sen = re.sub(r'[)]{1,100}', '', sen)
    sen = re.sub(r'[V]{1,100}', '', sen)
    sen = re.sub(r'[_]{1,100}', '', sen)
    sen = re.sub(r'[π]{1,100}', '', sen)
    sen = re.sub(r'[O]{1,100}', '', sen)
    sen = re.sub(r'[\\^]{1,100}', '', sen)
    sen = re.sub(r'[>]{1,100}', '', sen)
    sen = re.sub(r'[<]{1,100}', '', sen)
    sen = re.sub(r'[*]{1,100}', '', sen)
    sen = re.sub(r'[～]{1,100}', '', sen)


    return sen

def main(limit=20, x_limit=3, y_limit=6):
    # 只需要输出小于limit 的句子
    from word_sequence import WordSequence
    print('extract lines')
    fp = open("xiaohuangji50w_fenciA.conv", 'r', errors='ignore', encoding='utf-8')
    groups = []
    group = []
    for line in tqdm(fp):
        # 以M为开始
        if line.startswith('M '):
            line = line.replace('\n', '/') #换成换行
            if '/' in line:
                line = line[2:].split('/')#以/为切分
            else:
                line = list(line[2:])#从第二个开始
            line = line[:-1]
            group.append(list(regular(''.join(line))))
        else:
            if group:
                groups.append(group)
                group = []
    if group:
        groups.append(group)
        group = []
    print('extract group')
    x_data = []#问
    y_data = []# 答
    for group in tqdm(groups):
        for i, line in enumerate(group):#枚举类[（0,句子1），（1，句子2）]
            last_line = None
            if i > 0:
                last_line = group[i - 1]
                if not good_line(last_line):
                    last_line = None
            next_line = None
            if i < len(group) - 1:
                next_line = group[i + 1]
                if not good_line(next_line):
                    next_line = None
            next_next_line = None
            if i < len(group) -2:
                next_next_line = group[i + 2]
                if not good_line(next_next_line):
                    next_next_line = None

            if next_line:
                x_data.append(line)
                y_data.append(next_line)
            if last_line and next_line:
                x_data.append(last_line + make_split(last_line) + line)
                y_data.append(next_line)
            if next_line and next_next_line:
                x_data.append(line)
                y_data.append(next_line + make_split(next_line) + next_next_line)

    print(len(x_data), len(y_data))
    # zip 将数据整合 整合成list 的一个列表
    for ask, answer in zip(x_data[:20], y_data[:20]):
        print(''.join(ask))
        print(''.join(answer))
        print('-'*20)

    data = list(zip(x_data, y_data))
    data = [
        (x, y)
        for x, y in data
        if len(x) < limit \
        and len(y) < limit \
        and len(y) >= y_limit \
        and len(x) >= x_limit
    ]
    x_data, y_data = zip(*data)
    print('fit word_sequence')
    ws_input = WordSequence()
    ws_input.fit(x_data + y_data)
    print('dump')
    pickle.dump(
        (x_data, y_data),
        open('chatbot.pkl', 'wb')
    )
    pickle.dump(ws_input, open('ws.pkl', 'wb'))

    print('done')

if __name__ == '__main__':
    main()