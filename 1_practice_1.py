import random
import math
import random
import pandas as pd
import numpy as np
import re
from collections import Counter
import jieba
from functools import reduce
from operator import add, mul
import matplotlib.pyplot as plt

simple_grammar = """
sentence => noun_phrase verb_phrase
noun_phrase => Article Adj* noun
Adj* => null | Adj Adj*
verb_phrase => verb noun_phrase
Article => 一个 | 这个
noun => 女人 | 篮球 | 桌子 | 小猫
verb => 看着 | 坐在 | 听见 | 看见
Adj => 蓝色的 | 好看的 | 小小的
"""

def create_grammer(grammar_str, split='=>', line_split='\n'):
    grammar = {}
    for line in grammar_str.split(line_split):
        if not line.strip():
            continue
        exp, stmt = line.split(split)
        grammar[exp.strip()] = [s.split() for s in stmt.split('|')]
    return grammar
# strip 去除空格的效果
# split() 单个元素为加单个list

example_grammar = create_grammer(simple_grammar)

choice = random.choice
# 我的理解：这段经过2段式语法结构。从语法开始随机算内容，直到非键中文才完结。
# 再回返的时候，可以看到，先到target不是键值，然后才是从空格开始的累加。理不清的代码。不去耗时间
def generate(gram, target):
    if target not in gram:
        return target
    expaned = [generate(gram, t) for t in choice(gram[target])]
    return ''.join([e if e != '/n' else '\n' for e in expaned if e != 'null'])

generate(gram=example_grammar, target='sentence')

human = """
human = 自己 寻找 活动
自己 = 我 | 俺 | 我们 
寻找 = 找找 | 想找点 
活动 = 乐子 | 玩的
"""
# 一个“接待员”的语言可以定义为

host = """
host = 寒暄 报数 询问 业务相关 结尾 
寒暄 = 称谓 打招呼 | 打招呼
报数 = 我是 数字 号，
询问 = 请问你要 | 您需要
业务相关 = 玩玩 具体业务
数字 = 单个数字 | 数字 单个数字 
单个数字 = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 
称谓 = 人称 ,
人称 = 先生 | 女士 | 小朋友
打招呼 = 你好 | 您好 
玩玩 = null
具体业务 = 喝酒 | 打牌 | 打猎 | 赌博
结尾 = 吗？
"""
# 注意符号的处理，放在词汇的后方
for i in range(2):
    print(generate(gram=create_grammer(host, split='='), target='host'))


programming = """
stmt => if_exp | while_exp | assignment 
assignment => var = var
if_exp => if ( var ) { /n .... stmt }
while_exp=> while ( var ) { /n .... stmt }
var => chars number
chars => char | char char
char => student | name | info  | database | course
number => 1 | 2 | 3
"""
print(generate(gram=create_grammer(programming, split='=>'), target='stmt'))

random.choice(range(100))
filename = './2019-summer-学习资料/'
content = pd.read_csv(filename + "sqlResult_1558435.csv", encoding='gb18030')

content.head()

articles = content['content'].tolist()
len(articles)

def token(string):
    return re.findall('\w+', string)

# \w 匹配字母、数字、下划线。等价于'[A-Za-z0-9_]'。
# + 匹配前面的子表达式一次或多次。例如，'zo+' 能匹配 "zo" 以及 "zoo"，但不能匹配 "z"。+ 等价于 {1,}。
with_jieba_cut = Counter(jieba.cut(articles[110]))
with_jieba_cut.most_common()[:10]


# 可以理解for的前面是单步的取出值
print(''.join(token(articles[110])))

articles_clean = [''.join(token(str(a))) for a in articles]

len(articles_clean)
with open(filename + 'article_9k.txt', 'w') as f:
    for a in articles_clean:
        f.write(a + '\n')

# !ls
# !help

def cut(string):
    return list(jieba.cut(string))

print(len(articles_clean))

# 89611

TOKEN = []
# 文件打开后，关闭close()?
for i, line in enumerate(articles_clean):
    if i % 1000 == 0:
        print(i)
    if i > 86000:
        break
    try:
        TOKEN += cut(line)
    except:
        print(i, "行有问题")
        continue

from functools import reduce
from operator import add, mul

# reduce(function, iterable[, initializer])
# function -- 函数，有两个参数
# iterable -- 可迭代对象
# initializer -- 可选，初始参数
# def add(x, y) :
#     return x + y
reduce(add, [1, 2, 3, 4, 5, 8])
reduce(lambda x, y: x+y, [1, 2, 3, 4, 5, 8])


from collections import Counter
words_count = Counter(TOKEN)
# most_common https://docs.python.org/2/library/collections.html
words_count.most_common(100)
frequiences = [f for w,f in words_count.most_common(100)]
x = list(range(100))

# %matplotlib inline
plt.plot(x, frequiences)
plt.plot(x, np.log(frequiences))

def prob_1(word):
    return words_count[word] / len(TOKEN)
prob_1('我们')


# TOKEN 是从原文切好的词汇
print(TOKEN[:10])
TOKEN = [str(t) for t in TOKEN]
# 合并数据中相连的2个词汇
TOKEN_2_GRAM = [''.join(TOKEN[i:i+2]) for i in range(len(TOKEN[:-2]))]

print(TOKEN_2_GRAM[:10])
words_count_2 = Counter(TOKEN_2_GRAM)



# 计算比较牵强。是初级来理解的内容。后续词汇中会有任意情况，相连是无法准确描述的
def prob_2(word1, word2):
    if word1 + word2 in words_count_2:
        return words_count_2[word1 + word2] / len(TOKEN_2_GRAM)
    else:
        return 1 / len(TOKEN_2_GRAM)

print(prob_2('我们', '在'))
print(prob_2('在', '吃饭'))

# ╮(￣▽￣"")╭稀有句式好接近
def get_probablity(sentence):
    words = cut(sentence)

    sentence_pro = 1
    for i ,word in enumerate(words[:-1]):
        next_ = words[i+1]
        probablity = prob_2(word, next_)
        sentence_pro *= probablity
    # print(sentence_pro)
    return sentence_pro


get_probablity('小明今天抽奖抽到一台苹果手机')
get_probablity('小明今天抽奖抽到一架波音飞机')
get_probablity('洋葱奶昔来一杯')
get_probablity('养乐多绿来一杯')


# 验证之前的语法生成
for sen in [generate(gram=example_grammar, target='sentence') for i in range(10)]:
    print('sentence:{} \n with Prb: {}'.format(sen, get_probablity(sen)))

# 拓展到长段的文章
need_compared = [
    "今天晚上请你吃大餐，我们一起吃日料 明天晚上请你吃大餐，我们一起吃苹果",
    "真事一只好看的小猫 真是一只好看的小猫",
    "今晚我去吃火锅 今晚火锅去吃我",
    "洋葱奶昔来一杯 养乐多绿来一杯"
]
for s in need_compared:
    s1, s2 = s.split()
    p1, p2 = get_probablity(s1), get_probablity(s2)

    better = s1 if p1>p2 else s2

    print('{} is more possible'.format(better))
    print('-'*4 + '{} with probility_1 {}'.format(s1, p1))
    print('_'*4 + '{} with probility_2 {}'.format(s2, p2))

