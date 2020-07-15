# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import jieba
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


# %%
warnings.filterwarnings('ignore')


# %%
def cut_words(file_path):
    #对文本进行分词，
    # param ：file_path ：txt文件路径
    # return： text_cut_result 返回带空格的切词结果
    text_cut_result = ''
    text = open(file_path, 'r', encoding='gb18030').read()
    text_cut = jieba.cut(text)
    for word in text_cut:
        text_cut_result += word +' '
    return text_cut_result


# %%
def loadfile(file_dir, label):
    #将路径下的所有文件加载
    # param：file_dir:
    file_list = os.listdir(file_dir)
    word_list = []
    label_list = []
    for file_n in file_list:
        file_path = file_dir + '/' + file_n
        word_list.append(cut_words(file_path))
        label_list.append(label)
    return word_list, label_list


# %%
#训练数据
train_words_list1, train_labels1 = loadfile('/Applications/py_workspace/data_mining/data/text_classification/train/女性', '女性')
train_words_list2, train_labels2 = loadfile('/Applications/py_workspace/data_mining/data/text_classification/train/体育', '体育')
train_words_list3, train_labels3 = loadfile('/Applications/py_workspace/data_mining/data/text_classification/train/文学', '文学')
train_words_list4, train_labels4 = loadfile('/Applications/py_workspace/data_mining/data/text_classification/train/校园', '校园')

train_word_list = train_words_list1 + train_words_list2 + train_words_list3 + train_words_list4
train_labels = train_labels1 +train_labels2 + train_labels3 + train_labels4

test_words_list1, test_labels1 = loadfile('/Applications/py_workspace/data_mining/data/text_classification/test/女性', '女性')
test_words_list2, test_labels2 = loadfile('/Applications/py_workspace/data_mining/data/text_classification/test/体育', '体育')
test_words_list3, test_labels3 = loadfile('/Applications/py_workspace/data_mining/data/text_classification/test/文学', '文学')
test_words_list4, test_labels4 = loadfile('/Applications/py_workspace/data_mining/data/text_classification/test/校园', '校园')

test_word_list = test_words_list1 + test_words_list2 + test_words_list3 + test_words_list4
test_labels = test_labels1 + test_labels2 + test_labels3 + test_labels4
#加载停用词，过滤掉的，我们，词频较高，但无实际意义的词
stop_words = open('/Applications/py_workspace/data_mining/data/text_classification/stop/stopword.txt', 'r', encoding='utf-8').read()
stop_words = stop_words.encode('utf-8').decode('utf-8-sig') #处理头部\ufeff，若列表头部无特殊字符，可不用，
stop_words = stop_words.split('\n')


# %%
#计算单词权重 tf-idf，一个单词在一个文档中出现的次数多，又在其他文档中出现的次数少
# max_df=0.5，一个单词在50%的文档中都出现过了，那么只携带了很少的信息，因此不作为分词统计

tf = TfidfVectorizer(stop_words= stop_words, max_df=0.5)

train_features = tf.fit_transform(train_word_list)
test_features = tf.transform(test_word_list)


# %%
#多项式贝叶斯分类器 特征变量是离散变量，符合多项分布，在文档分类中特征变量体现在一个单词出现的次数，或者单词的tf-idf值等
#高斯朴素贝叶斯：特征变量是连续变量，符合高斯分布，例如人的身高，物体的长度等
#伯努利朴素贝叶斯分类器：特征变量是布尔变量，符合0/1分布，在文档分类中特征是 单词是否出现
#alpha ：平滑系数
# 若一个单词在训练样本中没有出现，这个单词的概率为0，但训练集样本是整体的抽样情况，不能因为一个事件没有观察到，就认为整个事件的概率为0，
#=1 laplace平滑，当样本很大时，+1得到的概率忽略不计
#当 0<alpha<1 时，使用的是 Lidstone 平滑。对于 Lidstone 平滑来说，alpha 越小，迭 代次数越多，精度越高。我们可以设置 alpha 为 0.001。


clf = MultinomialNB(alpha=0.001).fit(train_features, train_labels)
predicted_labels= clf.predict(test_features)

accuracy_rate = metrics.accuracy_score(test_labels, predicted_labels)
print(accuracy_rate)


# %%


