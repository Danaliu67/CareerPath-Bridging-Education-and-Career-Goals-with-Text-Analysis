import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

from gensim.models import Word2Vec
from utils import find_max_index
from utils import skill_classifier

# from sklearn.metrics import calinski_harabasz_score

def map_keywords_to_vectors(keyword_list, model):
    keyword_vectors = []
    for keyword in keyword_list:
        if keyword in model.wv.key_to_index:
            keyword_vectors.append(model.wv[keyword])
    return keyword_vectors

def calc_keyword_similarity(keyword_list_1, keyword_list_2, model):
    """
    计算两个关键词列表的余弦相似度
    :param keyword_list_1: 第一个关键词列表
    :param keyword_list_2: 第二个关键词列表
    :param model: 训练好的词向量模型
    :return: 两个关键词列表的余弦相似度
    """
    print("cal")
    print(keyword_list_1)
    print(keyword_list_2)
  
    # 将关键词列表映射到词向量空间
    keyword_vectors_1 = map_keywords_to_vectors(keyword_list_1, model)
    keyword_vectors_2 = map_keywords_to_vectors(keyword_list_2, model)
    # print(keyword_vectors_1)
    # print(keyword_vectors_2)
    len_ = min(len(keyword_vectors_1), len(keyword_vectors_2))
    print(len_)
    if(len_==0):
        return 0
    else:
        if(len(keyword_vectors_1) > len_):
            keyword_vectors_1 = keyword_vectors_1[:len_-len(keyword_vectors_1)]
        if(len(keyword_vectors_2) > len_):
            keyword_vectors_2 = keyword_vectors_2[:len_-len(keyword_vectors_2)]


        keyword_vectors_1 = np.array(keyword_vectors_1)
        keyword_vectors_2 = np.array(keyword_vectors_2)

        print(len(keyword_vectors_1))
        print(len(keyword_vectors_2))

        # 计算两个关键词列表的余弦相似度
        sim = np.dot(keyword_vectors_1, keyword_vectors_2.T) / \
            (np.linalg.norm(keyword_vectors_1, axis=1) * np.linalg.norm(keyword_vectors_2, axis=1))
        
        # 取平均值作为相似度
        sim_mean = np.mean(sim)
        
        return sim_mean


with open('../../skill_classifier/all_soft_skills.txt', 'r') as f:
    lines = [line.strip() for line in f.readlines()]
soft_skills = []
for line in lines:
    words = word_tokenize(line.lower())
    soft_skills.extend(words)

# read job
file_path = '../../data_key/keywords job(ver2).xls'   
raw_data = pd.read_excel(file_path, header=0)  
job_data = raw_data.values

# read program
program_file = '../../data_key/keywords_program(2).xlsx'   
program_raw_data = pd.read_excel(program_file, header=0)  
program_data = program_raw_data.values
program_row = program_data[0][1:]
print(len(program_data))

# to get job and program name
# read job
df1 = pd.read_csv("../../data_pre/job data_pre(ver2).csv")
job_title = df1['job title']

# read program
df2 = pd.read_csv("../../data_pre/program data_pre(ver2).csv")
school_name = df2['Schoole']
program_name = df2['program']

df3 = pd.read_csv("../../eval/clustered_data.csv") 
cluster_data = df3["cluster"].values

# 加载训练好的Word2Vec模型
model = Word2Vec.load("../checkpoints/word2vec_model.bin")

rmd_result = []
cosine_score = 0
jaccard_score = 0
euclidean_score = 0
best_score = 0
worst_score = 0

labels = []
X = []
for test_id in range(0,len(job_data)):
    # print(job_data[test_id])
    row = job_data[test_id][1:]
    # newlist = [x if pd.isnull(x) == False else '<pad>' for x in row]
    newlist = [x for x in row if pd.isnull(x) == False]
    newlist_soft, newlist_hard = skill_classifier(newlist, soft_skills)
    print("new list skills:", newlist_soft, newlist_hard)

    cosine_list = []
    jaccard_list = []
    euclidean_list = []
    for program in program_data:
        
        program = program[1:]
      
        program = [x for x in program if pd.isnull(x) == False]
        
        program_soft, program_hard = skill_classifier(program, soft_skills)
        soft_simi = calc_keyword_similarity(newlist_soft, program_soft, model)
        hard_simi = calc_keyword_similarity(newlist_hard, program_hard, model)
        cosine_list.append(0.3*soft_simi + 0.7*hard_simi)
        

    cosine_max_number, cosine_max_index = find_max_index(cosine_list, 3)
    
    # cosine_max_one = np.argmax(cosine_list)
    # label = program_data[cosine_max_one]
    # labels.append(cosine_max_one)

    recom = [job_title[test_id]]
    print(job_title[test_id])
    print("matched programs:",end=' ')
    for item in cosine_max_index:
        print("university name:",school_name[item],end=' ')
        print("program name:", program_name[item])
        recom.append(school_name[item] + '/' + program_name[item])
    
    cosine_score += 3-len(set(cluster_data[cosine_max_index]))
    best_score += 2

    rmd_result.append(recom)

print(cosine_score)

print(0, best_score)
print(cosine_score/best_score)

# import csv
# header = ['job title', 'recommand1', 'recommand2', 'recommand3']

# with open('rmd_program_word2vec_cosine_skill.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(header)
#     writer.writerows(rmd_result)



