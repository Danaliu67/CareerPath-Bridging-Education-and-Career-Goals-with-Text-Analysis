import numpy as np
import pandas as pd
#from gensim.models import Word2Vec
from key_extractor import key_extractor
from utils import find_max_index
from sklearn.metrics import calinski_harabasz_score
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
import torch

class Job_prediction():
    def __init__(self):
        
        #file_path = '../data_key/keywords job(ver2).xls'
        file_path = 'C:/Users/cream/Desktop/datamining/predictor2.0/predictor2.0/data_key/keywords job(ver2).xls'
        raw_data = pd.read_excel(file_path, header=0)  
        self.job_data = raw_data.values

        df1 = pd.read_csv("C:/Users/cream/Desktop/datamining/predictor2.0/predictor2.0/data_pre/job data_pre(ver2).csv")
        self.job_title = df1['job title']

        config=RobertaConfig.from_pretrained('roberta-base')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaModel.from_pretrained('roberta-base',config=config)

        self.job_vectors = []
        for i in range(len((self.job_data))):
            row = self.job_data[i][1:]
            newlist = [x for x in row if pd.isnull(x) == False]
            input_1=self.tokenizer(' '.join(newlist),return_tensors="pt")
            output_1=self.model(**input_1)
            keyword_vectors_1 = torch.mean(output_1.last_hidden_state, dim=1).detach().numpy()
            self.job_vectors.append(keyword_vectors_1)



    def map_keywords_to_vectors(self, keywords):
        input =self.tokenizer(' '.join(keywords),return_tensors="pt")
        output = self.model(**input)
        keywords_vector = torch.mean(output.last_hidden_state, dim=1).detach().numpy()
        return keywords_vector

    def calc_bert_keyword_similarity(self,keyword_vectors_1, keyword_vectors_2):
        # 计算两个关键词列表的余弦相似度
        sim = np.dot(keyword_vectors_1, keyword_vectors_2.T) / \
            (np.linalg.norm(keyword_vectors_1) * np.linalg.norm(keyword_vectors_2))
        #     print(sim)
        return sim

    def predict(self, program_keywords):
        cosine_list = []
        program_vector = self.map_keywords_to_vectors(program_keywords)
        for job_id in range(len(self.job_vectors)):
            cosine_list.append(self.calc_bert_keyword_similarity(self.job_vectors[job_id], program_vector))

        cosine_max_number, cosine_max_index = find_max_index(cosine_list, 3)
        recom = []
        # print("matched jobs:")
        for item in cosine_max_index:
            # print("\tJob title:",self.job_title[item])
            recom.append(self.job_title[item])
        return recom
'''
# 原始文本
program_description = "A course exploring the creation and design of handmade books, including bookbinding, letterpress printing, and artist's books as a form of artistic expression."
key_etc = key_extractor()
program_key = key_etc.get_key(program_description)
print(program_key)

# program_key = ["vibrant",	"breadth",	"ambition",	"curiosity",	"talent",	"equip",	"expertise",	"offering",	"student",	"innovation"]
predictor = Job_prediction()
print(predictor.predict(program_key))'''