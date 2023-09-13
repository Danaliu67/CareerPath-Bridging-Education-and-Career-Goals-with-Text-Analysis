import numpy as np
import pandas as pd
import torch
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig

from ..utils import find_max_index


class JobPrediction():
    def __init__(self):
        file_path = "data/job/keywords.xls"
        raw_data = pd.read_excel(file_path, header=0)  
        self.job_data = raw_data.values
        df1 = pd.read_csv("data/job/total.csv")
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
        # calculate cosine similarity
        sim = np.dot(keyword_vectors_1, keyword_vectors_2.T) / \
            (np.linalg.norm(keyword_vectors_1) * np.linalg.norm(keyword_vectors_2))
        return sim

    def predict(self, program_keywords):
        cosine_list = []
        program_vector = self.map_keywords_to_vectors(program_keywords)
        for job_id in range(len(self.job_vectors)):
            cosine_list.append(self.calc_bert_keyword_similarity(self.job_vectors[job_id], program_vector))
        _, cosine_max_index = find_max_index(cosine_list, 3)
        recom = []
        for item in cosine_max_index:
            recom.append(self.job_title[item])
        return recom
