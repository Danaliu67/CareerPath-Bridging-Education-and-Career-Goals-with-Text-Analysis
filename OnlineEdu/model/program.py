import numpy as np
import pandas as pd
import torch
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig

from ..utils import find_max_index
from ..text import KeyExtractor


class ProgramPrediction():
    def __init__(self):
        # read program
        program_file = "data/program/keywords.xls"
        program_raw_data = pd.read_excel(program_file, header=0)  
        self.program_data = program_raw_data.values
        # read program
        df2 = pd.read_csv("data/program/total.csv")
        self.school_name = df2['Schoole']
        self.program_name = df2['program']
        config=RobertaConfig.from_pretrained('roberta-base')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaModel.from_pretrained('roberta-base',config=config)
        self.program_vectors = []
        for j in range(len(self.program_data)):
            row = self.program_data[j][1:]
            newlist = [x for x in row if pd.isnull(x) == False]
            input_2=self.tokenizer(' '.join(newlist),return_tensors="pt")
            output_2=self.model(**input_2)
            keyword_vectors_2 = torch.mean(output_2.last_hidden_state, dim=1).detach().numpy()
            self.program_vectors.append(keyword_vectors_2) 

    def map_keywords_to_vectors(self, keywords):
        input =self.tokenizer(' '.join(keywords),return_tensors="pt")
        output = self.model(**input)
        keywords_vector = torch.mean(output.last_hidden_state, dim=1).detach().numpy()
        return keywords_vector

    def calc_bert_keyword_similarity(self,keyword_vectors_1, keyword_vectors_2):
        sim = np.dot(keyword_vectors_1, keyword_vectors_2.T) / \
            (np.linalg.norm(keyword_vectors_1) * np.linalg.norm(keyword_vectors_2))
        return sim

    def predict(self, job_keywords):
        cosine_list = []
        job_vector = self.map_keywords_to_vectors(job_keywords)
        for program_id in range(len(self.program_vectors)):
            cosine_list.append(self.calc_bert_keyword_similarity(self.program_vectors[program_id], job_vector))
        _, cosine_max_index = find_max_index(cosine_list, 3)
        recom = []
        for item in cosine_max_index:
            recom.append(self.school_name[item] + '/' + self.program_name[item])
        return recom
    

if __name__ == "__main__":
    job_requirement = """
        Diverse role with close cooperation with the Managing Director and team. 
        Fantastic team culture. 
        Have oversight over a diverse range of accounts.
    """
    key_etc = KeyExtractor()
    job_key = key_etc.get_key(job_requirement)
    print(job_key)
