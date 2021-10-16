from sklearn.linear_model import LogisticRegression
import sys, os


from .data_preprocessing import get_clean_dataframe
from sklearn.metrics import classification_report
import pandas as pd
from .data_preprocessing import clear_text
import pickle


class LogisticRegressor():
    def __init__(self, file_dir):
        self.train_post_list,self.test_post_list,self.train_target_list,self.test_target_list,self.target_encoder_list, self.vectorizer_list = get_clean_dataframe(file_dir)
        self.filename_log = 'model_log_list.sav'
        self.filename_vector = 'model_vector_list.sav'

    def learn(self):   
        self.model_log_list = []
        for i in range(4):
            model_log=LogisticRegression(max_iter=3000,C=0.5,n_jobs=-1)
            model_log.fit(self.train_post_list[i],self.train_target_list[i])
            self.model_log_list.append(model_log)
        
        pickle.dump(self.model_log_list, open(self.filename_log, 'wb'))
        pickle.dump(self.vectorizer_list, open(self.filename_vector, 'wb'))

    def get_report(self):
        for i in range(4):
            print('train classification report \n ',classification_report(self.train_target_list[i],self.model_log_list[i].predict(self.train_post_list[i]),target_names=self.target_encoder_list[i].inverse_transform([n for n in range(2)])))


def mbti_vec_to_str(self, mbti_vec):
        mbti_str = ""
        mbti_str += 'I' if mbti_vec[0] else 'E'
        mbti_str += 'S' if mbti_vec[0] else 'N'
        mbti_str += 'T' if mbti_vec[0] else 'F'
        mbti_str += 'P' if mbti_vec[0] else 'J'
        return mbti_str



def predict_mbti(data):
    data_posts = data.rename(columns={'text':'posts'})  
    data_posts = data_posts.applymap(str)
    data_posts = pd.DataFrame({'posts':[' '.join([x for x in data_posts.posts])]},columns=['posts'])
    sample_post_list =[]
    filename_log = r'predict_model\ml_model\model_log_list.sav'
    model_log_list = pickle.load(open(filename_log, 'rb'))
    filename_vector = r'predict_model\ml_model\model_vector_list.sav'
    vectorizer_list = pickle.load(open(filename_vector, 'rb'))
    for i in range(4):
        data_posts.posts,train_length=clear_text(data_posts)
        sample_post_list.append(vectorizer_list[i].transform(data_posts.posts).toarray())
    mbti_vec = ""
    for i in range(4):
        mbti_vec += str(model_log_list[i].predict(sample_post_list[i]))
    return mbti_vec_to_str(mbti_vec)
