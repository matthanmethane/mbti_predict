import pandas as pd
import re

from pandas.core.frame import DataFrame
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

class Lemmatizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
    def __call__(self, sentence):
        return [self.lemmatizer.lemmatize(word) for word in sentence.split() if len(word)>2]


        
def clear_text(data):
    data_length=[]
    cleaned_text=[]
    for sentence in tqdm(data.posts):
        sentence=sentence.lower()
        #Removing links from text data
        sentence=re.sub('https?://[^\s<>"]+|www\.[^\s<>"]+',' ',sentence)
        #Chaging ..+ to keyword helium
        sentence=re.sub('\.\.+','helium',sentence)
        #Chaging ?+ to keyword barium
        sentence=re.sub('\?+','barium',sentence)
        #Chaging !+ to keyword zinc
        sentence=re.sub('\!+','zinc',sentence)
        #Removing other symbols
        sentence=re.sub('[^0-9a-z]',' ',sentence)
        
        data_length.append(len(sentence.split()))
        cleaned_text.append(sentence)
    return cleaned_text,data_length

def data_preprocessor(data: DataFrame):
    mbti_type_list = []
    mbti_df_list = []
    train_data_list, test_data_list = [], []
    train_length_list, test_length_list = [] , []
    vectorizer_list = []
    train_post_list, test_post_list = [],[]
    train_target_list, test_target_list, target_encoder_list = [] , [] , []
    for i in range(4):
        mbti_type_list.append([mbti[i] for mbti in data['type']])
        content_list = data.posts.to_list()
        mbti_df_list.append(pd.DataFrame({'type':mbti_type_list[i], 'posts': content_list}, columns=['type','posts']))
        train_data, test_data = train_test_split(mbti_df_list[i],test_size=0.25,random_state=35,stratify=data.type)
        train_data_list.append(train_data)
        test_data_list.append(test_data)
        train_data_list[i].posts,train_length=clear_text(train_data_list[i])
        train_length_list.append(train_length)
        test_data_list[i].posts,test_length=clear_text(test_data_list[i])
        test_length_list.append(test_length)
        vectorizer = TfidfVectorizer( max_features=5000,stop_words='english',tokenizer=Lemmatizer())
        vectorizer.fit(train_data_list[i].posts)
        vectorizer_list.append(vectorizer)
        train_post_list.append(vectorizer_list[i].transform(train_data_list[i].posts).toarray())
        test_post_list.append(vectorizer_list[i].transform(test_data_list[i].posts).toarray())
        target_encoder=LabelEncoder()
        train_target_list.append(target_encoder.fit_transform(train_data_list[i].type))
        test_target_list.append(target_encoder.fit_transform(test_data_list[i].type))
        target_encoder_list.append(target_encoder)
    return train_post_list,test_post_list,train_target_list,test_target_list,target_encoder_list, vectorizer_list

def get_clean_dataframe(file_dir):
    data = pd.read_csv(file_dir)
    return data_preprocessor(data)
