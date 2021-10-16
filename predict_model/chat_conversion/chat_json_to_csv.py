import json
import pandas as pd

def chat_json_to_csv(f: json):
    data = json.load(f)

    msgs = data['messages']

    dmain = pd.DataFrame(msgs)
    df= dmain.filter(['from', 'text'])
    person = df['from'].unique()
    A = person[0]
    B = person[1]
    A_data=df[(df['from']== A)]
    # A_data.to_csv('A_data.csv')
    B_data=df[(df['from']== B)]
    # B_data.to_csv('B_data.csv')
    return A_data, B_data