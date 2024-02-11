from rouge import Rouge
import pandas as pd
from settings import *
import numpy as np
from tqdm import tqdm

def get_Rouge_score(list_generated,list_abstract):
    rouge = Rouge()
    dict_score=rouge.get_scores(list_generated,list_abstract,avg=True)
    return dict_score

def get_rouge_from_df(generate_df, rouge_type = 'rouge-l', metric = 'f'):
    df = pd.read_csv(os.path.join(OUT_DIR, generate_df))   
    value = 0
    for idx, row in df.iterrows():
        value_dic = get_Rouge_score(row['generate'], row['abstract'])
        value += value_dic[rouge_type][metric]
    return value / len(df)

def get_rouge_list_from_all_df(save_name, lower, upper, df_name):
    find_file_name =  f"{save_name}_{lower}_{upper}_{df_name}"
    file_list = [i for i in os.listdir(OUT_DIR) if '_'.join(i.split('_')[:-1]) == find_file_name]
    value_list = []
    for file in tqdm(file_list, desc = 'Get Rouge List From all Dataframe', total = len(file_list)):
        value = get_rouge_from_df(file)
        value_list.append(value)
    value_list = np.array(value_list)
    return value_list        

def statistic_from_rouge_list(result_name):
    rouge_dic = np.load(os.path.join(STATS_DIR, result_name), allow_pickle= 'TRUE').item()
    mean = round(np.mean(rouge_dic['values']), 3)
    std = round(np.std(rouge_dic['values']), 3)
    print('Rouge List: ', rouge_dic['values'])
    print(f"Mean :{mean}")
    print(f"Standard Deviation:{std}")



def save_rouge_avg(avg_array, save_name):
    result_dic = {'model_name':save_name, 'values':avg_array}
    if not os.path.exists(os.path.join(STATS_DIR, f"{save_name}_result.npy")):
        np.save(os.path.join(STATS_DIR, f"{save_name}_result.npy"), result_dic)
    else:
        print('기존 파일이 존재합니다. 다른 save_name을 설정해주세요.')
        raise