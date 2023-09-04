import json
import numpy as np
import random, math
import argparse,torch
import os
import json, tqdm, requests
import yaml
from models.models import *


    
def processdata(instance, noise_rate, passage_num, filename, correct_rate = 0):
    query = instance['query']
    ans = instance['answer']

    neg_num = math.ceil(passage_num * noise_rate)
    pos_num = passage_num - neg_num

    if '_int' in filename:
        for i in instance['positive']:
            random.shuffle(i)
        print(len(instance['positive']))
        docs = [i[0] for i in instance['positive']]
        maxnum = max([len(i) for i in instance['positive']])
        for i in range(1,maxnum):
            for j in instance['positive']:
                if len(j) > i:
                    docs.append(j[i])
                    if len(docs) == pos_num:
                        break
            if len(docs) == pos_num:
                break
        neg_num = passage_num - len(docs)
        if neg_num > 0:
            negative = instance['negative'][:neg_num]
            docs += negative
    elif '_fact' in filename:
        correct_num = math.ceil(passage_num * correct_rate)
        pos_num = passage_num - neg_num - correct_num
        indexs = list(range(len(instance['positive'])))
        selected = random.sample(indexs,min(len(indexs),pos_num))
        docs = [instance['positive_wrong'][i] for i in selected]
        remain = [i for i in indexs if i not in selected]
        if correct_num > 0 and len(remain) > 0:
            docs += [instance['positive'][i] for i in random.sample(remain,min(len(remain),correct_num))]
        if neg_num > 0:
            docs += instance['negative'][:neg_num]
    else:
        if noise_rate == 1:
            neg_num = passage_num
            pos_num = 0
        else:
            if neg_num > len(instance['negative']):
                neg_num = len(instance['negative'])
                pos_num = passage_num - neg_num
            elif pos_num > len(instance['positive']):
                pos_num = len(instance['positive'])
                neg_num = passage_num - pos_num
        

        positive = instance['positive'][:pos_num]
        negative = instance['negative'][:neg_num]

        docs = positive + negative

    random.shuffle(docs)
    
    return query, ans, docs


def checkanswer(prediction, ground_truth):
    prediction = prediction.lower()
    if type(ground_truth) is not list:
        ground_truth = [ground_truth]
    labels = []
    for instance in ground_truth:
        flag = True
        if type(instance)  == list:
            flag = False 
            instance = [i.lower() for i in instance]
            for i in instance:
                if i in prediction:
                    flag = True
                    break
        else:
            instance = instance.lower()
            if instance not in prediction:
                flag = False
        labels.append(int(flag))
    return labels

def getevalue(results):
    results = np.array(results)
    results = np.max(results,axis = 0)
    if 0 in results:
        return False
    else:
        return True
            

def predict(query, ground_truth, docs, model, system, instruction, temperature, dataset):

    '''
    label: 0 for positive, 1 for negative, -1 for not enough information

    '''
    if len(docs) == 0:

        text = instruction.format(QUERY=query, DOCS='')
        prediction = model.generate(text, temperature)

    else:

        docs = '\n'.join(docs)

        

        text = instruction.format(QUERY=query, DOCS=docs)

        prediction = model.generate(text, temperature, system)

    if 'zh' in dataset:
        prediction = prediction.replace(" ","")

    if '信息不足' in prediction or 'insufficient information' in prediction:
        labels = [-1]
    else:
        labels = checkanswer(prediction, ground_truth)
    
    factlabel = 0

    if '事实性错误' in prediction or 'factual errors' in prediction:
        factlabel = 1

    return labels,prediction, factlabel



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--modelname', type=str, default='chatgpt',
        help='model name'
    )
    parser.add_argument(
        '--dataset', type=str, default='en',
        help='evaluetion dataset',
        choices=['en','zh','en_int','zh_int','en_fact','zh_fact']
    )
    parser.add_argument(
        '--api_key', type=str, default='api_key',
        help='api key of chatgpt'
    )
    parser.add_argument(
        '--plm', type=str, default='THUDM/chatglm-6b',
        help='name of plm'
    )
    parser.add_argument(
        '--url', type=str, default='https://api.openai.com/v1/completions',
        help='url of chatgpt'
    )
    parser.add_argument(
        '--temp', type=float, default=0.7,
        help='corpus id'
    )
    parser.add_argument(
        '--noise_rate', type=float, default=0.0,
        help='rate of noisy passages'
    )
    parser.add_argument(
        '--correct_rate', type=float, default=0.0,
        help='rate of correct passages'
    )
    parser.add_argument(
        '--passage_num', type=int, default=5,
        help='number of external passages'
    )
    parser.add_argument(
        '--factchecking', type=bool, default=False,
        help='whether to fact checking'
    )
    
    args = parser.parse_args()

    modelname = args.modelname
    temperature = args.temp
    noise_rate = args.noise_rate
    passage_num = args.passage_num

    instances = []
    with open(f'data/{args.dataset}.json','r') as f:
        for line in f:
            instances.append(json.loads(line))
    if 'en' in args.dataset:
        resultpath = 'result-en'
    elif 'zh' in args.dataset:
        resultpath = 'result-zh'
    if not os.path.exists(resultpath):
        os.mkdir(resultpath)

    if args.factchecking:
        prompt = yaml.load(open('config/instruction_fact.yaml', 'r'), Loader=yaml.FullLoader)[args.dataset[:2]]
        resultpath = resultpath + '/fact'
    else:
        prompt = yaml.load(open('config/instruction.yaml', 'r'), Loader=yaml.FullLoader)[args.dataset[:2]]

    system = prompt['system']
    instruction = prompt['instruction']

    if modelname == 'chatgpt':
        model = OpenAIAPIModel(api_key = args.api_key, url = args.url)
    elif 'Llama-2' in modelname:
        model = LLama2(plm = args.plm)
    elif 'chatglm' in modelname:
        model = ChatglmModel(plm = args.plm)
    elif 'moss' in modelname:
        model = Moss(plm = args.plm)
    elif 'vicuna' in modelname:
        model = Vicuna(plm = args.plm)
    elif 'Qwen' in modelname:
        model = Qwen(plm = args.plm)
    elif 'Baichuan' in modelname:
        model = Baichuan(plm = args.plm)
    elif 'WizardLM' in modelname:
        model = WizardLM(plm = args.plm)
    elif 'BELLE' in modelname:
        model = BELLE(plm = args.plm)
    


    filename = f'{resultpath}/prediction_{args.dataset}_{modelname}_temp{temperature}_noise{noise_rate}_passage{passage_num}_correct{args.correct_rate}.json'
    useddata = {}
    if os.path.exists(filename):
        with open(filename) as f:
            for line in f:
                data = json.loads(line)
                useddata[data['id']] = data
  
    results = []
    with open(filename,'w') as f:
        for instance in tqdm.tqdm(instances):
            if instance['id'] in useddata and instance['query'] == useddata[instance['id']]['query'] and instance['answer']  == useddata[instance['id']]['ans']:
                results.append(useddata[instance['id']])
                f.write(json.dumps(useddata[instance['id']], ensure_ascii=False)+'\n')
                continue
            try:
                random.seed(2333)
                if passage_num == 0:
                    query = instance['query']
                    ans = instance['answer']
                    docs = []
                else:
                    query, ans, docs = processdata(instance, noise_rate, passage_num, args.dataset, args.correct_rate)
                label,prediction,factlabel = predict(query, ans, docs, model,system,instruction,temperature,args.dataset)
                instance['label'] = label
                newinstance = {
                    'id': instance['id'],
                    'query': query,
                    'ans': ans,
                    'label': label,
                    'prediction': prediction,
                    'docs': docs,
                    'noise_rate': noise_rate,
                    'factlabel': factlabel
                }
                results.append(newinstance)
                f.write(json.dumps(newinstance, ensure_ascii=False)+'\n')
            except Exception as e:
                print("Error:", e)
                continue
    tt = 0
    for i in results:
        label = i['label']
        if noise_rate == 1 and label[0] == -1:
            tt += 1
        elif 0 not in label and 1 in label:
            tt += 1
    print(tt/len(results))
    scores = {
    'all_rate': (tt)/len(results),
    'noise_rate': noise_rate,
    'tt':tt,
    'nums': len(results),
    }
    if '_fact' in args.dataset:
        fact_tt = 0
        correct_tt = 0
        for i in results:
            if i['factlabel'] == 1:
                fact_tt += 1
                if 0 not in i['label']:
                    correct_tt += 1
        fact_check_rate = fact_tt/len(results)
        if fact_tt > 0:
            correct_rate = correct_tt/fact_tt
        else:
            correct_rate = 0
        scores['fact_check_rate'] = fact_check_rate
        scores['correct_rate'] = correct_rate
        scores['fact_tt'] = fact_tt
        scores['correct_tt'] = correct_tt

    

    json.dump(scores,open(f'{resultpath}/prediction_{args.dataset}_{modelname}_temp{temperature}_noise{noise_rate}_passage{passage_num}_correct{args.correct_rate}_result.json','w'),ensure_ascii=False,indent=4)