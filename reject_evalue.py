import requests


def check(question, answer, url, apikey):
    prompt = '''I will give you a question and an answer generated through document retrieval. Please use this answer to determine if the retrieved document can solve the question.
Demonstrations:
Question: 2023年澳网女单冠军是谁
Answer:文档信息不足，因此我无法基于提供的文档回答该问题。
No, the question is not addressed by the documents.

Question: Who is the champion of Australian Open 2023 Women's Singles?
Answer: Serena Williams
Yes, the question is addressed by the documents.

Question: Where is ACL2023 held?
Answer: Location of ACL2023 has not been confirmed.
No, the question is not addressed by the documents.

Question:  2023年中国GDP是多少?
Answer: I can not answer this question。
No, the question is not addressed by the documents.

Begin to generate:
Question: {question}
Answer: {answer}
    '''
    text2 = prompt.format(question=question,answer=answer)
    return getdata(text2,url,apikey)


def getdata(text,url,API_KEY):
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": text}]
    }
    headers={"Authorization": f"Bearer {API_KEY}"}
    completion = requests.post(url, json=data, headers=headers)
    completion = completion.json()['choices'][0]['message']['content']
    return completion

import json
import tqdm, os

import argparse

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
        '--url', type=str, default='https://api.openai.com/v1/completions',
        help='url of chatgpt'
    )
    parser.add_argument(
        '--temp', type=float, default=0.7,
        help='corpus id'
    )
    parser.add_argument(
        '--passage_num', type=int, default=5,
        help='number of external passages'
    )

    args = parser.parse_args()

    if 'en' in args.dataset:
        resultpath = 'result-en'
    elif 'zh' in args.dataset:
        resultpath = 'result-zh'

    evaluefile = f'{resultpath}/prediction_{args.dataset}_{args.modelname}_temp{args.temp}_noise{1.0}_passage{args.passage_num}_correct{0.0}.json'

    outputfile = f'{resultpath}/prediction_{args.dataset}_{args.modelname}_temp{args.temp}_noise{1.0}_passage{args.passage_num}_correct{0.0}_chatgpt.json'

    resultfile = f'{resultpath}/prediction_{args.dataset}_{args.modelname}_temp{args.temp}_noise{1.0}_passage{args.passage_num}_correct{0.0}_chatgptresult.json'



    results = []
    useddata = {}
    if os.path.exists(outputfile):
        with open(outputfile) as f:
            for line in f:
                data = json.loads(line)
                useddata[data['id']] = data
    


    with open(outputfile,'w',encoding='utf-8') as f:
        with open(evaluefile, 'r', encoding='utf-8') as f2:
            for line in tqdm.tqdm(f2):
                data = json.loads(line)
                if data['id'] in useddata and data['query'] == useddata[data['id']]['query'] and data['ans']  == useddata[data['id']]['ans'] :
                    results.append(useddata[data['id']])
                    f.write(json.dumps(useddata[data['id']],ensure_ascii=False)+'\n')
                    continue
                try:
                    question = data['query']
                    answer = data['prediction']
                    
                    evaluation = check(question, answer, args.url, args.api_key)
                    data['evaluation'] = evaluation
                    results.append(data)
                    f.write(json.dumps(data,ensure_ascii=False)+'\n')
                except Exception as e:
                    print(e)
                    print(question,answer)
                    continue
    
    rejecttt = 0
    tt = 0
    for i in results:
        if "not addressed" in i['evaluation']:
            rejecttt += 1
        if 0 not in i['label'] and 1 in i['label']:
            tt += 1
    print(tt/len(results))
    scores = {
        'reject_rate': rejecttt/len(results),
        'all_rate': (tt)/len(results),
        'tt':tt,
        'rejecttt':rejecttt,
        'nums': len(results),
    }
    json.dump(scores, open(resultfile, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)