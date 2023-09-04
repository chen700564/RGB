import requests


def check(question, answer, url,apikey):
    prompt = '''Please determine whether the model can identify factual errors in the document. I will provide you with a response from the model, and if that response mentions that the document contains factual errors, it means that the model has successfully identified them. Please note that you should not determine whether the reply itself contains factual errors.
Demonstrations:
Response:文档信息有误，2008年的奥运会在北京举办。
Yes, the model has identified the factual errors.

Response:2014年澳网女单冠军是大阪直美。
NO, the model fail to identify the factual errors.

Response: The director of the Silent Hill movie is Justin Kurzel.
NO, the model fail to identify the factual errors.

Response: Harry Potter is written by J. K. Rowling.
NO, the model fail to identify the factual errors.

Response:  There are factual errors in the provided documents. The correct answer is 2023.
Yes, the model has identified the factual errors.

Begin to generate:
Answer: {answer}
    '''
    text2 = prompt.format(answer=answer)
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
    parser.add_argument(
        '--noise_rate', type=float, default=0.0,
        help='rate of noisy passages'
    )
    parser.add_argument(
        '--correct_rate', type=float, default=0.0,
        help='rate of correct passages'
    )

    args = parser.parse_args()

    if 'en' in args.dataset:
        resultpath = 'result-en'
    elif 'zh' in args.dataset:
        resultpath = 'result-zh'

    evaluefile = f'{resultpath}/prediction_{args.dataset}_{args.modelname}_temp{args.temp}_noise{args.noise_rate}_passage{args.passage_num}_correct{args.correct_rate}.json'

    outputfile = f'{resultpath}/prediction_{args.dataset}_{args.modelname}_temp{args.temp}_noise{args.noise_rate}_passage{args.passage_num}_correct{args.correct_rate}_chatgpt.json'

    resultfile = f'{resultpath}/prediction_{args.dataset}_{args.modelname}_temp{args.temp}_noise{args.noise_rate}_passage{args.passage_num}_correct{args.correct_rate}_chatgptresult.json'



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
                if data['id'] in useddata:
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
    correct_tt = 0
    for i in results:
        if "has identified" in i['evaluation'] or "Yes" in i['evaluation']:
            rejecttt += 1
            if 0 not in i['label'] and 1 in i['label']:
                correct_tt += 1
        if 0 not in i['label'] and 1 in i['label']:
            tt += 1
    print(tt/len(results))
    scores = {
        'reject_rate': rejecttt/len(results),
        'all_rate': (tt)/len(results),
        'correct_rate': correct_tt/rejecttt if rejecttt > 0 else 0,
        'tt':tt,
        'rejecttt':rejecttt,
        'correct_tt':correct_tt,
        'nums': len(results),
        'noise_rate': args.noise_rate,
    }
    json.dump(scores, open(resultfile, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)