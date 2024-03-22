# RGB

- An implementation for [Benchmarking Large Language Models in Retrieval-Augmented Generation](https://arxiv.org/abs/2309.01431) 

## News

- \[2024/03\]  We refine the retrieved documents and some answers of `en.json` and `zh.json`, and name the new data files as `en_refine.json` and `zh_refine.json`. 

## Quick links

* [Environment](#Environment)
* [Retrieval-Augmented Generation Benchmark](#Retrieval-Augmented)
* [Evaluation](#Evaluation)
* [Licence](#Licence)

### Environment

```bash
conda create -n rgb python=3.10.0
conda activate rgb
bash env.sh
```

### Retrieval-Augmented Generation Benchmark

The data is putted in `data/`

```text
data/
├── en.json
├── en_refine.json
├── en_int.json
├── en_fact.json
├── zh.json
├── zh_refine.json
├── zh_int.json
└── zh_fact.json
```

To evalute the Information Integration, you should use `zh_int` or `en_int` for Chinese questions or English questions. 

To evalute the Counterfactual Robustness, you should use `zh_fact` or `en_fact` for Chinese questions or English questions. 

#### The refined data

We refine the retrieved documents and some answers of `en.json` and `zh.json`, and name the new data files as `en_refine.json` and `zh_refine.json`:

+ Removing incorrect positive and negative documents

+ Adding some positive documents.

+ Correcting some inaccurate answers.

### Evaluation

For evaluating ChatGPT, you can run as:

```bash
python evalue.py \
--dataset en \
--modelname chatgpt \
--temp 0.2 \
--noise_rate 0.6 \
--api_key YourAPIKEY \
--passage_num 5
```

For evaluating other models, you can run as:

```bash
python evalue.py \
--dataset en \
--modelname chatglm2-6b \
--temp 0.2 \
--noise_rate 0.6 \
--plm THUDM/chatglm-6b \
--passage_num 5
```

You should change `modelname` and `plm` for different models, where `plm` is the path of model.

`temp` is the temperature of model.

`noise_rate` is rate of noisy documents in inputs.

`passage_num` is number of provided documents for LLM (default is 5).

The outputs are:

+ all_rate: The accuracy (noise_rate<1) or rejection rate (noise_rate=1)
+ fact_check_rate: the error detection rates (ED)

---

To evaluate rejection using ChatGPT, you should first run the `evalue.py` in noise_rate=1 to obtain the generation result, and then run:

```bash
python reject_evalue.py \
--dataset en \
--modelname chatglm2-6b \
--api_key YourAPIKEY
```

The "reject_rate" in the outputs are the reject rate (Rej\*).

---

To evaluate counterfactual robustness using ChatGPT, you should first run the `evalue.py` in dataset=en_fact/zh_fact to obtain the generation result, and then run:

```bash
python fact_evalue.py \
--dataset en_fact \
--modelname chatglm2-6b \
--api_key YourAPIKEY
```

The "reject_rate" in the outputs are the error detection rates (ED\*). The `correct_rate` in the outputs are the error correction rate (CR)

## License

The code and data are released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License for Noncommercial use only. Any commercial use should get formal permission first.

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
