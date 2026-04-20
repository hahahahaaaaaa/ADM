---
license: cc-by-nc-4.0
task_categories:
- text-generation
language:
- en
tags:
- adversarial robustness
- human red teaming
extra_gated_fields:
  Name: text
  Affiliation: text
  Email (affiliation email if possible): text
  Country: text
  Purpose: text
size_categories:
- n<1K
---


<style>
  button {
    /* margin: calc(20vw / 100); */
    margin: 0.5em;
    padding-left:  calc(40vw / 100);
    padding-right:  calc(40vw / 100);
    padding-bottom: calc(0vw / 100);
    text-align: center;
    font-size: 12px;
    height: 25px;
    transition: 0.5s;
    background-size: 200% auto;
    color: white;
    border-radius: calc(60vw / 100);
    display: inline;
    /* border: 2px solid black; */
    font-weight: 500;
    box-shadow: 0px 0px 14px -7px #f09819;
    background-image: linear-gradient(45deg, #64F 0%, #000000 51%, #FF512F 100%);
    cursor: pointer;
    user-select: none;
    -webkit-user-select: none;
    touch-action: manipulation;
  }
  button:hover {
    background-position: right center;
    color: #fff;
    text-decoration: none;
  }
  button:active {
    transform: scale(0.95);
  }
</style>

# Dataset Card for MHJ (08/27/2024)

<a href="https://scale.com/research/mhj" style="text-decoration:none">
  <button>Homepage</button>
</a>
<a href="https://huggingface.co/ScaleAI/mhj-llama3-8b-rmu" style="text-decoration:none">
  <button>Llama3-8B-RMU</button>
</a>

This card contains datasets for the paper *LLM Defenses Are Not Robust to Multi-Turn Human Jailbreaks*. 

We compile successful jailbreaks into the Multi-Turn Human Jailbreaks
(MHJ) dataset, consisting of 2,912 prompts across 537 multi-turn conversations. We include
relevant metadata for each submission, including design choice comments from each red teamer for their
jailbreak. The resulting attack success rate (ASR) of our human red teaming is shown as follows. 

![](./main_results.png)

[*CYGNET](https://www.grayswan.ai/) is closed
source, hence results for AutoDAN, GCG, and PAIR are cited from the original papers and should
not be directly compared against human ASR.

Due to potential export control concerns, we omitted some of the HarmBench behaviors and redacted some of the completions. We examined the HarmBench test set, standard and contextual behaviors for this research. For the same reason, we omitted the responses for WMDP-bio behaviors (if you just want to download the same WMDP-bio set we are using, please find here: https://huggingface.co/datasets/ScaleAI/mhj-wmdp-bio. 


## Paper Abstract

Recent large language model (LLM) defenses have greatly improved models’ ability to refuse harmful
queries, even when adversarially attacked. However, LLM defenses are primarily evaluated against
automated adversarial attacks in a single turn of conversation, an insufficient threat model for real-
world malicious use. We demonstrate that multi-turn human jailbreaks uncover significant vulnerabilities,
exceeding 70% attack success rate (ASR) on HarmBench against defenses that report single-digit ASRs
with automated single-turn attacks. Human jailbreaks also reveal vulnerabilities in machine unlearning
defenses, successfully recovering dual-use biosecurity knowledge from unlearned models. We compile
these results into Multi-Turn Human Jailbreaks (MHJ), a dataset of 2,912 prompts across 537 multi-turn
jailbreaks. We publicly release MHJ alongside a compendium of jailbreak tactics developed across dozens
of commercial red teaming engagements, supporting research towards stronger LLM defenses.

## Bibtex Citation
If you use this dataest, consider to cite
```
@misc{li2024llmdefensesrobustmultiturn,
      title={LLM Defenses Are Not Robust to Multi-Turn Human Jailbreaks Yet}, 
      author={Nathaniel Li and Ziwen Han and Ian Steneker and Willow Primack and Riley Goodside and Hugh Zhang and Zifan Wang and Cristina Menghini and Summer Yue},
      year={2024},
      eprint={2408.15221},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.15221}, 
}
```