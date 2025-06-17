SIZES = [
    100_000,
    500_000,
    1_000_000,
]

VOCABS = [
   8_000,
   16_000,
   32_000,
]

TOKENIZER_PAIRS = [
   ("fra", "ita", SIZES),
   ("ces", "ukr", SIZES),
   ("ita", "mlt", [100_000]),
   ("deu", "hsb", [60_000])
]

ALIGNMENTS = [
    "align",
]

FLORES_SPLITS = [
    "dev",
    "devtest",
]

# MT_SEEDS = [3,5,42,1,25]
MT_SEEDS = [3,5,42]

wildcard_constraints:
    l1="[a-z]{3}",
    l2="[a-z]{3}",
    lang="[a-z]{3}",
    size="[0-9]+",
    sp_type="bpe|unigram",
    vocab="[0-9]+",
    alignment="align|phrases",
    flores_split="dev|devtest",

def training_data_tokenizer(*args, **kargs):
    files = []
    for l1, l2, sizes in TOKENIZER_PAIRS:
        files.append(f"data/{l1}-{l2}/flores.dev.{l1}")
        files.append(f"data/{l1}-{l2}/flores.dev.{l2}")
        files.append(f"data/{l1}-{l2}/flores.devtest.{l1}")
        files.append(f"data/{l1}-{l2}/flores.devtest.{l2}")
        for size in sizes:
            files.append(f"data/{l1}-{l2}/train.{size}.{l1}")
            files.append(f"data/{l1}-{l2}/train.{size}.{l2}")
    return files

def training_aligned_data(*args, **kargs):
    files = []
    for l1, l2, sizes in TOKENIZER_PAIRS:
        for size in sizes:
            for alignment in ALIGNMENTS:
                files.append(f"data/{l1}-{l2}/{alignment}/train.{size}.{l1}")
                files.append(f"data/{l1}-{l2}/{alignment}/train.{size}.{l2}")
                files.append(f"data/{l2}-{l1}/{alignment}/train.{size}.{l1}")
                files.append(f"data/{l2}-{l1}/{alignment}/train.{size}.{l2}")
    return files

def sp_models(*args, **kargs):
    files = []
    for l1, l2, sizes in TOKENIZER_PAIRS:
        for size in sizes:
            for vocab in VOCABS:
                files.append(f"models/{l1}-{l2}/{vocab}/{size}/unigram.{l1}.model")
                files.append(f"models/{l1}-{l2}/{vocab}/{size}/unigram.{l2}.model")
    return files

def psp_models(*args, **kargs):
    files = []
    for l1, l2, sizes in TOKENIZER_PAIRS:
        for size in sizes:
            for vocab in VOCABS:
                for alignment in ALIGNMENTS:
                    files.append(f"models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.tar.gz")
                    files.append(f"models/{l2}-{l1}/{vocab}/{size}/{alignment}/{l1}.tar.gz")
                    files.append(f"models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.hard.tar.gz")
                    files.append(f"models/{l2}-{l1}/{vocab}/{size}/{alignment}/{l1}.hard.tar.gz")
    return files

def psp_flores(*args, **kargs):
    files = []
    for l1, l2, sizes in TOKENIZER_PAIRS:
        for size in sizes:
            for vocab in VOCABS:
                for alignment in ALIGNMENTS:
                    files.append(f"data/{l1}-{l2}/psp/{vocab}/{size}/{alignment}/flores.devtest.{l2}.tok")
                    files.append(f"data/{l2}-{l1}/psp/{vocab}/{size}/{alignment}/flores.devtest.{l1}.tok")
    return files

def intrinsic_scores(*args, **kargs):
    files = []
    for l1, l2, sizes in TOKENIZER_PAIRS:
        for size in sizes:
            for vocab in VOCABS:
                for alignment in ALIGNMENTS:
                    files.append(f"models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.scores")
                    files.append(f"models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.scores.unpaired")                    
                    files.append(f"models/{l2}-{l1}/{vocab}/{size}/{alignment}/{l1}.scores")
                    files.append(f"models/{l2}-{l1}/{vocab}/{size}/{alignment}/{l1}.scores.unpaired")
                    files.append(f"models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.scores.hard")                    
                    files.append(f"models/{l2}-{l1}/{vocab}/{size}/{alignment}/{l1}.scores.hard")
                    files.append(f"models/{l1}-{l2}/{vocab}/{size}/{alignment}/{alignment}_unigram.{l2}.scores.eflomal") # PSP
                    files.append(f"models/{l2}-{l1}/{vocab}/{size}/{alignment}/{alignment}_unigram.{l1}.scores.eflomal")
                    files.append(f"models/{l1}-{l2}/{vocab}/{size}/{alignment}_unigram.{l2}.scores.eflomal") # SP
                    files.append(f"models/{l2}-{l1}/{vocab}/{size}/{alignment}_unigram.{l1}.scores.eflomal") # SP
                files.append(f"models/{l1}-{l2}/{vocab}/{size}/unigram.{l2}.scores")
                files.append(f"models/{l2}-{l1}/{vocab}/{size}/unigram.{l1}.scores")
    return files

def mt_scores(*args, **kargs):
    files = []
    for l1, l2, sizes in TOKENIZER_PAIRS:
        size = max(sizes)
        for vocab in VOCABS:            
            for seed in MT_SEEDS:
                for alignment in ["align"]:
                    files.append(f"models/mt/{l1}-{l2}/{vocab}/{size}/{alignment}/seed_{seed}/model.npz.scores")
                    files.append(f"models/mt/{l2}-{l1}/{vocab}/{size}/{alignment}/seed_{seed}/model.npz.scores")
                    files.append(f"models/mt/{l1}-{l2}/{vocab}/{size}/{alignment}/seed_{seed}/model.npz.comet")
                    files.append(f"models/mt/{l2}-{l1}/{vocab}/{size}/{alignment}/seed_{seed}/model.npz.comet")
                files.append(f"models/mt/{l1}-{l2}/{vocab}/{size}/baseline/seed_{seed}/model.npz.scores")
                files.append(f"models/mt/{l2}-{l1}/{vocab}/{size}/baseline/seed_{seed}/model.npz.scores")
                files.append(f"models/mt/{l1}-{l2}/{vocab}/{size}/baseline/seed_{seed}/model.npz.comet")
                files.append(f"models/mt/{l2}-{l1}/{vocab}/{size}/baseline/seed_{seed}/model.npz.comet")
    return files

def lm_scores(*args, **kargs):
    files = []
    for l1, l2, sizes in TOKENIZER_PAIRS:
        size = max(sizes)
        for vocab in VOCABS:            
            for seed in MT_SEEDS:
                for alignment in ["align"]:
                    files.append(f"models/lm/{l1}-{l2}/{vocab}/{size}/{alignment}/seed_{seed}/all_results.json")
                    files.append(f"models/lm/{l2}-{l1}/{vocab}/{size}/{alignment}/seed_{seed}/all_results.json")
                files.append(f"models/lm/{l1}-{l2}/{vocab}/{size}/baseline/seed_{seed}/all_results.json")
                files.append(f"models/lm/{l2}-{l1}/{vocab}/{size}/baseline/seed_{seed}/all_results.json")
    return files

def eflomal_scores(*args, **kargs):
    files = []
    for l1, l2, sizes in TOKENIZER_PAIRS:
        for size in sizes:
            for vocab in VOCABS:
                for alignment in ALIGNMENTS:
                    files.append(f"models/{l1}-{l2}/{vocab}/{size}/{alignment}/{alignment}_unigram.{l2}.scores.eflomal2") # PSP
                    files.append(f"models/{l2}-{l1}/{vocab}/{size}/{alignment}/{alignment}_unigram.{l1}.scores.eflomal2")
                    files.append(f"models/{l1}-{l2}/{vocab}/{size}/{alignment}_unigram.{l2}.scores.eflomal2") # SP
                    files.append(f"models/{l2}-{l1}/{vocab}/{size}/{alignment}_unigram.{l1}.scores.eflomal2") # SP
    return files

localrules: all

rule all:
    input:
        # eflomal_scores
        # psp_models,
        # mt_scores  
        lm_scores

include: "Snakefiles/download.smk"
include: "Snakefiles/align.smk"
include: "Snakefiles/sentencepiece.smk"
include: "Snakefiles/paired_sp.smk"
include: "Snakefiles/intrinsic.smk"
include: "Snakefiles/mt.smk"
include: "Snakefiles/lm.smk"

ruleorder: merge_sp_parts > merge_psp_parts > merge_usp_parts > sp_tokenize_train > tokenize_sp 
