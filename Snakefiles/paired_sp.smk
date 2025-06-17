def mem_estimate(wildcards):
    if wildcards.alignment == "phrases":
        return 50_000
    elif wildcards.alignment == "align":
        if wildcards.size == "100000":
            return 7_000
        elif wildcards.size == "500000":
            return 15_000
        elif wildcards.size == "1000000":
            return 20_000
    return 35_000
        

rule train_psp:
    input:
        src_tok="data/{l1}-{l2}/unigram/{vocab}/{size}/{alignment}/train.{size}.{l1}.tok",
        trg_plain="data/{l1}-{l2}/{alignment}/train.{size}.{l2}",
    output:
        "models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.tar.gz",
    resources:
        mem_mb=mem_estimate,
        cpus_per_task=1,
        tasks=1,
    params:
        iters=100,
        sub_iters=2,
        char_coverage=1,
        reduce=0.75,
        threshold=lambda wildcards: 500 if wildcards.alignment == "phrases" else 0,
    log:
        "models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.tar.gz.log",
    shell:
        "/usr/bin/time -v python -m paired_sp train -m cond-exact --reduce {params.reduce} --src {input.src_tok} --trg {input.trg_plain} -p "
            "-v {wildcards.vocab} -o {output} -i {params.iters} --sub-iter {params.sub_iters} --threshold {params.threshold} "
            "--char-coverage {params.char_coverage} --info &> {log}"

rule train_psp_hard:
    input:
        src_tok="data/{l1}-{l2}/unigram/{vocab}/{size}/{alignment}/train.{size}.{l1}.tok",
        trg_plain="data/{l1}-{l2}/{alignment}/train.{size}.{l2}",
    output:
        "models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.hard.tar.gz",
    resources:
        mem_mb=mem_estimate,
        cpus_per_task=1,
        tasks=1,
    params:
        iters=100,
        sub_iters=2,
        char_coverage=1,
        reduce=0.75,
        threshold=lambda wildcards: 500 if wildcards.alignment == "phrases" else 0,
    log:
        "models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.tar.gz.log",
    shell:
        "/usr/bin/time -v python -m paired_sp train -m cond-exact --reduce {params.reduce} --src {input.src_tok} --trg {input.trg_plain} -p "
            "-v {wildcards.vocab} -o {output} -i {params.iters} --sub-iter {params.sub_iters} --threshold {params.threshold} "
            "--char-coverage {params.char_coverage} --trainer em --info &> {log}"


rule tokenize_psp_paired_flores:
    input:
        src_tok="data/{l1}-{l2}/unigram/{vocab}/{size}/flores.{flores_split}.{l1}.tok",
        trg_plain="data/{l1}-{l2}/flores.{flores_split}.{l2}",
        model="models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.tar.gz",
    output:
        "data/{l1}-{l2}/psp/{vocab}/{size}/{alignment}/flores.{flores_split}.{l2}.tok",
    resources:
        mem_mb=4_000,
        cpus_per_task=1,
        tasks=1,
    shell:
        "/usr/bin/time -v python -m paired_sp tokenize -m cond-exact --model-file {input.model} --src {input.src_tok} --trg {input.trg_plain} -o {output}"

use rule tokenize_psp_paired_flores as tokenize_psp_hard_flores with:
    input:
        src_tok="data/{l1}-{l2}/unigram/{vocab}/{size}/flores.{flores_split}.{l1}.tok",
        trg_plain="data/{l1}-{l2}/flores.{flores_split}.{l2}",
        model="models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.hard.tar.gz",
    output:
        "data/{l1}-{l2}/psp/{vocab}/{size}/{alignment}/flores.{flores_split}.{l2}.hard.tok",

rule tokenize_psp_unpaired_flores:
    input:
        src_tok="data/{l1}-{l2}/unigram/{vocab}/{size}/flores.{flores_split}.{l1}.tok",
        trg_plain="data/{l1}-{l2}/flores.{flores_split}.{l2}",        
        model="models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.tar.gz",
    output:
        "data/{l1}-{l2}/psp/{vocab}/{size}/{alignment}/flores.{flores_split}.{l2}.tok.unpaired",
    resources:
        mem_mb=4_000,
        cpus_per_task=1,
        tasks=1,
    shell:
        "/usr/bin/time -v python -m paired_sp tokenize -m cond-exact --model-file {input.model} -m sum-unpaired --src {input.src_tok} --trg {input.trg_plain} -o {output}"

rule psp_vocab:
    input:
        "models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.tar.gz"
    output:
        "models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.yml"
    resources:
        mem_mb=4_000,
        cpus_per_task=1,
        tasks=1,
    shell:
        "python -m paired_sp vocab -m cond-exact --model-file {input} -o {output}"

use rule psp_vocab as psp_vocab_hard with:
    input:
        "models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.hard.tar.gz"
    output:
        "models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.hard.yml"
