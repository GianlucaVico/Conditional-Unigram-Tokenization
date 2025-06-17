rule train_sp:
    input:
        data="data/{l1}-{l2}/train.{size}.{lang}",
    output:
        model=protected("models/{l1}-{l2}/{vocab}/{size}/{sp_type}.{lang}.model"),
        vocab=protected("models/{l1}-{l2}/{vocab}/{size}/{sp_type}.{lang}.vocab"),
    params:
        base_name="models/{l1}-{l2}/{vocab}/{size}/{sp_type}.{lang}"
    resources:
        mem_mb=4_000,
        cpus_per_task=1,
        tasks=1,
    log:
        temp("models/{l1}-{l2}/{vocab}/{size}/{sp_type}.{lang}.log")
    shell:
        "python sp_tools/sp_train.py -i {input.data} -o {params.base_name} -v {wildcards.vocab} --type {wildcards.sp_type} &> {log}"

rule tokenize_sp:
    input:
        data="data/{l1}-{l2}/{name}.{lang}",
        model="models/{l1}-{l2}/{vocab}/{size}/{sp_type}.{lang}.model",
    output:
        "data/{l1}-{l2}/{sp_type}/{vocab}/{size}/{name}.{lang}.tok",
    resources:
        mem_mb=4_000,
        cpus_per_task=1,
        tasks=1,
    shell:
        "python sp_tools/sp_tokenize.py -i {input.data} -o {output} -m {input.model}"

rule vocab_sp:
    input:
        "models/{l1}-{l2}/{vocab}/{size}/{sp_type}.{lang}.model",
    output:
        "models/{l1}-{l2}/{vocab}/{size}/{sp_type}.{lang}.yml",
    resources:
        mem_mb=1_000,
        cpus_per_task=1,
        tasks=1,
    shell:
        "python sp_tools/sp_vocab.py -m {input} -o {output}"
    