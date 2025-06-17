# Download data
ruleorder: download_multiparacrawl > download_full_multiparacrawl > download_nllb > download_full_nllb

rule download_full_nllb:
    output:
        protected("data/{l1}-{l2}/train.{l1}"),
        protected("data/{l1}-{l2}/train.{l2}"),
        protected("data/{l2}-{l1}/train.{l1}"),
        protected("data/{l2}-{l1}/train.{l2}"),
    resources:
        mem_mb=4_000,
        cpus_per_task=1,
        tasks=1,
    shell:
        "python3 downloader/download_nllb.py --langs {wildcards.l1} {wildcards.l2}"

rule download_full_multiparacrawl:
    output:
        protected("data/{l1}-{l2}/train.{l1}"),
        protected("data/{l1}-{l2}/train.{l2}"),
        protected("data/{l2}-{l1}/train.{l1}"),
        protected("data/{l2}-{l1}/train.{l2}"),
    wildcard_constraints:
        l2="mlt"
    resources:
        mem_mb=4_000,
        cpus_per_task=1,
        tasks=1,
    shell:
        "python3 downloader/download_multiparacrawl.py --langs {wildcards.l1} {wildcards.l2}"
    
rule download_flores:
    output:
        protected("data/{l1}-{l2}/flores.dev.{l1}"),
        protected("data/{l1}-{l2}/flores.dev.{l2}"),
        protected("data/{l1}-{l2}/flores.devtest.{l1}"),
        protected("data/{l1}-{l2}/flores.devtest.{l2}"),
    resources:
        mem_mb=4_000,
        cpus_per_task=1,
        tasks=1,
    shell:
        "python3 downloader/download_flores.py --langs {wildcards.l1} {wildcards.l2}"

rule download_nllb:
    output:
        protected("data/{l1}-{l2}/train.{size}.{l1}"),
        protected("data/{l1}-{l2}/train.{size}.{l2}"),
        protected("data/{l2}-{l1}/train.{size}.{l1}"),
        protected("data/{l2}-{l1}/train.{size}.{l2}"),    
    params:
        seed=42
    resources:
        mem_mb=4_000,
        cpus_per_task=1,
        tasks=1,
    shell:
        "python3 downloader/download_nllb.py --langs {wildcards.l1} {wildcards.l2} --size {wildcards.size} --seed {params.seed}"

rule download_multiparacrawl:
    output:
        protected("data/{l1}-{l2}/train.{size}.{l1}"),
        protected("data/{l1}-{l2}/train.{size}.{l2}"),
        protected("data/{l2}-{l1}/train.{size}.{l1}"),
        protected("data/{l2}-{l1}/train.{size}.{l2}"),
    wildcard_constraints:        
        l2="mlt"
    params:
        seed=42
    resources:
        mem_mb=4_000,
        cpus_per_task=1,
        tasks=1,
    shell:
        "python3 downloader/download_multiparacrawl.py --langs {wildcards.l1} {wildcards.l2} --size {wildcards.size} --seed {params.seed}"