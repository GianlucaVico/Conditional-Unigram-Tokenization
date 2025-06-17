rule prepare_align:
    input:
        l1="data/{l1}-{l2}/{name}.{l1}",
        l2="data/{l1}-{l2}/{name}.{l2}",
        _l1="data/{l2}-{l1}/{name}.{l1}",
        _l2="data/{l2}-{l1}/{name}.{l2}",
    output:
        l1="data/{l1}-{l2}/align/{name}.{l1}",
        l2="data/{l1}-{l2}/align/{name}.{l2}",
        _l1="data/{l2}-{l1}/align/{name}.{l1}",
        _l2="data/{l2}-{l1}/align/{name}.{l2}",
    params:
        folder="data/{l1}-{l2}/align",
        prefix="data/{l1}-{l2}/align/{name}",
    wildcard_constraints:
        name=r"train(\.[0-9]+)?"
    resources:
        mem_mb=4_000,
        cpus_per_task=1,
        tasks=1,
    shell:
        """
        mkdir -p {params.folder}
        python -m pretokenize --src-input {input.l1} --trg-input {input.l2} --prefix {params.prefix} --src-output {output.l1} --trg-output {output.l2} --type align
        """

rule prepare_phrases:
    input:
        l1="data/{l1}-{l2}/{name}.{l1}",
        l2="data/{l1}-{l2}/{name}.{l2}",
        _l1="data/{l2}-{l1}/{name}.{l1}",
        _l2="data/{l2}-{l1}/{name}.{l2}",
    output:
        l1="data/{l1}-{l2}/phrases/{name}.{l1}",
        l2="data/{l1}-{l2}/phrases/{name}.{l2}",
        _l1="data/{l2}-{l1}/phrases/{name}.{l1}",
        _l2="data/{l2}-{l1}/phrases/{name}.{l2}",        
    params:
        folder="data/{l1}-{l2}/phrases",
        prefix="data/{l1}-{l2}/phrases/{name}",
    wildcard_constraints:
        name=r"train(\.[0-9]+)?"
    resources:
        mem_mb=4_000,
        cpus_per_task=1,
        tasks=1,
    shell:
        """
        mkdir -p {params.folder}
        python -m pretokenize --src-input {input.l1} --trg-input {input.l2} --prefix {params.prefix} --src-output {output.l1} --trg-output {output.l2} --type phrases
        """
