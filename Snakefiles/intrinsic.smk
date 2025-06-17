rule intrinsic_score:
    input:
        tok="data/{l1}-{l2}/psp/{vocab}/{size}/{alignment}/flores.devtest.{l2}.tok",
        sp_vocab="models/{l1}-{l2}/{vocab}/{size}/unigram.{l2}.yml",
        psp_vocab="models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.yml",
    output:        
        "models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.scores"
    resources:
        mem_mb=1_000,
        cpus_per_task=1,
        tasks=1
    shell:
        "python -m intrinsic --trg {input.tok} -p 3 "
            "--src-vocab {input.sp_vocab} "
            "--trg-vocab {input.psp_vocab} "
            " > {output}"

use rule intrinsic_score as unpaired_score with:
    input:
        tok="data/{l1}-{l2}/psp/{vocab}/{size}/{alignment}/flores.devtest.{l2}.tok.unpaired",
        sp_vocab="models/{l1}-{l2}/{vocab}/{size}/unigram.{l2}.yml",
        psp_vocab="models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.yml",
    output:
        "models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.scores.unpaired"

use rule intrinsic_score as hard_score with:
    input:
        tok="data/{l1}-{l2}/psp/{vocab}/{size}/{alignment}/flores.devtest.{l2}.hard.tok",
        sp_vocab="models/{l1}-{l2}/{vocab}/{size}/unigram.{l2}.yml",
        psp_vocab="models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.hard.yml",
    output:
        "models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.scores.hard"
    

use rule intrinsic_score as baseline_score with:
    input:
        tok="data/{l1}-{l2}/{sp_type}/{vocab}/{size}/flores.devtest.{l2}.tok",
        sp_vocab="models/{l1}-{l2}/{vocab}/{size}/{sp_type}.{l2}.yml",
        psp_vocab="models/{l1}-{l2}/{vocab}/{size}/{sp_type}.{l2}.yml",
    output:
        "models/{l1}-{l2}/{vocab}/{size}/{sp_type}.{l2}.scores",

rule eflomal_score:
    input:
        src_train="data/{l1}-{l2}/{sp_type}/{vocab}/{size}/flores.dev.{l1}.tok",
        trg_sp_train="data/{l1}-{l2}/{sp_type}/{vocab}/{size}/flores.dev.{l2}.tok",
        trg_psp_train="data/{l1}-{l2}/psp/{vocab}/{size}/{alignment}/flores.dev.{l2}.tok",
        src="data/{l1}-{l2}/{sp_type}/{vocab}/{size}/flores.devtest.{l1}.tok",
        trg_sp="data/{l1}-{l2}/{sp_type}/{vocab}/{size}/flores.devtest.{l2}.tok",
        trg_psp="data/{l1}-{l2}/psp/{vocab}/{size}/{alignment}/flores.devtest.{l2}.tok",
        src_original_train="data/{l2}-{l1}/{sp_type}/{vocab}/{size}/train.{l1}.truncated.tok",
        trg_original_sp_train="data/{l1}-{l2}/{sp_type}/{vocab}/{size}/train.{l2}.truncated.tok",
        trg_original_psp_train="data/{l1}-{l2}/psp/{vocab}/{size}/{alignment}/train.{l2}.truncated.tok"
    output:
        scores_psp="models/{l1}-{l2}/{vocab}/{size}/{alignment}/{alignment}_{sp_type}.{l2}.scores.eflomal2",
        scores_sp="models/{l1}-{l2}/{vocab}/{size}/{alignment}_{sp_type}.{l2}.scores.eflomal2",
    resources:
        mem_mb=35_000,
        cpus_per_task=1,
        tasks=1
    shell:
        "python -m intrinsic.eflomal_eval "
            "--src-train {input.src_train} "
            "--trg-sp-train {input.trg_sp_train} "
            "--trg-psp-train {input.trg_psp_train} "
            "--src {input.src} "
            "--trg-sp {input.trg_sp} "
            "--trg-psp {input.trg_psp} "
            "--src-vocab-size {wildcards.vocab} "
            "--trg-vocab-size {wildcards.vocab} "
            "--output-sp {output.scores_sp} "
            "--output-psp {output.scores_psp} "
            "--src-original-train {input.src_original_train} "
            "--trg-original-sp-train {input.trg_original_sp_train} "
            "--trg-original-psp-train {input.trg_original_psp_train} "

rule psp_tokenize_train:
    input:
        model="models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.tar.gz",
        trg_dir=directory("data/{l1}-{l2}/train.{l2}_parts/"),
        # trg_train="data/{l1}-{l2}/train.{l2}_parts/00",
        src_tok="data/{l1}-{l2}/unigram/{vocab}/{size}/train.{l2}.truncated.tok"
    output:
        "data/{l1}-{l2}/psp/{vocab}/{size}/{alignment}/train.{l2}.truncated.tok"
    resources:
        mem_mb=5_000,
        cpus_per_task=1,
        tasks=1
    params:
        lines=1000,
        trg_train="data/{l1}-{l2}/train.{l2}_parts/00",
        truncated="data/{l1}-{l2}/psp/{vocab}/{size}/{alignment}/train.{l2}.truncated"
    shell:
        """
        head -n {params.lines} {params.trg_train} > {params.truncated}
        /usr/bin/time -v python -m paired_sp tokenize -m cond-exact --model-file {input.model} --src {input.src_tok} --trg {params.truncated} -o {output}
        """

rule sp_tokenize_train:
    input:
        # train="data/{l1}-{l2}/train.{l2}_parts/00",
        trg_dir=directory("data/{l1}-{l2}/train.{l2}_parts"),
        model="models/{l1}-{l2}/{vocab}/{size}/unigram.{l2}.model",
    output:
        "data/{l1}-{l2}/unigram/{vocab}/{size}/train.{l2}.truncated.tok"
    resources:
        mem_mb=5_000,
        cpus_per_task=1,
        tasks=1
    params:
        lines=1000,
        train="data/{l1}-{l2}/train.{l2}_parts/00",
        truncated="data/{l1}-{l2}/unigram/{vocab}/{size}/train.{l2}.truncated"
    shell:
        """
        head -n {params.lines} {params.train} > {params.truncated}
        python sp_tools/sp_tokenize.py -i {params.truncated} -o {output} -m {input.model}
        """
