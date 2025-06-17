def mt_memory(wildcards):
    if wildcards.l1 == "mlt" or wildcards.l2 == "mlt":
        return 10_000
    if wildcards.l1 == "ces" or wildcards.l2 == "ces":
        return 50_000
    if wildcards.l1 == "fra" or wildcards.l2 == "fra":
        return 70_000
    return 50_000

PARTITIONS="20"

rule train_mt:
    input:
        src="data/{l1}-{l2}/unigram/{vocab}/{size}/train.{l1}.tok",
        trg="data/{l1}-{l2}/psp/{vocab}/{size}/{alignment}/train.{l2}.tok",
        src_valid="data/{l1}-{l2}/unigram/{vocab}/{size}/flores.dev.{l1}.tok",
        trg_valid="data/{l1}-{l2}/psp/{vocab}/{size}/{alignment}/flores.dev.{l2}.tok",
        src_vocab="models/{l1}-{l2}/{vocab}/{size}/unigram.{l1}.yml",
        trg_vocab="models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.yml",
    output:
        "models/mt/{l1}-{l2}/{vocab}/{size}/{alignment}/seed_{seed}/model.npz"
    log:
        "models/mt/{l1}-{l2}/{vocab}/{size}/{alignment}/seed_{seed}/model.npz.log"
    resources:
        mem_mb=mt_memory,
        cpus_per_task=1,
        slurm_partition="",
        slurm_extra="--gres=gpu:1",
        tasks=1,        
    params:
        config="mt/transformer_base.yml",        
    shell:
        """
        ./mt/marian_train.sh {params.config} {output} {input.src} {input.trg} {input.src_valid} {input.trg_valid} {input.src_vocab} {input.trg_vocab} {wildcards.seed} 0 &> {log}
        rm {output}.optimizer.npz
        """

use rule train_mt as train_mt_baseline with:
    input:
        src="data/{l1}-{l2}/unigram/{vocab}/{size}/train.{l1}.tok",
        trg="data/{l1}-{l2}/unigram/{vocab}/{size}/train.{l2}.tok",
        src_valid="data/{l1}-{l2}/unigram/{vocab}/{size}/flores.dev.{l1}.tok",
        trg_valid="data/{l1}-{l2}/unigram/{vocab}/{size}/flores.dev.{l2}.tok",
        src_vocab="models/{l1}-{l2}/{vocab}/{size}/unigram.{l1}.yml",
        trg_vocab="models/{l1}-{l2}/{vocab}/{size}/unigram.{l2}.yml",
    output:
        "models/mt/{l1}-{l2}/{vocab}/{size}/baseline/seed_{seed}/model.npz"
    log:
        "models/mt/{l1}-{l2}/{vocab}/{size}/baseline/seed_{seed}/model.npz.log"

rule test_mt:
    input:
        src_tok="data/{l1}-{l2}/unigram/{vocab}/{size}/flores.devtest.{l1}.tok",
        trg="data/{l1}-{l2}/flores.devtest.{l2}",
        src_vocab="models/{l1}-{l2}/{vocab}/{size}/unigram.{l1}.yml",
        trg_vocab="models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.yml",
        model="models/mt/{l1}-{l2}/{vocab}/{size}/{alignment}/seed_{seed}/model.npz"
    output:        
        scores="models/mt/{l1}-{l2}/{vocab}/{size}/{alignment}/seed_{seed}/model.npz.scores",
        comet="models/mt/{l1}-{l2}/{vocab}/{size}/{alignment}/seed_{seed}/model.npz.comet"
    params:
        # config="mt/transformer_base.yml",
        config="models/mt/{l1}-{l2}/{vocab}/{size}/{alignment}/seed_{seed}/model.npz.decoder.yml",
        tok="models/mt/{l1}-{l2}/{vocab}/{size}/{alignment}/seed_{seed}/model.npz.tok",
        detok="models/mt/{l1}-{l2}/{vocab}/{size}/{alignment}/seed_{seed}/model.npz.detok",
    log:
        "models/mt/{l1}-{l2}/{vocab}/{size}/{alignment}/seed_{seed}/model.npz.test.log"
    resources:
        mem_mb=20_000,
        cpus_per_task=1,
        slurm_partition="",
        slurm_extra="--gres=gpu:1",
        tasks=1,
    shell:
        "./mt/marian_test.sh {params.config} {input.model} {input.src_tok} {input.trg} "
            "{input.src_vocab} {input.trg_vocab} {params.tok} {params.detok} "
            "{output.scores} {output.comet} 0 &> {log}"

use rule test_mt as test_mt_baseline with:
    input:
        src_tok="data/{l1}-{l2}/unigram/{vocab}/{size}/flores.devtest.{l1}.tok",
        trg="data/{l1}-{l2}/flores.devtest.{l2}",
        src_vocab="models/{l1}-{l2}/{vocab}/{size}/unigram.{l1}.yml",
        trg_vocab="models/{l1}-{l2}/{vocab}/{size}/unigram.{l2}.yml",
        model="models/mt/{l1}-{l2}/{vocab}/{size}/baseline/seed_{seed}/model.npz"
    output:        
        scores="models/mt/{l1}-{l2}/{vocab}/{size}/baseline/seed_{seed}/model.npz.scores",
        comet="models/mt/{l1}-{l2}/{vocab}/{size}/baseline/seed_{seed}/model.npz.comet"
    params:
        # config="mt/transformer_base.yml",
        config="models/mt/{l1}-{l2}/{vocab}/{size}/baseline/seed_{seed}/model.npz.decoder.yml",
        tok="models/mt/{l1}-{l2}/{vocab}/{size}/baseline/seed_{seed}/model.npz.tok",
        detok="models/mt/{l1}-{l2}/{vocab}/{size}/baseline/seed_{seed}/model.npz.detok",
    log:
        "models/mt/{l1}-{l2}/{vocab}/{size}/baseline/seed_{seed}/model.npz.test.log"

rule split_data:
    input:
        lang="data/{l1}-{l2}/train.{lang}",
        _lang="data/{l2}-{l1}/train.{lang}"
    output:
        # touch("data/{l1}-{l2}/train.{lang}.split.flag")
        lang=directory("data/{l1}-{l2}/train.{lang}_parts/"),
        _lang=directory("data/{l2}-{l1}/train.{lang}_parts/")
    resources:
        mem_mb=20_000,
        cpus_per_task=1,
        tasks=1
    params:
        l=len(PARTITIONS),
        partitions=PARTITIONS
    shell:
        """
        mkdir -p {input.lang}_parts
        split -a {params.l} -d -n r/{params.partitions} {input.lang} {output.lang}/
        """
rule sp_tokenize_parts:
    input:
        # flag="data/{l1}-{l2}/train.{lang}.split.flag",
        input_folder="data/{l1}-{l2}/train.{l1}_parts/",
        model="models/{l1}-{l2}/{vocab}/{size}/{sp_type}.{l1}.model"
    output:
        # temp(touch("data/{l1}-{l2}/{sp_type}/{vocab}/{size}/train.{lang}.sp_tok.flag")),
        output_folder=directory("data/{l1}-{l2}/{sp_type}/{vocab}/{size}/train.{l1}_parts/"),
        _output_folder=directory("data/{l2}-{l1}/{sp_type}/{vocab}/{size}/train.{l1}_parts/")
    log:
        temp("data/{l1}-{l2}/{sp_type}/{vocab}/{size}/train.{l1}.sp_tok.log")
    resources:
        mem_mb=1_000,
        cpus_per_task=1,
        tasks=1
    params:
        partitions=PARTITIONS,
        # input_folder="data/{l1}-{l2}/train.{lang}_parts/",
        # output_folder="data/{l1}-{l2}/{sp_type}/{vocab}/{size}/train.{lang}_parts/"
    shell:
        """
        mkdir -p {output.output_folder}
        sbatch --wait -J '{wildcards.l1}_{wildcards.vocab}_{wildcards.size}' -o {log} -a 0-$(({params.partitions}-1)) -c 1 --mem=4000 -q low ./mt/tokenize_sp.sh {input.model} {input.input_folder} {output.output_folder}
        """

rule psp_tokenize_parts:
    input:
        # flag_data="data/{l1}-{l2}/train.{l2}.split.flag",
        # flag_sp="data/{l1}-{l2}/unigram/{vocab}/{size}/train.{l1}.sp_tok.flag",
        input_src_folder="data/{l1}-{l2}/unigram/{vocab}/{size}/train.{l1}_parts/",
        input_trg_folder="data/{l1}-{l2}/train.{l2}_parts/",      
        model="models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.tar.gz"
    output:
        # temp(touch("data/{l1}-{l2}/psp/{vocab}/{size}/train.{l2}.psp_tok.flag")),
        output_folder=directory("data/{l1}-{l2}/psp/{vocab}/{size}/{alignment}/train.{l2}_parts/"),
        _output_folder=directory("data/{l2}-{l1}/psp/{vocab}/{size}/{alignment}/train.{l2}_parts/")
    log:
        temp("data/{l1}-{l2}/psp/{vocab}/{size}/{alignment}/train.{l2}.log")
    resources:
        mem_mb=1_000,
        cpus_per_task=1,
        tasks=1
    params:
        partitions=PARTITIONS,
        # input_src_folder="data/{l1}-{l2}/unigram/{vocab}/{size}/train.{l1}_parts/",
        # input_trg_folder="data/{l1}-{l2}/train.{l2}_parts/",        
        # output_folder="data/{l1}-{l2}/psp/{vocab}/{size}/train.{l2}_parts/"
    shell:
        """
        mkdir -p {output.output_folder}
        sbatch --wait -J '{wildcards.l2}_{wildcards.vocab}_{wildcards.size}' -o {log} -a 0-$(({params.partitions}-1)) -c 1 --mem=16000 -q low ./mt/tokenize_psp.sh {input.model} {input.input_src_folder} {input.input_trg_folder} {output.output_folder}
        """

rule merge_psp_parts:
    input:
        l1="data/{l2}-{l1}/psp/{vocab}/{size}/{alignment}/train.{l1}_parts",
        l2="data/{l1}-{l2}/psp/{vocab}/{size}/{alignment}/train.{l2}_parts",
    output:
        l1="data/{l2}-{l1}/psp/{vocab}/{size}/{alignment}/train.{l1}.tok",
        l2="data/{l1}-{l2}/psp/{vocab}/{size}/{alignment}/train.{l2}.tok",
    resources:
        mem_mb=5_000,
        cpus_per_task=1,
        tasks=1
    shell:
        """
        cat {input.l1}/*.tok > {output.l1}
        cat {input.l2}/*.tok > {output.l2}
        """

rule merge_sp_parts:
    input:
        l1="data/{l1}-{l2}/unigram/{vocab}/{size}/train.{l1}_parts",
        _l1="data/{l2}-{l1}/unigram/{vocab}/{size}/train.{l1}_parts",
        l2="data/{l1}-{l2}/unigram/{vocab}/{size}/train.{l2}_parts",
        _l2="data/{l2}-{l1}/unigram/{vocab}/{size}/train.{l2}_parts",
    output:
        l1="data/{l1}-{l2}/unigram/{vocab}/{size}/train.{l1}.tok",
        _l1="data/{l2}-{l1}/unigram/{vocab}/{size}/train.{l1}.tok",
        l2="data/{l1}-{l2}/unigram/{vocab}/{size}/train.{l2}.tok",
        _l2="data/{l2}-{l1}/unigram/{vocab}/{size}/train.{l2}.tok",
    resources:
        mem_mb=5_000,
        cpus_per_task=1,
        tasks=1
    shell:
        """
        cat {input.l1}/*.tok > {output.l1}
        cat {input.l2}/*.tok > {output.l2}
        """
   
