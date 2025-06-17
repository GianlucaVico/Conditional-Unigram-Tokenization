rule lm:
    input:
        train="data/{l1}-{l2}/psp/{vocab}/{size}/{alignment}/train.{l2}.tok.unpaired",
        valid="data/{l1}-{l2}/psp/{vocab}/{size}/{alignment}/flores.dev.{l2}.tok.unpaired",
        test="data/{l1}-{l2}/psp/{vocab}/{size}/{alignment}/flores.devtest.{l2}.tok.unpaired",
        vocab="models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.yml",
        vocab2="models/{l2}-{l1}/{vocab}/{size}/unigram.{l1}.yml",
        train2="data/{l2}-{l1}/unigram/{vocab}/{size}/train.{l1}.tok",
    output:
        model="models/lm/{l1}-{l2}/{vocab}/{size}/{alignment}/seed_{seed}/model.safetensors",
        results="models/lm/{l1}-{l2}/{vocab}/{size}/{alignment}/seed_{seed}/all_results.json",
    log:
        "models/lm/{l1}-{l2}/{vocab}/{size}/{alignment}/seed_{seed}/model.log"
    resources:
        mem_mb=40_000,
        cpus_per_task=1,
        slurm_partition="",
        slurm_extra="--gres=gpu:1",
        tasks=1,        
    params:
        steps="2000000",
        batch="64",
        output_dir=subpath(output.model, parent=True),
    shell:
        """
        /usr/bin/time -v ./lm/train.py --train {input.train} --train-other {input.train2} --valid {input.valid} --test {input.test} --vocab {input.vocab} --additional-vocab {input.vocab2} --output-dir {params.output_dir} --steps {params.steps} --batch {params.batch} --seed {wildcards.seed}
        rm -rd {params.output_dir}/checkpoint-*
        """

use rule lm as lm_baseline with:
    input:
        train="data/{l1}-{l2}/unigram/{vocab}/{size}/train.{l2}.tok",
        valid="data/{l1}-{l2}/unigram/{vocab}/{size}/flores.dev.{l2}.tok",
        test="data/{l1}-{l2}/unigram/{vocab}/{size}/flores.devtest.{l2}.tok",
        vocab="models/{l1}-{l2}/{vocab}/{size}/unigram.{l2}.yml",
        vocab2="models/{l2}-{l1}/{vocab}/{size}/unigram.{l1}.yml",
        train2="data/{l2}-{l1}/unigram/{vocab}/{size}/train.{l1}.tok",
    output:
        model="models/lm/{l1}-{l2}/{vocab}/{size}/baseline/seed_{seed}/model.safetensors",
        results="models/lm/{l1}-{l2}/{vocab}/{size}/baseline/seed_{seed}/all_results.json",
    log:
        "models/lm/{l1}-{l2}/{vocab}/{size}/baseline/seed_{seed}/model.log"


rule usp_tokenize_parts:
    input:
        input_trg_folder="data/{l1}-{l2}/train.{l2}_parts/",      
        model="models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.tar.gz"
    output:
        output_folder=directory("data/{l1}-{l2}/psp/{vocab}/{size}/{alignment}/train.{l2}_parts_unpaired/"),
        _output_folder=directory("data/{l2}-{l1}/psp/{vocab}/{size}/{alignment}/train.{l2}_parts_unpaired/")
    log:
        temp("data/{l1}-{l2}/psp/{vocab}/{size}/{alignment}/train.{l2}.unpaired.log")
    resources:
        mem_mb=1_000,
        cpus_per_task=1,
        tasks=1
    params:
        partitions=PARTITIONS,
    shell:
        """
        mkdir -p {output.output_folder}
        sbatch --wait -J '{wildcards.l2}_{wildcards.vocab}_{wildcards.size}' -o {log} -a 0-$(({params.partitions}-1)) -c 1 --mem=4000 -q low ./mt/tokenize_usp.sh {input.model} {input.input_trg_folder} {output.output_folder}
        """

rule merge_usp_parts:
    input:
        l1="data/{l2}-{l1}/psp/{vocab}/{size}/{alignment}/train.{l1}_parts_unpaired",
        l2="data/{l1}-{l2}/psp/{vocab}/{size}/{alignment}/train.{l2}_parts_unpaired",
    output:
        l1="data/{l2}-{l1}/psp/{vocab}/{size}/{alignment}/train.{l1}.tok.unpaired",
        l2="data/{l1}-{l2}/psp/{vocab}/{size}/{alignment}/train.{l2}.tok.unpaired",
    resources:
        mem_mb=5_000,
        cpus_per_task=1,
        tasks=1
    shell:
        """
        cat {input.l1}/*.tok.unpaired > {output.l1}
        cat {input.l2}/*.tok.unpaired > {output.l2}
        """
