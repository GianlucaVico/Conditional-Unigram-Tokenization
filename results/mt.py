#%% [markdown]

#  # Machine Translation

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import pandas as pd
import json
import os
import seaborn as sns
import seaborn.objects as so
sns.set_theme("paper", "dark", "colorblind")


# %%
score_template = "../models/mt/{l1}-{l2}/{vocab}/{size}/align/seed_{seed}/model.npz.scores"
comet_template = "../models/mt/{l1}-{l2}/{vocab}/{size}/align/seed_{seed}/model.npz.comet"
baseline_score_template = "../models/mt/{l1}-{l2}/{vocab}/{size}/baseline/seed_{seed}/model.npz.scores"
baseline_comet_template = "../models/mt/{l1}-{l2}/{vocab}/{size}/baseline/seed_{seed}/model.npz.comet"


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
    ("fra", "ita", SIZES[-1]),
    ("ces", "ukr", SIZES[-1]),
    ("ita", "mlt", 100_000),
    ("deu", "hsb", 60_000)
]

FULL_NAME = {
    "fra": "French",
    "ita": "Italian",
    "ces": "Czech",
    "ukr": "Ukrainian",
    "mlt": "Maltese",
    "deu": "German",
    "hsb": "Upper Sorbian"
}

SEEDS = [3,5,42]

ALIGN_NAME = {
    "align": "Word aligned",
    "phrases": "Phrase extraction"
}

def map_numbers(x):
    return [f"{i/1000:.1f}k" for i in x]

#%%

def read_experiment(path_scores: str, path_comet: str) -> dict[str, float]:
    try:
        data = {}
        with open(path_scores, "r") as f:
            scores = json.load(f)
        for score in scores:
            data[score["name"]] = float(score["score"])
        with open(path_comet, "r") as f:
            line = f.readline()
            data["comet"] = float(line.split()[-1])
    except FileNotFoundError:
        print(f"File not found: {path_scores} or {path_comet}")
        data = {
            "comet": 0,
            "BLEU": 0,
            "TER": 0,
            "chrF2++": 0,
        }
    return data

def load(score_template: str, comet_template: str, label: str) -> pd.DataFrame: 
    # pair, size, vocab, alignment, length, renyi, vocab-overlap, single-char-macro, single-char-micro, vocab-usage, fertility
    rows = []
    cols = ["pair", "vocab", "model", "BLEU", "chrF2++", "TER", "comet"]
    for l1, l2, size in TOKENIZER_PAIRS:
        for vocab in VOCABS:
            for seed in SEEDS:
                file_scores = score_template.format(l1=l1, l2=l2, vocab=vocab, size=size, seed=seed)
                file_comet = comet_template.format(l1=l1, l2=l2, vocab=vocab, size=size, seed=seed)
                data = read_experiment(file_scores, file_comet)
                data["pair"] = f"{l1}→{l2}"
                data["vocab"] = vocab
                data["model"] = label
                rows.append([data[i] for i in cols])

                file_scores = score_template.format(l1=l2, l2=l1, vocab=vocab, size=size, seed=seed)
                file_comet = comet_template.format(l1=l2, l2=l1, vocab=vocab, size=size, seed=seed)
                data = read_experiment(file_scores, file_comet)
                data["pair"] = f"{l2}→{l1}"
                data["vocab"] = vocab
                data["model"] = label
                rows.append([data[i] for i in cols])
    df = pd.DataFrame(rows, columns=cols)
    # g = df.groupby(["pair", "vocab"], as_index=False)
    # return g.aggregate("mean"), g.aggregate("std")
    return df

def plot_series(psp, baseline, metric):
    for lang1, lang2, size in TOKENIZER_PAIRS:
        for l1, l2 in [(lang1, lang2), (lang2, lang1)]:   # Directions
            data=psp[psp["pair"]==f"{l1}-{l2}"]
            data_baseline=baseline[baseline["pair"]==f"{l1}-{l2}"]

            fig = plt.figure(figsize=(10, 8))
            fig.suptitle(f"{FULL_NAME[l1]} → {FULL_NAME[l2]} {metric.capitalize()}")
            ax = sns.lineplot(data, x="vocab", y=metric, markers=True, style="label")
            ax = sns.lineplot(data_baseline, x="vocab", y=metric, markers=True, ax=ax, style="label")
            # ax.legend(["PairedSP", "Baseline"])       
            ax.set_xticks(VOCABS, map_numbers(VOCABS))

            fig.tight_layout()
            os.makedirs("mt", exist_ok=True)
            fig.savefig(f"mt/{l1}-{l2}_{metric}.pdf", bbox_inches='tight')

def plot_scatter_pair(psp, baseline, metrics, title=None):
   
    data = pd.concat([psp, baseline], axis=0)
    data["vocab size"] = pd.Categorical(data["vocab"], VOCABS)
    
    p = so.Plot(data, x="pair", color="label", fill="label", marker="vocab size", group="vocab size")  
    p = p.pair(y=metrics) 
    p = p.label(
        x="Language Pair", 
        y0=metrics[0] if title is None else title[0], 
        y1=metrics[1] if title is None else title[1], 
        color=str.capitalize, 
        marker=str.capitalize
    )
    p = p.layout(extent=[0, 0, 1, 2])       
    p = p.add(so.Dot(edgecolor="w", alpha=0.8), so.Dodge(by=["group"]))    
    p.show()
    p.save(f"mt/{"_".join(metrics)}.png", bbox_inches='tight', dpi=300)
    p.save(f"mt/{"_".join(metrics)}.pdf", bbox_inches='tight', dpi=300)

def plot_scatter(psp, baseline, metric, title=None):
   
    data = pd.concat([baseline, psp], axis=0)
    data["vocab size"] = pd.Categorical(data["vocab"], VOCABS)
    
    p = so.Plot(
        data, 
        x="pair", 
        y=metric, 
        marker="model",
        color="vocab size", 
        fill="model", 
        group="vocab size"
    )  
    p = p.layout(size=(8, 3))
    p = p.label(
        x="Language Pair", 
        y=title if title is not None else metric, 
        color=str.capitalize, 
        marker=str.capitalize
    )       
    p = p.add(so.Dot(edgecolor="w", alpha=0.8), so.Dodge(by=["group"])) 
    plotter = p.plot()   
    for ax in plotter._figure.get_axes():
        ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator([0.5,1.5,2.5,3.5,4.5,5.5,6.5]))
        ax.grid(True, which="minor", axis="x")
        ax.grid(True, which="major", axis="y")
    plotter.show()    
    plotter.save(f"mt/{metric}.png", bbox_inches='tight', dpi=300)
    plotter.save(f"mt/{metric}.pdf", bbox_inches='tight', dpi=300)
    plotter

# %% [markdown]
#  ## Scatter plots

plot_scatter(
    load(score_template, comet_template, "PairedSP"),
    load(baseline_score_template, baseline_comet_template, "Baseline"),
    "chrF2++",
    "chrF++ (↑)"
)


# %%
table_template = r"""
\begin{table}[t]
    \centering
    \footnotesize
    \setlength{\tabcolsep}{3pt}
    \caption{\TODO §title§}
    \label{tab:§label§}
    \begin{tabular}{l ccc ccc}
        \toprule
        \multicolumn{7}{c}{\textbf{§title§}} \\
        \midrule
        Model & 8k & 16k & 32k & 8k & 16k & 32k \\
        \midrule
        & \multicolumn{3}{c}{\textbf{Fra $\rightarrow$ Ita}} & \multicolumn{3}{c}{\textbf{Ita $\rightarrow$ Fra}} \\ 
            \cmidrule(lr){2-4} \cmidrule(lr){5-7}
        \spsrc + \psp & §§§ & §§§ & §§§ &
            §§§ & §§§ & §§§ \\
        \spsrc + \sptgt & §§§ & §§§ & §§§ &
            §§§ & §§§ & §§§ \\
        \midrule
        & \multicolumn{3}{c}{\textbf{Ces $\rightarrow$ Ukr}} &
            \multicolumn{3}{c}{\textbf{Ukr $\rightarrow$ Ces}} \\ \cmidrule(lr){2-4} \cmidrule(lr){5-7}
        \spsrc + \psp & §§§ & §§§ & §§§ &
            §§§ & §§§ & §§§ \\
        \spsrc + \sptgt & §§§ & §§§ & §§§ &
            §§§ & §§§ & §§§ \\
        \midrule
        & \multicolumn{3}{c}{\textbf{Ita $\rightarrow$ Mlt}} &
            \multicolumn{3}{c}{\textbf{Mlt $\rightarrow$ Ita}} \\ \cmidrule(lr){2-4} \cmidrule(lr){5-7}
        \spsrc + \psp & §§§ & §§§ & §§§ &
            §§§ & §§§ & §§§ \\
        \spsrc + \sptgt & §§§ & §§§ & §§§ &
            §§§ & §§§ & §§§ \\
        \midrule
        & \multicolumn{3}{c}{\textbf{Deu $\rightarrow$ Hsb}} &
            \multicolumn{3}{c}{\textbf{Hsb $\rightarrow$ Deu}} \\ \cmidrule(lr){2-4} \cmidrule(lr){5-7}
        \spsrc + \psp & §§§ & §§§ & §§§ &
            §§§ & §§§ & §§§ \\
        \spsrc + \sptgt & §§§ & §§§ & §§§ &
            §§§ & §§§ & §§§ \\
        \bottomrule
    \end{tabular}

\end{table}
"""
def fill_table(template, psp, baseline, metric, title):
    data = pd.concat([psp, baseline], axis=0)
    
    langs = [
        ("fra→ita", "ita→fra"), 
        ("ces→ukr", "ukr→ces"), 
        ("ita→mlt", "mlt→ita"), 
        ("deu→hsb", "hsb→deu")
    ]
    models = ["PairedSP", "Baseline"]

    for pairs in langs:
        for m in models:            
            for pair in pairs:
                for v in VOCABS:
                    mask = (data["pair"] == pair) & (data["model"] == m) & (data["vocab"] == v)
                    value = data[mask][metric].mean()
                    template = template.replace(f"§§§", f"{value:.3f}", 1)
    
    template = template.replace("§label§", metric)
    template = template.replace("§title§", title)
    with open(f"mt/{metric}.tex", "w") as f:
        f.write(template)

# %%
fill_table(
    table_template,
    load(score_template, comet_template, "PairedSP"),
    load(baseline_score_template, baseline_comet_template, "Baseline"),
    "chrF2++",
    "chrF++"
)

fill_table(
    table_template,
    load(score_template, comet_template, "PairedSP"),
    load(baseline_score_template, baseline_comet_template, "Baseline"),
    "BLEU",
    "BLEU (↑)",
)

fill_table(
    table_template,
    load(score_template, comet_template, "PairedSP"),
    load(baseline_score_template, baseline_comet_template, "Baseline"),
    "TER",
    "TER (↓)",
)
# %%
fill_table(
    table_template,
    load(score_template, comet_template, "PairedSP"),
    load(baseline_score_template, baseline_comet_template, "Baseline"),
    "comet",
    "COMET (↑)",
)

# %%
