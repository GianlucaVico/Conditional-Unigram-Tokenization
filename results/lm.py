# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
import seaborn as sns
import seaborn.objects as so
sns.set_theme("paper", "dark", "colorblind")

# %% 
score_template = "../models/lm/{l1}-{l2}/{vocab}/{size}/align/seed_{seed}/all_results.json"
baseline_template = "../models/lm/{l1}-{l2}/{vocab}/{size}/baseline/seed_{seed}/all_results.json"

BASE_COLS = ["pair", "vocab", "model"]
COLS = ["eval_bpb", "eval_bpc", "eval_loss", "eval_ppl", "eval_ppl_bytes", "eval_ppl_chars", "eval_ppl_token", "eval_ppl_words"]
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

# %%

def read_experiment(path_scores: str) -> dict[str, float]:
    try:
        with open(path_scores, "r") as f:
            scores = json.load(f)
        data = scores["test"]            
    except FileNotFoundError:
        print(f"File not found: {path_scores}")
        data = {}
    return data

def load(score_template: str, label: str, metrics: list[str] = COLS ) -> pd.DataFrame: 
    # pair, size, vocab, alignment, length, renyi, vocab-overlap, single-char-macro, single-char-micro, vocab-usage, fertility
    rows = []
    cols =  BASE_COLS + metrics
    for l1, l2, size in TOKENIZER_PAIRS:
        for vocab in VOCABS:
            for seed in SEEDS:
                file_scores = score_template.format(l1=l1, l2=l2, vocab=vocab, size=size, seed=seed)
                data = read_experiment(file_scores)
                if len(data) != 0:
                    data["pair"] = f"({l1}→){l2}"
                    data["vocab"] = vocab
                    data["model"] = label
                    rows.append([data.get(i) for i in cols])

                file_scores = score_template.format(l1=l2, l2=l1, vocab=vocab, size=size, seed=seed)
                data = read_experiment(file_scores)
                if len(data) != 0:
                    data["pair"] = f"({l2}→){l1}"
                    data["vocab"] = vocab
                    data["model"] = label
                    rows.append([data.get(i) for i in cols])
    df = pd.DataFrame(rows, columns=cols)
    return df

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
        y=title, 
        marker=str.capitalize, 
        color=str.capitalize)       
    p = p.add(so.Dot(edgecolor="w", alpha=0.8), so.Dodge(by=["group"]))    
    plotter = p.plot()   
    for ax in plotter._figure.get_axes():
        ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator([0.5,1.5,2.5,3.5,4.5,5.5,6.5]))
        ax.grid(True, which="minor", axis="x")
        ax.grid(True, which="major", axis="y")
    # plotter._figure.get_axes()[-1].tick_params(axis="x", labelrotation=45)    
    plotter.save(f"lm_bi/{metric}.png", bbox_inches='tight', dpi=300)
    plotter.save(f"lm_bi/{metric}.pdf", bbox_inches='tight', dpi=300)
    plotter.show()
    return plotter
    

# %% [markdown]
# ## PLot PPL Byte

plotter = plot_scatter(
    load(score_template, r"$\mathregular{PairedSP_{M}}$"),
    load(baseline_template, "Baseline"),
    "eval_ppl_bytes",
    "Perplexity per Byte (↓)"
)
plotter
# %%

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
        & \multicolumn{3}{c}{\textbf{(Fra $\rightarrow$) Ita}} & \multicolumn{3}{c}{\textbf{(Ita $\rightarrow$) Fra}} \\ 
            \cmidrule(lr){2-4} \cmidrule(lr){5-7}
        \psp & §§§ & §§§ & §§§ &
            §§§ & §§§ & §§§ \\
        \sptgt & §§§ & §§§ & §§§ &
            §§§ & §§§ & §§§ \\
        \midrule
        & \multicolumn{3}{c}{\textbf{(Ces $\rightarrow$) Ukr}} &
            \multicolumn{3}{c}{\textbf{(Ukr $\rightarrow$) Ces}} \\ \cmidrule(lr){2-4} \cmidrule(lr){5-7}
        \psp & §§§ & §§§ & §§§ &
            §§§ & §§§ & §§§ \\
        \sptgt & §§§ & §§§ & §§§ &
            §§§ & §§§ & §§§ \\
        \midrule
        & \multicolumn{3}{c}{\textbf{(Ita $\rightarrow$) Mlt}} &
            \multicolumn{3}{c}{\textbf{(Mlt $\rightarrow$) Ita}} \\ \cmidrule(lr){2-4} \cmidrule(lr){5-7}
        \psp & §§§ & §§§ & §§§ &
            §§§ & §§§ & §§§ \\
        \sptgt & §§§ & §§§ & §§§ &
            §§§ & §§§ & §§§ \\
        \midrule
        & \multicolumn{3}{c}{\textbf{(Deu $\rightarrow$) Hsb}} &
            \multicolumn{3}{c}{\textbf{(Hsb $\rightarrow$) Deu}} \\ \cmidrule(lr){2-4} \cmidrule(lr){5-7}
        \psp & §§§ & §§§ & §§§ &
            §§§ & §§§ & §§§ \\
        \sptgt & §§§ & §§§ & §§§ &
            §§§ & §§§ & §§§ \\
        \bottomrule
    \end{tabular}
\end{table}
"""

def fill_table(template, psp, baseline, metric, title):
    data = pd.concat([psp, baseline], axis=0)
    
    langs = [
        ("(fra→)ita", "(ita→)fra"), 
        ("(ces→)ukr", "(ukr→)ces"), 
        ("(ita→)mlt", "(mlt→)ita"), 
        ("(deu→)hsb", "(hsb→)deu")
    ]
    models = ["PairedSP", "Baseline"]
    label = title if title is not None else metric

    for pairs in langs:
        for m in models:            
            for pair in pairs:
                for v in VOCABS:
                    mask = (data["pair"] == pair) & (data["model"] == m) & (data["vocab"] == v)
                    value = data[mask][metric].mean()
                    template = template.replace(f"§§§", f"{value:.3f}", 1)
    
    template = template.replace("§label§", metric)
    template = template.replace("§title§", title)
    with open(f"lm/{metric}.tex", "w") as f:
        f.write(template)

fill_table(
    table_template,
    load(score_template, "PairedSP"),
    load(baseline_template, "Baseline"),
    "eval_ppl_bytes",
    r"Perplexity per Byte ($\downarrow$)"
)
# %%
