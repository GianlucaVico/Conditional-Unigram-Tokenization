#%% [markdown]

#  # Intrinsic Evaluation

import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
import seaborn as sns
import seaborn.objects as so
sns.set_theme("paper", "dark", "colorblind")
so.Plot.config.theme['mathtext.default'] = 'regular'
matplotlib.rcParams['mathtext.default'] = 'regular'

# %%
score_template = "../models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.scores"
score_hard_template = "../models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.scores.hard"
score_unpaired_template = "../models/{l1}-{l2}/{vocab}/{size}/{alignment}/{l2}.scores.unpaired"
baseline_template = "../models/{l1}-{l2}/{vocab}/{size}/unigram.{l2}.scores"

eflomal_template = "../models/{l1}-{l2}/{vocab}/{size}/{alignment}/align_unigram.{l2}.scores.eflomal2" # TODO make it more general
eflomal_baseline_template = "../models/{l1}-{l2}/{vocab}/{size}/align_unigram.{l2}.scores.eflomal2"

BASE_COLS = ["pair", "size", "vocab", "alignment", "model"]
COLS = ["length", "renyi", "vocab-overlap", "single-char-macro", "single-char-micro", "vocab-usage", "fertility", "start-word"]
EFLOMAL_COLS = ["one-to-one", "eflomal", "unaligned", "ono-to-one-disjoint", "one-to-one-train", "unaligned-train", "eflomal-train"]


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

FULL_NAME = {
    "fra": "French",
    "ita": "Italian",
    "ces": "Czech",
    "ukr": "Ukrainian",
    "mlt": "Maltese",
    "deu": "German",
    "hsb": "Upper Sorbian"
}

ALIGNMENTS = [
    "align",
]

ALIGN_NAME = {
    "align": "Word aligned",
    "phrases": "Phrase extraction"
}

def map_numbers(x):
    return [f"{i/1000:.1f}k" for i in x]

def read_experiment(path: str) -> dict[str, float]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {path}")
        data = {
            "length": 0,
            "renyi": 0,
            "vocab-overlap": 0,
            "single-char-macro": 0,
            "single-char-micro": 0,
            "vocab-usage": 0,
            "fertility": 0,
            "start-word": 0,
        }
    return data

def load(template: str, metrics: list[str] = COLS, model = None) -> pd.DataFrame:    
    rows = []
    cols = BASE_COLS + metrics
    for l1, l2, sizes in TOKENIZER_PAIRS:
        for size in sizes:
            for vocab in VOCABS:
                for alignment in ALIGNMENTS:
                    data = read_experiment(template.format(l1=l1, l2=l2, vocab=vocab, size=size, alignment=alignment))
                    data["pair"] = f"{l1}→{l2}"
                    data["size"] = size
                    data["vocab"] = vocab
                    data["alignment"] = alignment
                    data["model"] = model
                    rows.append([data[i] for i in cols])
                    data = read_experiment(template.format(l1=l2, l2=l1, vocab=vocab, size=size, alignment=alignment))
                    data["pair"] = f"{l2}→{l1}"
                    data["size"] = size
                    data["vocab"] = vocab
                    data["alignment"] = alignment
                    data["model"] = model
                    rows.append([data[i] for i in cols])
    df = pd.DataFrame(rows, columns=cols)
    df = df.set_index(["pair", "alignment", "size", "vocab",])
    return df

def plot_ratio(psp, usp, hsp, baseline, metric):    
    for alignment in ALIGNMENTS:
        for lang1, lang2, sizes in TOKENIZER_PAIRS:
            for l1, l2 in [(lang1, lang2), (lang2, lang1)]:   # Directions             
                data_psp = np.zeros((len(sizes), len(VOCABS)))
                data_usp = np.zeros((len(sizes), len(VOCABS)))
                data_hsp = np.zeros((len(sizes), len(VOCABS)))
                data_src = np.zeros((len(sizes), len(VOCABS)))
                data_trg = np.zeros((len(sizes), len(VOCABS)))
                for i, size in enumerate(sizes):
                    for j, vocab in enumerate(VOCABS):
                        data_psp[i, j] = psp.loc[(f"{l1}→{l2}", alignment, size, vocab), metric]
                        data_usp[i, j] = usp.loc[(f"{l1}→{l2}", alignment, size, vocab), metric]
                        data_hsp[i, j] = hsp.loc[(f"{l1}→{l2}", alignment, size, vocab), metric]
                        data_trg[i, j] = baseline.loc[(f"{l1}→{l2}", alignment, size, vocab), metric]
                        data_src[i, j] = baseline.loc[(f"{l2}→{l1}", alignment, size, vocab), metric]
                fig, axs = plt.subplots(2,3, figsize=(10, 8), sharex=True, sharey=True)
                fig.suptitle(f"{FULL_NAME[l1]} → {FULL_NAME[l2]} ({ALIGN_NAME[alignment]}) {metric.capitalize()}")
                
                axs[0,0].set_title("Paired")
                axs[0,1].set_title("Unpaired")
                axs[0,2].set_title("Hard Counting")

                tmp_ax = axs[0,0].twinx()
                tmp_ax.yaxis.set_label_position('left')
                tmp_ax.spines['left'].set_position(('axes', -0.1))
                tmp_ax.spines['left'].set_visible(False)
                tmp_ax.set_yticks([])
                tmp_ax.set_ylabel(f"vs {FULL_NAME[l1]} SP", rotation=90, size='large', ha='right', va='center')
                tmp_ax = axs[1,0].twinx()
                tmp_ax.yaxis.set_label_position('left')
                tmp_ax.spines['left'].set_position(('axes', -0.1))
                tmp_ax.spines['left'].set_visible(False)
                tmp_ax.set_yticks([])
                tmp_ax.set_ylabel(f"vs {FULL_NAME[l2]} SP", rotation=90, size='large', ha='right', va='center')
                
                mats = [data_psp/data_src, data_usp/data_src, data_hsp/data_src, data_psp/data_trg, data_usp/data_trg, data_hsp/data_trg]
                min_ = min([i.min() for i in mats])
                max_ = max([i.max() for i in mats])
                for mat, ax in zip(mats, axs.flatten()):
                    sns.heatmap(
                        mat, 
                        annot=True, 
                        fmt=".2f", 
                        vmin=min_, 
                        vmax=max_, 
                        xticklabels=map_numbers(VOCABS), 
                        yticklabels=map_numbers(sizes),
                        ax=ax,
                        cbar=False
                    )                
                fig.tight_layout()
                os.makedirs("intrinsic", exist_ok=True)
                fig.savefig(f"intrinsic/{l1}-{l2}_{alignment}_{metric}.pdf", bbox_inches='tight')

def plot_simple(psp, usp, hsp, baseline, metric):    
    for alignment in ALIGNMENTS:
        for lang1, lang2, sizes in TOKENIZER_PAIRS:
            for l1, l2 in [(lang1, lang2), (lang2, lang1)]:   # Directions             
                data_psp = np.zeros((len(sizes), len(VOCABS)))
                data_usp = np.zeros((len(sizes), len(VOCABS)))
                data_hsp = np.zeros((len(sizes), len(VOCABS)))
                data_src = np.zeros((len(sizes), len(VOCABS)))
                data_trg = np.zeros((len(sizes), len(VOCABS)))
                for i, size in enumerate(sizes):
                    for j, vocab in enumerate(VOCABS):
                        data_psp[i, j] = psp.loc[(f"{l1}→{l2}", alignment, size, vocab), metric]
                        data_usp[i, j] = usp.loc[(f"{l1}→{l2}", alignment, size, vocab), metric]
                        data_hsp[i, j] = hsp.loc[(f"{l1}→{l2}", alignment, size, vocab), metric]
                        data_trg[i, j] = baseline.loc[(f"{l1}→{l2}", alignment, size, vocab), metric]
                        data_src[i, j] = baseline.loc[(f"{l2}→{l1}", alignment, size, vocab), metric]
                fig, axs = plt.subplots(2,3, figsize=(10, 8), sharex=True, sharey=True)
                fig.suptitle(f"{FULL_NAME[l1]} → {FULL_NAME[l2]} ({ALIGN_NAME[alignment]}) {metric.capitalize()}")
                
                axs[0,0].set_title("Paired")
                axs[0,1].set_title("Unpaired")
                axs[0,2].set_title("Hard Counring")
                axs[1,0].set_title("Source SentencePiece")
                axs[1,1].set_title("Target SentencePiece")
                
                mats = [data_psp, data_usp, data_hsp, data_src, data_trg]
                min_ = min([i.min() for i in mats])
                max_ = max([i.max() for i in mats])
                for mat, ax in zip(mats, axs.flatten()):
                    sns.heatmap(
                        mat, 
                        annot=True, 
                        fmt=".2f", 
                        vmin=min_, 
                        vmax=max_, 
                        xticklabels=map_numbers(VOCABS), 
                        yticklabels=map_numbers(sizes),
                        ax=ax,
                        cbar=False
                    )                
                fig.tight_layout()
                os.makedirs("intrinsic", exist_ok=True)
                fig.savefig(f"intrinsic/{l1}-{l2}_{alignment}_{metric}.pdf", bbox_inches='tight')

def plot_eflomal(psp, baseline, metric):
    for alignment in ALIGNMENTS:
        for lang1, lang2, sizes in TOKENIZER_PAIRS:
            for l1, l2 in [(lang1, lang2), (lang2, lang1)]:   # Directions             
                data_psp = np.zeros((len(sizes), len(VOCABS)))
                data_trg = np.zeros((len(sizes), len(VOCABS)))
                for i, size in enumerate(sizes):
                    for j, vocab in enumerate(VOCABS):
                        data_psp[i, j] = psp.loc[(f"{l1}→{l2}", alignment, size, vocab), metric]
                        data_trg[i, j] = baseline.loc[(f"{l1}→{l2}", alignment, size, vocab), metric]
                fig, axs = plt.subplots(1,2, figsize=(10, 8), sharex=True, sharey=True)
                fig.suptitle(f"{FULL_NAME[l1]} → {FULL_NAME[l2]} ({ALIGN_NAME[alignment]}) {metric.capitalize()}")
                
                axs[0].set_title("Paired")
                axs[1].set_title("Target SentencePiece")
                
                mats = [data_psp, data_trg]
                min_ = min([i.min() for i in mats])
                max_ = max([i.max() for i in mats])
                for mat, ax in zip(mats, axs.flatten()):
                    sns.heatmap(
                        mat, 
                        annot=True, 
                        fmt=".2f", 
                        vmin=min_, 
                        vmax=max_, 
                        xticklabels=map_numbers(VOCABS), 
                        yticklabels=map_numbers(sizes),
                        ax=ax,
                        cbar=False
                    )                
                fig.tight_layout()
                os.makedirs("intrinsic", exist_ok=True)
                fig.savefig(f"intrinsic/{l1}-{l2}_{alignment}_{metric}.pdf", bbox_inches='tight')
                    
def plot_ratio_trend(psp, baseline, metric, title=None):
    for alignment in ALIGNMENTS:
        # fig, ax = plt.subplots(1,1, figsize=(10, 8), sharex=True, sharey=True)
        fig, ax = plt.subplots(1,1)
        if title is None:
            fig.suptitle(f"{metric.capitalize()}")
        else:
            fig.suptitle(title)
        df = pd.DataFrame(index=VOCABS)
        df_baseline = pd.DataFrame(index=VOCABS)

        for lang1, lang2, sizes in TOKENIZER_PAIRS:
            size = sizes[-1] 
            for l1, l2 in [(lang1, lang2), (lang2, lang1)]:   # Directions             
                for vocab in VOCABS:
                    df.loc[vocab, f"{l1}→{l2}"] = psp.loc[(f"{l1}→{l2}", alignment, size, vocab), metric] / baseline.loc[(f"{l2}→{l1}", alignment, size, vocab), metric]
                    df_baseline.loc[vocab, f"Baseline {l1}→{l2}"] = baseline.loc[(f"{l1}→{l2}", alignment, size, vocab), metric] / baseline.loc[(f"{l2}→{l1}", alignment, size, vocab), metric]
        
        ax.set_xticks(VOCABS, map_numbers(VOCABS))
        ax.set_xlabel("Vocabulary size")
        sns.lineplot(df, ax=ax, marker="o", dashes=False)
        sns.lineplot(df_baseline, ax=ax, marker="x", dashes=[(2,2)]*8)
        os.makedirs("intrinsic", exist_ok=True)
        fig.savefig(f"intrinsic/{metric}.png", bbox_inches='tight', dpi=300)

def plot_trend(psp, baseline, metric, title=None):
    for alignment in ALIGNMENTS:
        fig, ax = plt.subplots(1,1)
        if title is None:
            fig.suptitle(f"{metric.capitalize()}")
        else:
            fig.suptitle(title)
        df = pd.DataFrame(index=VOCABS)
        df_baseline = pd.DataFrame(index=VOCABS)
        for lang1, lang2, sizes in TOKENIZER_PAIRS:
            size = sizes[-1] 
            for l1, l2 in [(lang1, lang2), (lang2, lang1)]:   # Directions             
                for vocab in VOCABS:
                    df.loc[vocab, f"{l1}→{l2}"] = psp.loc[(f"{l1}→{l2}", alignment, size, vocab), metric]
                    df_baseline.loc[vocab, f"Baseline {l1}→{l2}"] = baseline.loc[(f"{l1}→{l2}", alignment, size, vocab), metric]

        ax.set_xticks(VOCABS, map_numbers(VOCABS))  
        ax.set_xlabel("Vocabulary size")
        sns.lineplot(df, ax=ax, marker="o", dashes=False)
        sns.lineplot(df_baseline, ax=ax, marker="x", dashes=[(2,2)]*8)
        os.makedirs("intrinsic", exist_ok=True)
        fig.savefig(f"intrinsic/{metric}.png", bbox_inches='tight', dpi=300)
# %% 
def plot_scatter(data, metrics, y_fn=None, fill=False, mask=None): # TODO adjust labels and layout
    data["vocab size"] = pd.Categorical(data["vocab"], VOCABS)
    if mask is not None:
        data = data[mask]

    if fill:
        p = so.Plot(
            data, 
            x="pair", 
            marker="model", 
            color="vocab size", 
            group="vocab size",
            fill="model",
        )  
    else:
        p = so.Plot(
            data, 
            x="pair", 
            marker="model", 
            color="vocab size", 
            group="vocab size",
        )
    p = p.pair(y=metrics)
    p = p.layout(size=(7, 4))
    p = p.label(
        x="Language Pair", 
        y=y_fn, 
        marker=str.capitalize, color=str.capitalize
    )
    p.theme({'mathtext.default': 'regular'})
    p = p.add(so.Dot(edgecolor="w", alpha=0.5), so.Dodge(by=["group", "marker"]))
    plotter = p.plot()   
    # plotter._figure.get_axes()[-1].tick_params(axis="x", labelrotation=45)
    # plotter._figure.suptitle(title)
    for ax in plotter._figure.get_axes():
        ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator([0.5,1.5,2.5,3.5,4.5,5.5,6.5]))
        ax.grid(True, which="minor", axis="x")
        ax.grid(True, which="major", axis="y")
    plotter.save(f"intrinsic/{"".join(metrics)}.png", bbox_inches='tight', dpi=300)
    plotter.save(f"intrinsic/{"".join(metrics)}.pdf", bbox_inches='tight', dpi=300)
    plotter.show()    
    plotter

# %%

def make_tok_table_data():
    psp = load(score_template, model="PairedSP")
    usp = load(score_unpaired_template, model=r"$\mathregular{PairedSP_{M}}$")
    hard = load(score_hard_template, model=r"$\mathregular{PairedSP_{EM}}$")
    baseline = load(baseline_template, model="Baseline")
    psp["parity"] = -100.0
    usp["parity"] = -100.0
    hard["parity"] = -100.0
    baseline["parity"] = -100.0
    for l1, l2, sizes in TOKENIZER_PAIRS:
        for lang1, lang2 in [(l1, l2), (l2, l1)]:   # Directions
            for size in sizes:
                for vocab in VOCABS:
                    psp.loc[(f"{lang1}→{lang2}", "align", size, vocab), "parity"] = \
                        psp.loc[(f"{lang1}→{lang2}", "align", size, vocab), "length"] /\
                            baseline.loc[(f"{lang2}→{lang1}", "align", size, vocab), "length"]
                    usp.loc[(f"{lang1}→{lang2}", "align", size, vocab), "parity"] = \
                        usp.loc[(f"{lang1}→{lang2}", "align", size, vocab), "length"] /\
                            baseline.loc[(f"{lang2}→{lang1}", "align", size, vocab), "length"]
                    hard.loc[(f"{lang1}→{lang2}", "align", size, vocab), "parity"] = \
                        hard.loc[(f"{lang1}→{lang2}", "align", size, vocab), "length"] /\
                            baseline.loc[(f"{lang2}→{lang1}", "align", size, vocab), "length"]
                    baseline.loc[(f"{lang1}→{lang2}", "align", size, vocab), "parity"] = \
                        baseline.loc[(f"{lang1}→{lang2}", "align", size, vocab), "length"] /\
                            baseline.loc[(f"{lang2}→{lang1}", "align", size, vocab), "length"]
    data = pd.concat([baseline, psp, usp, hard], axis=0)
    data = data.reset_index()
    return data
#%%
data = make_tok_table_data()
# data.to_csv("intrinsic/tok_data.csv", index=False)
plot_scatter(
    data,
    ["fertility", "parity"],
    y_fn=lambda x: f"{x.capitalize()} (↓)",
    mask=data["fertility"] < 3,
)

# %%
psp = load(eflomal_template, EFLOMAL_COLS, model="PairedSP")
baseline = load(eflomal_baseline_template, EFLOMAL_COLS, model="Baseline")
data = pd.concat([baseline, psp], axis=0)
data = data.reset_index()
data.to_csv("intrinsic/align_data.csv", index=False)
plot_scatter(
    data,
    ["one-to-one", "unaligned"],
    y_fn=lambda x: f"{x.capitalize()} (↑)" if x == "one-to-one" else f"{x.capitalize()} (↓)",
    fill=True,
)

plot_scatter(
    data,
    ["one-to-one-train", "ono-to-one-disjoint"],
    fill=True,
)

# %% 
def fill_table(template, data, metric, title):
    template = template.replace("§label§", metric)
    template = template.replace("§metric§", title)
    for lang1, lang2, sizes in TOKENIZER_PAIRS:        
        size = sizes[-1]
        for model in ["PairedSP", "$\mathregular{PairedSP_{M}}$", "$\mathregular{PairedSP_{EM}}$", "Baseline"]:
            for l1, l2 in [(lang1, lang2), (lang2, lang1)]:
                for vocab in VOCABS:
                    mask = (data["pair"] == f"{l1}→{l2}") & (data["alignment"] == "align") & (data["size"] == size) & (data["vocab"] == vocab) & (data["model"] == model)
                    value = data[mask][metric].mean()
                    template = template.replace("§", f"{value:.2f}", 1)
    with open(f"intrinsic/{metric}.tex", "w") as f:
        f.write(template)
# %% 
template = r"""\begin{table}[t]
    \caption{\TODO 4 of these tables}
    \label{tab:§label§}
    \centering
    \footnotesize
    \setlength{\tabcolsep}{3pt}
    \begin{tabular}{ll ccc ccc}
        \toprule
        \multicolumn{8}{c}{\textbf{§metric§}} \\        
        \midrule
        Size & Model & 8k & 16k & 32k & 8k & 16k & 32k \\
        \midrule
        & & \multicolumn{3}{c}{\textbf{Fra $\rightarrow$ Ita}} & 
            \multicolumn{3}{c}{\textbf{Ita $\rightarrow$ Fra}} \\
            \cmidrule(lr){3-5} \cmidrule(lr){6-8}
        \multirow{4}*{1M} & \psp & § & § & § & § & § & § \\
         & \usp & § & § & § & § & § & § \\
         & \pspem & § & § & § & § & § & § \\
        & \sptgt & § & § & § & § & § & § \\
        \midrule
        & & \multicolumn{3}{c}{\textbf{Ces $\rightarrow$ Ukr}} & 
            \multicolumn{3}{c}{\textbf{Ukr $\rightarrow$ Ces}} \\
            \cmidrule(lr){3-5} \cmidrule(lr){6-8}
        \multirow{4}*{1M} & \psp & § & § & § & § & § & § \\
         & \usp & § & § & § & § & § & § \\
         & \pspem & § & § & § & § & § & § \\
        & \sptgt & § & § & § & § & § & § \\
        \midrule
        & & \multicolumn{3}{c}{\textbf{Ita $\rightarrow$ Mlt}} & 
            \multicolumn{3}{c}{\textbf{Mlt $\rightarrow$ Ita}} \\
            \cmidrule(lr){3-5} \cmidrule(lr){6-8}
        \multirow{4}*{100k} & \psp & § & § & § & § & § & § \\
         & \usp & § & § & § & § & § & § \\
         & \pspem & § & § & § & § & § & § \\
        & \sptgt & § & § & § & § & § & § \\
        \midrule
        & & \multicolumn{3}{c}{\textbf{Deu $\rightarrow$ Hsb}} & 
            \multicolumn{3}{c}{\textbf{Hsb $\rightarrow$ Deu}} \\
            \cmidrule(lr){3-5} \cmidrule(lr){6-8}
        \multirow{4}*{60k} & \psp & § & § & § & § & § & § \\
         & \usp & § & § & § & § & § & § \\
         & \pspem & § & § & § & § & § & § \\
        & \sptgt & § & § & § & § & § & § \\
        \bottomrule
    \end{tabular}
\end{table}
"""




data = make_tok_table_data()
fill_table(template, data, "parity", "Parity")
fill_table(template, data, "fertility", "Fertility")

# %%

template = r"""\begin{table}[t]
    \caption{\TODO 4 of these tables}
    \label{tab:§label§}
    \centering
    \footnotesize
    \setlength{\tabcolsep}{3pt}
    \begin{tabular}{ll ccc ccc}
        \toprule
        \multicolumn{8}{c}{\textbf{§metric§}} \\        
        \midrule
        Size & Model & 8k & 16k & 32k & 8k & 16k & 32k \\
        \midrule
        & & \multicolumn{3}{c}{\textbf{Fra $\rightarrow$ Ita}} & 
            \multicolumn{3}{c}{\textbf{Ita $\rightarrow$ Fra}} \\
            \cmidrule(lr){3-5} \cmidrule(lr){6-8}
        \multirow{2}*{1M} & \psp & § & § & § & § & § & § \\
        & \sptgt & § & § & § & § & § & § \\
        \midrule
        & & \multicolumn{3}{c}{\textbf{Ces $\rightarrow$ Ukr}} & 
            \multicolumn{3}{c}{\textbf{Ukr $\rightarrow$ Ces}} \\
            \cmidrule(lr){3-5} \cmidrule(lr){6-8}
        \multirow{2}*{1M} & \psp & § & § & § & § & § & § \\
        & \sptgt & § & § & § & § & § & § \\
        \midrule
        & & \multicolumn{3}{c}{\textbf{Ita $\rightarrow$ Mlt}} & 
            \multicolumn{3}{c}{\textbf{Mlt $\rightarrow$ Ita}} \\
            \cmidrule(lr){3-5} \cmidrule(lr){6-8}
        \multirow{2}*{100k} & \psp & § & § & § & § & § & § \\
        & \sptgt & § & § & § & § & § & § \\
        \midrule
        & & \multicolumn{3}{c}{\textbf{Deu $\rightarrow$ Hsb}} & 
            \multicolumn{3}{c}{\textbf{Hsb $\rightarrow$ Deu}} \\
            \cmidrule(lr){3-5} \cmidrule(lr){6-8}
        \multirow{2}*{60k} & \psp & § & § & § & § & § & § \\
        & \sptgt & § & § & § & § & § & § \\
        \bottomrule
    \end{tabular}
\end{table}
"""

def fill_align_table(template, data, metric, title):
    template = template.replace("§label§", metric)
    template = template.replace("§metric§", title)
    for lang1, lang2, sizes in TOKENIZER_PAIRS:        
        size = sizes[-1]
        for model in ["PairedSP", "Baseline"]:
            for l1, l2 in [(lang1, lang2), (lang2, lang1)]:
                for vocab in VOCABS:
                    mask = (data["pair"] == f"{l1}→{l2}") & (data["alignment"] == "align") & (data["size"] == size) & (data["vocab"] == vocab) & (data["model"] == model)
                    value = data[mask][metric].mean()
                    template = template.replace("§", f"{value:.2f}", 1)
    with open(f"intrinsic/{metric}.tex", "w") as f:
        f.write(template)

psp = load(eflomal_template, EFLOMAL_COLS, model="PairedSP")
baseline = load(eflomal_baseline_template, EFLOMAL_COLS, model="Baseline")
data = pd.concat([baseline, psp], axis=0)
data = data.reset_index()

fill_align_table(template, data, "one-to-one", "One-to-one")
fill_align_table(template, data, "unaligned", "Unaligned")
fill_align_table(template, data, "eflomal", "Eflomal scores")


# %%
import numpy as np

metrics = ["parity", "fertility", "one-to-one"]
# Linear
for metric in metrics:    
    if metric == "one-to-one":
        psp = load(eflomal_template, EFLOMAL_COLS, model="PairedSP")
        baseline = load(eflomal_baseline_template, EFLOMAL_COLS, model="Baseline")
        data = pd.concat([baseline, psp], axis=0)
        data = data.reset_index()
    else:
        data = make_tok_table_data()
    print(f"## {metric} ##")
    for lang1, lang2 in (("fra", "ita"), ("ita", "fra"), ("ces", "ukr"), ("ukr", "ces")):                    
        for vocab in VOCABS:
            mask = (data["pair"] == f"{lang1}→{lang2}") & (data["alignment"] == "align") & (data["vocab"] == vocab) & (data["model"] == "PairedSP")
            xs = data[mask]["size"].values
            ys = data[mask][metric].values
            mask = (data["pair"] == f"{lang1}→{lang2}") & (data["alignment"] == "align") & (data["vocab"] == vocab) & (data["model"] == "Baseline")
            threshold = data[mask][metric].max()
            assert len(xs) == len(ys) == 3, f"Expected 3 values for {lang1}→{lang2} {vocab}, got {len(xs)}"
            A = np.vstack([xs, np.ones(len(xs))]).T
            m, c = np.linalg.lstsq(A, ys)[0]
            # assert m > 0, f"Expected positive slope for {lang1}→{lang2} {vocab}, got {m}"
            print(f"{lang1}→{lang2} {vocab}: y = {m}x + {c}, threshold = {threshold}")
            x = (threshold - c) / m
            print("Expected size:", x)

# %%
import scipy.optimize

for metric in metrics:    
    if metric == "one-to-one":
        psp = load(eflomal_template, EFLOMAL_COLS, model="PairedSP")
        baseline = load(eflomal_baseline_template, EFLOMAL_COLS, model="Baseline")
        data = pd.concat([baseline, psp], axis=0)
        data = data.reset_index()
    else:
        data = make_tok_table_data()
    print(f"## {metric} ##")
    for lang1, lang2 in (("fra", "ita"), ("ita", "fra"), ("ces", "ukr"), ("ukr", "ces")):                    
        for vocab in VOCABS:
            mask = (data["pair"] == f"{lang1}→{lang2}") & (data["alignment"] == "align") & (data["vocab"] == vocab) & (data["model"] == "PairedSP")
            xs = data[mask]["size"].values
            ys = data[mask][metric].values
            mask = (data["pair"] == f"{lang1}→{lang2}") & (data["alignment"] == "align") & (data["vocab"] == vocab) & (data["model"] == "Baseline")
            threshold = data[mask][metric].max()
            assert len(xs) == len(ys) == 3, f"Expected 3 values for {lang1}→{lang2} {vocab}, got {len(xs)}"
            (a, b, c), _ = scipy.optimize.curve_fit(lambda x, a, b, c: a*np.exp(b*x) + c, xs, ys)
            # assert m > 0, f"Expected positive slope for {lang1}→{lang2} {vocab}, got {m}"
            print(f"{lang1}→{lang2} {vocab}: y = {a}e^{b}*x + {c}, threshold = {threshold}")
            x = (np.log(threshold - c) - np.log(a)) / b
            print("Expected size:", x)

# %% 

def make_tok_appendix_data():
    psp = load(score_template, model="PairedSP")
    usp = load(score_unpaired_template, model=r"$\mathregular{PairedSP_{M}}$")
    hard = load(score_hard_template, model=r"$\mathregular{PairedSP_{EM}}$")
    baseline = load(baseline_template, model="Baseline")
    psp["parity"] = -100.0
    usp["parity"] = -100.0
    hard["parity"] = -100.0
    baseline["parity"] = -100.0
    for l1, l2, sizes in TOKENIZER_PAIRS:
        for lang1, lang2 in [(l1, l2), (l2, l1)]:   # Directions
            for size in sizes:
                for vocab in VOCABS:
                    psp.loc[(f"{lang1}→{lang2}", "align", size, vocab), "parity"] = \
                        psp.loc[(f"{lang1}→{lang2}", "align", size, vocab), "length"] /\
                            baseline.loc[(f"{lang2}→{lang1}", "align", size, vocab), "length"]
                    usp.loc[(f"{lang1}→{lang2}", "align", size, vocab), "parity"] = \
                        usp.loc[(f"{lang1}→{lang2}", "align", size, vocab), "length"] /\
                            baseline.loc[(f"{lang2}→{lang1}", "align", size, vocab), "length"]
                    hard.loc[(f"{lang1}→{lang2}", "align", size, vocab), "parity"] = \
                        hard.loc[(f"{lang1}→{lang2}", "align", size, vocab), "length"] /\
                            baseline.loc[(f"{lang2}→{lang1}", "align", size, vocab), "length"]
                    baseline.loc[(f"{lang1}→{lang2}", "align", size, vocab), "parity"] = \
                        baseline.loc[(f"{lang1}→{lang2}", "align", size, vocab), "length"] /\
                            baseline.loc[(f"{lang2}→{lang1}", "align", size, vocab), "length"]
                    
                    psp.loc[(f"{lang1}→{lang2}", "align", size, vocab), "renyi-ratio"] = \
                        psp.loc[(f"{lang1}→{lang2}", "align", size, vocab), "renyi"] /\
                            baseline.loc[(f"{lang1}→{lang2}", "align", size, vocab), "renyi"]
                    usp.loc[(f"{lang1}→{lang2}", "align", size, vocab), "renyi-ratio"] = \
                        usp.loc[(f"{lang1}→{lang2}", "align", size, vocab), "renyi"] /\
                            baseline.loc[(f"{lang1}→{lang2}", "align", size, vocab), "renyi"]
                    hard.loc[(f"{lang1}→{lang2}", "align", size, vocab), "renyi-ratio"] = \
                        hard.loc[(f"{lang1}→{lang2}", "align", size, vocab), "renyi"] /\
                            baseline.loc[(f"{lang1}→{lang2}", "align", size, vocab), "renyi"]
                    baseline.loc[(f"{lang1}→{lang2}", "align", size, vocab), "renyi-ratio"] = \
                        baseline.loc[(f"{lang1}→{lang2}", "align", size, vocab), "renyi"] /\
                            baseline.loc[(f"{lang1}→{lang2}", "align", size, vocab), "renyi"]
    
                    psp.loc[(f"{lang1}→{lang2}", "align", size, vocab), "length-ratio"] = \
                        psp.loc[(f"{lang1}→{lang2}", "align", size, vocab), "length"] /\
                            baseline.loc[(f"{lang1}→{lang2}", "align", size, vocab), "length"]
                    usp.loc[(f"{lang1}→{lang2}", "align", size, vocab), "length-ratio"] = \
                        usp.loc[(f"{lang1}→{lang2}", "align", size, vocab), "length"] /\
                            baseline.loc[(f"{lang1}→{lang2}", "align", size, vocab), "length"]
                    hard.loc[(f"{lang1}→{lang2}", "align", size, vocab), "length-ratio"] = \
                        hard.loc[(f"{lang1}→{lang2}", "align", size, vocab), "length"] /\
                            baseline.loc[(f"{lang1}→{lang2}", "align", size, vocab), "length"]
                    baseline.loc[(f"{lang1}→{lang2}", "align", size, vocab), "length-ratio"] = \
                        baseline.loc[(f"{lang1}→{lang2}", "align", size, vocab), "length"] /\
                            baseline.loc[(f"{lang1}→{lang2}", "align", size, vocab), "length"]
    data = pd.concat([baseline, psp, usp, hard], axis=0)
    data = data.reset_index()
    return data

template = r"""\begin{table}[t]
    \caption{\TODO 4 of these tables}
    \label{tab:§label§}
    \centering
    \footnotesize
    \setlength{\tabcolsep}{3pt}
    \begin{tabular}{ll ccc ccc}
        \toprule
        \multicolumn{8}{c}{\textbf{§metric§}} \\        
        \midrule
        Size & Model & 8k & 16k & 32k & 8k & 16k & 32k \\
        \midrule
        & & \multicolumn{3}{c}{\textbf{Fra $\rightarrow$ Ita}} & 
            \multicolumn{3}{c}{\textbf{Ita $\rightarrow$ Fra}} \\
            \cmidrule(lr){3-5} \cmidrule(lr){6-8}
        \multirow{4}*{100k} & \psp & § & § & § & § & § & § \\
         & \usp & § & § & § & § & § & § \\
         & \pspem & § & § & § & § & § & § \\
        & \sptgt & § & § & § & § & § & § \\
        \midrule
        \multirow{4}*{500k} & \psp & § & § & § & § & § & § \\
         & \usp & § & § & § & § & § & § \\
         & \pspem & § & § & § & § & § & § \\
        & \sptgt & § & § & § & § & § & § \\
        \midrule
        \multirow{4}*{1M} & \psp & § & § & § & § & § & § \\
         & \usp & § & § & § & § & § & § \\
         & \pspem & § & § & § & § & § & § \\
        & \sptgt & § & § & § & § & § & § \\
        \midrule
        & & \multicolumn{3}{c}{\textbf{Ces $\rightarrow$ Ukr}} & 
            \multicolumn{3}{c}{\textbf{Ukr $\rightarrow$ Ces}} \\
            \cmidrule(lr){3-5} \cmidrule(lr){6-8}
        \multirow{4}*{100k} & \psp & § & § & § & § & § & § \\
         & \usp & § & § & § & § & § & § \\
         & \pspem & § & § & § & § & § & § \\
        & \sptgt & § & § & § & § & § & § \\
        \midrule
        \multirow{4}*{500k} & \psp & § & § & § & § & § & § \\
         & \usp & § & § & § & § & § & § \\
         & \pspem & § & § & § & § & § & § \\
        & \sptgt & § & § & § & § & § & § \\
        \midrule
        \multirow{4}*{1M} & \psp & § & § & § & § & § & § \\
         & \usp & § & § & § & § & § & § \\
         & \pspem & § & § & § & § & § & § \\
        & \sptgt & § & § & § & § & § & § \\
        \midrule
        & & \multicolumn{3}{c}{\textbf{Ita $\rightarrow$ Mlt}} & 
            \multicolumn{3}{c}{\textbf{Mlt $\rightarrow$ Ita}} \\
            \cmidrule(lr){3-5} \cmidrule(lr){6-8}
        \multirow{4}*{100k} & \psp & § & § & § & § & § & § \\
         & \usp & § & § & § & § & § & § \\
         & \pspem & § & § & § & § & § & § \\
        & \sptgt & § & § & § & § & § & § \\
        \midrule
        & & \multicolumn{3}{c}{\textbf{Deu $\rightarrow$ Hsb}} & 
            \multicolumn{3}{c}{\textbf{Hsb $\rightarrow$ Deu}} \\
            \cmidrule(lr){3-5} \cmidrule(lr){6-8}
        \multirow{4}*{60k} & \psp & § & § & § & § & § & § \\
         & \usp & § & § & § & § & § & § \\
         & \pspem & § & § & § & § & § & § \\
        & \sptgt & § & § & § & § & § & § \\
        \bottomrule
    \end{tabular}
\end{table}
"""

def fill_table_appendix(template, data, metric, title):
    template = template.replace("§label§", metric)
    template = template.replace("§metric§", title)
    for lang1, lang2, sizes in TOKENIZER_PAIRS:        
        for size in sizes:
            for model in ["PairedSP", "$\mathregular{PairedSP_{M}}$", "$\mathregular{PairedSP_{EM}}$", "Baseline"]:
                for l1, l2 in [(lang1, lang2), (lang2, lang1)]:
                    for vocab in VOCABS:
                        mask = (data["pair"] == f"{l1}→{l2}") & (data["alignment"] == "align") & (data["size"] == size) & (data["vocab"] == vocab) & (data["model"] == model)
                        value = data[mask][metric].mean()
                        template = template.replace("§", f"{value:.2f}", 1)
    with open(f"intrinsic/{metric}.tex", "w") as f:
        f.write(template)

data = make_tok_appendix_data()
for metric, name in zip(
    ["fertility", "parity", "vocab-overlap", "single-char-micro", "vocab-usage", "start-word", "renyi-ratio", "length-ratio"],
    ["Fertiliy", "Parity", "Vocabulary Overlap", "Single Character", "Vocabulary Usage", "Start Word", "Rényi Ratio", "Length Ratio"]
):
    fill_table_appendix(template, data, metric, name)
# %%
