import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

ROOT = os.path.dirname(os.path.dirname(__file__))
PATH_FILE = os.path.join(ROOT, "out", "result.csv")
PATH_PNG1 =  os.path.join(ROOT, "out", "result_map.png")
PATH_PNG2 =  os.path.join(ROOT, "out", "result_n.png")

def main():
    data = pd.read_csv(PATH_FILE)
    data = data.melt(id_vars=["size", "model", "iter"],
            var_name="metrics").query("metrics != 'n_all'")

    # boxplot
    sns.set_theme(palette="Set2",)
    # figure 1
    sns.relplot(x="size", 
                y="value", 
                kind="line",
                hue="model",
                col="metrics",
                col_wrap=2,
                facet_kws=dict(sharey=False),
                data=data.query("metrics in ['precision', 'recall', 'map50', 'map5095']"))

    # save figure
    plt.savefig(PATH_PNG1, dpi=300)

    # figure 2
    sns.relplot(x="size", 
            y="value", 
            kind="line",
            hue="model",
            col="metrics",
            col_wrap=2,
            facet_kws=dict(sharey=False),
            data=data.query("metrics in ['n_missed', 'n_false']"))
    # save figure
    plt.savefig(PATH_PNG2, dpi=300)



if __name__ == "__main__":
    main()