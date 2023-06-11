import seaborn as sns
import pandas as pd
import os

ROOT = os.path.dirname(os.path.dirname(__file__))

PATH_FILE = os.path.join(ROOT, "out", "result.csv")


data = pd.read_csv(PATH_FILE)
data = data.melt(id_vars=["size", "model", "iter"],
          var_name="metrics").query("metrics != 'n_all'")

# boxplot
sns.set_theme(palette="Set2",)
sns.relplot(x="size", 
            y="value", 
            kind="line",
            hue="model",
            col="metrics",
            col_wrap=3,
            facet_kws=dict(sharey=False),
            # err_style="bars",
            # errorbar=("se", 2), 
            data=data.query("metrics in ['precision', 'recall', 'map50', 'map5095']"))



data