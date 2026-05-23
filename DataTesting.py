from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cbook as cbk

def box_plots(data : pd.DataFrame):
    with open("outliers.csv", "w") as f:
        f.write("argument,mean,med,q1,q3,outliers\n")
        for col in data.columns.to_list():
            box = plt.boxplot(data[col])
            boxstats = cbk.boxplot_stats(data[col])[0]
            plt.title(f"Boxplot of the attribute '{col}'")
            outliers_count = len(box.get("fliers")[0].get_xdata())
            f.write(f"{col},{boxstats['mean']},{boxstats['med']},{boxstats['q1']},{boxstats['q3']},{len(boxstats['fliers'])}\n")
            plt.savefig(f"plots/{col}.png")
            plt.clf()

def blueWins_correlation_plot(data: pd.DataFrame):
    corrs = []
    key_func = lambda x: x[0]
    for col in data.columns.to_list():
        if str(col) == "blueWins" or str(col) == "gameId":
            continue
        spearman_corr = data["blueWins"].corr(other=data[col],method = "spearman")
        corrs.append((spearman_corr,col))
    sorted_corrs = sorted(corrs, key=key_func)
    corrs = [x[0] for x in sorted_corrs]
    cols = [x[1] for x in sorted_corrs]
    plt.barh(cols, corrs, edgecolor='black')
    plt.title("Attributes correlation with target attribute")
    plt.show()


def correlation_plot(data : pd.DataFrame):
    spearman_corr = data.corr(method = "spearman")
    size = len(data.columns)
    _, ax = plt.subplots(figsize=(size,size))
    sns.heatmap(spearman_corr, cmap = "coolwarm", annot=True, annot_kws={'fontsize':5}, ax=ax)
    plt.title("Correlation Heatmap")
    plt.show()

def normality_test(data : pd.DataFrame):
    with open("result.csv", "w") as r:
                r.write("column_name,value\n")
                for col in data.columns.to_list():
                    print(col)
                    _,res = stats.normaltest(data[col].tolist())
                    if float(res) != 0:
                        r.write(f"{col}, {res}\n")

def read_data():
    with open('high_diamond_ranked_10min.csv', 'r') as file:
        hdg = pd.read_csv(file) 
        df = pd.DataFrame(hdg)
        box_plots(df)
        blueWins_correlation_plot(df)
        normality_test(df)
        correlation_plot(df)

read_data()