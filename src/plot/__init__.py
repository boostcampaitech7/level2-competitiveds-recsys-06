import pandas as pd


def draw_hist_from_df(
    df: pd.DataFrame, dir_path: str = "./plot", group: str = "target"
):
    import plotly.express as px
    import os

    os.makedirs(dir_path, exist_ok=True)
    for c in df.columns:
        fig = px.histogram(df, x=c, color=group)
        fig.add_histogram(x=df[c], name="All")
        fig.write_html(os.path.join(dir_path, f"Hist-{group}-{c}.html"))
        fig = None


def draw_line_from_df(
    df: pd.DataFrame, x: str = "ID", dir_path: str = "./plot", group: str = "target"
):
    import plotly.express as px
    import os

    os.makedirs(dir_path, exist_ok=True)
    for c in df.columns:
        dtype = df[c].dtype.name
        if not dtype.__contains__("int") and not (dtype.__contains__("float")):
            continue
        fig = px.line(df, x=x, y=c, color="target")
        fig.add_scatter(x=df[x], y=df[c], name="All")
        fig.write_html(os.path.join(dir_path, f"Line-{group}-{c}-{x}.html"))
        fig = None
