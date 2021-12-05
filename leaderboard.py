import pickle, os, shutil
import pandas as pd
import numpy as np

def gen_column_names():
    columns = []
    sizes = ["small", "medium", "large"]
    n = 301
    for size in sizes:
        for i in range(1, n):
            columns.append("{}-{}".format(size, i))
    columns.remove('small-184')
    return columns

def create_leaderboard():
    path = "leaderboard/leaderboard.pkl"
    if os.path.exists(path):
        return
    df = pd.DataFrame(columns=gen_column_names())
    return df

def create_rankings():
    path = "leaderboard/rankings.pkl"
    if os.path.exists(path):
        return
    df = pd.DataFrame(columns=gen_column_names())
    return df

def load_leaderboard():
    path = "leaderboard/leaderboard.pkl"
    return pd.read_pickle(path)

def save_leaderboard(df):    
    path = "leaderboard/leaderboard.pkl"
    df.to_pickle(path)

def save_leaderboard_to_csv(df):    
    path = "leaderboard/leaderboard.csv"
    df.to_csv(path)

def load_rankings():
    path = "leaderboard/rankings.pkl"
    return pd.read_pickle(path)

def save_rankings(df):    
    path = "leaderboard/rankings.pkl"
    df.to_pickle(path)

def save_rankings_to_csv(df):    
    path = "leaderboard/rankings.csv"
    df.to_csv(path)

def update_rankings():
    leaderboard = load_leaderboard()
    rankings = create_rankings()
    for index, row in leaderboard.iterrows():
        rankings = rankings.append(pd.Series(name=index))

    for col in leaderboard:
        scores = np.array(leaderboard[col])
        neg_scores = -scores
        order = neg_scores.argsort()
        ranks = order.argsort() + 1
        rankings[col] = ranks

    average_ranks = []
    for index, row in rankings.iterrows():
        average_rank = np.mean(row)
        average_ranks.append(average_rank)
    rankings = rankings.assign(average_rank=pd.Series())
    rankings["average_rank"] = average_ranks

    save_rankings_to_csv(rankings)

def save_scores(solver_name, scores):
    df = load_leaderboard()
    if solver_name not in df.index:
        df = df.append(pd.Series(name=solver_name))
    for input_name in scores:
        df[input_name][solver_name] = scores[input_name]
    save_leaderboard(df)
    save_leaderboard_to_csv(df)

def get_best_outputs():
    df = load_leaderboard()
    solver_names = list(df.index)
    for input in df:
        best_solver = solver_names[np.argmax(df[input].values)]
        print(input, best_solver)
        size = input.split('-')[0]
        shutil.copyfile('all_outputs/{}/{}/{}.out'.format(best_solver, size, input), 'outputs/{}/{}.out'.format(size, input))

if __name__ == "__main__":
    update_rankings()
    get_best_outputs()