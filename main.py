import numpy as np
import pandas as pd
import time
from Tree import Node

BLUE_WINS_INDEX = 0
SAMPLES = 9782
def gini_left(left_pos, left_neg):
    if left_pos == 0.0 or left_neg == 0.0:
        return 0
    return 1 - ((left_pos/(left_pos + left_neg))**2) - ((left_neg/(left_pos + left_neg))**2) 

def gini_right(right_pos, right_neg):
    if right_pos == 0.0 or right_neg == 0.0:
        return 0
    return 1 - ((right_pos/(right_pos + right_neg))**2) - ((right_neg/(right_pos + right_neg))**2)

def gini_gain(g_left, g_right, left, right):
    if left == 0.0 or right == 0.0:
        return 0
    return 1 - ((left/(left+right))*g_left) - ((right/(left+right))*g_right) 

def get_gain(series : pd.DataFrame, mean : float):
    left_pos, left_neg = 0, 0
    right_pos, right_neg = 0, 0

    for r in series.itertuples(index=False):
        if r[0] < mean: #left
            if r[1] == 1:
                left_pos += 1
            else:
                left_neg += 1
        else:
            if r[1] == 1:
                right_pos += 1
            else:
                right_neg += 1

    result_gain = gini_gain(
        g_left=gini_left(
            left_pos=left_pos,
            left_neg=left_neg
            ),
        g_right=gini_right(
            right_pos=right_pos,
            right_neg=right_neg
        ),
        left = (left_pos+ left_neg),
        right = (right_pos +right_neg)
    )
    return result_gain

def find_best_feat(series : pd.DataFrame): #(feat, blueWins)
    b_gain = 0.0
    b_mean = 0.0
    prev = None
    for r in series.itertuples(index=False):
        if prev is None:
            prev = r[0]
            continue
        mean = (r[0] + prev)/2
        prev = r[0]
        g = get_gain(series=series, mean=mean)
        if b_gain < g:
            b_gain = g
            b_mean = mean
    return b_gain, b_mean

def partition(data : pd.DataFrame, feat : int, mean : float):
    left = []
    right = []
    for r in data.itertuples(index=False):
        if float(r[feat]) <= mean:
            left.append(r)
        else:
            right.append(r)
    return pd.DataFrame(left),pd.DataFrame(right)

def check_label(node_data: pd.DataFrame, isLeaf : bool):
    c_1, c_0 = 0, 0
    for x in node_data:
        if x == 1:
            c_1 += 1
        elif x == 0:
            c_0 += 1
    if c_1 > 0 and c_0 == 0:
        return 1
    elif c_0 > 0 and c_1 == 0:
        return 0
    elif isLeaf: #If depth threshold reached, just return majority label
        return 1 if c_1 >= c_0 else 0
    else:
        return 2
    
def train(data : pd.DataFrame, depth : int, depth_threshold : int):  
    if depth >= depth_threshold: #STOP
        res = check_label(data['blueWins'], True)
        return Node(feature_idx=0, threshold=0, left=None, right=None, label = res, is_Leaf=True)
    res = check_label(data['blueWins'], False)
    if res != 2:
        return Node(feature_idx=0, threshold=0, left=None, right=None, label = res, is_Leaf=True)
    else:
        b_feat = None
        b_g, b_mean = 0, 0
        for index, feat in enumerate(data.columns):
            if feat in ['gameId', 'blueWins', 'index']:
                continue
            series = data[[feat, 'blueWins']].sort_values(feat)
            g, mean = find_best_feat(series)
            if b_g < g:
                b_g = g
                b_feat = index
                b_mean = mean
        left, right = partition(data, b_feat, b_mean)
        return Node(feature_idx=b_feat, threshold=b_mean, left = train(left, depth=depth+1, depth_threshold=depth_threshold), right=train(right, depth=depth+1, depth_threshold=depth_threshold), is_Leaf=False)

#Przejdź po drzewie i w liściu sprawdź odp
def classify(node : Node, sample) -> int:
    if node.is_Leaf: #Return label of reached leaf
        return node.label
    else:
        if sample[node.feature_idx] < node.threshold:
            return classify(node.left, sample)
        else:
            return classify(node.right, sample)

def test(tree_root : Node, data : pd.DataFrame):
    accurate = 0
    for s in data.itertuples(index=False):
        s_class = classify(tree_root, s)
        if s_class == s[BLUE_WINS_INDEX]:
            accurate += 1
    print(f"Accurate -> {accurate} from {len(data)}")
    print(f"Acc = {accurate/len(data)}")
    return accurate/len(data)


def print_node(node : Node, depth : int):
    if node is None:
        return
    else:
        print(f"depth {depth}",node.__str__(), )
        print_node(node.left, depth+1)
        print_node(node.right, depth+1)

def read_data(data: pd.DataFrame, s : int):
    t1 = int(s*0.1)
    return data[0:t1], data[t1:s]


data = pd.read_csv('high_diamond_ranked_10min.csv', usecols=['blueWins','blueWardsPlaced','blueWardsDestroyed','blueFirstBlood','blueKills','blueDeaths','blueAssists','blueEliteMonsters','blueDragons','blueHeralds','blueTowersDestroyed','blueTotalGold','blueTotalExperience','blueTotalMinionsKilled','blueTotalJungleMinionsKilled','blueGoldDiff','blueExperienceDiff','redWardsPlaced','redWardsDestroyed','redAssists','redEliteMonsters','redDragons','redHeralds','redTowersDestroyed','redTotalGold','redTotalExperience','redTotalMinionsKilled','redTotalJungleMinionsKilled']),
rand_data= data[0].sample(frac=1)

size = [2000]
dept = [1,2,3,4,5]
for s in size:
    test_data, training_data  = read_data(rand_data, s)
    #print(test_data, training_data)
    for d in dept:
        start = time.perf_counter()
        tree_root = train(training_data, 1, d)
        stop = time.perf_counter()
        print(f"Training time -> {stop-start} for depth -> {d}")
        result = test(tree_root, test_data)

