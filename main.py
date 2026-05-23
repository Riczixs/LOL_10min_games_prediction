import pandas as pd
import time
from Tree import Node
import Graph

BLUE_WINS_INDEX = 0
SAMPLES = 9782
id_gen = 0

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
    global id_gen
    if depth >= depth_threshold: #STOP
        res = check_label(data['blueWins'], True)
        id_gen+=1
        return Node(id=id_gen, samples = len(data), feature_idx=0, threshold=0, left=None, right=None, label = res, is_Leaf=True)
    res = check_label(data['blueWins'], False)
    if res != 2:
        id_gen += 1
        return Node(id=id_gen, samples = len(data), feature_idx=0, threshold=0, left=None, right=None, label = res, is_Leaf=True)
    else:
        b_feat, b_name = None, None
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
                b_name = feat
        left, right = partition(data, b_feat, b_mean)
        id_gen += 1
        return Node(id=id_gen, gini = round(b_g, 3), samples = len(data),feat_name=b_name, feature_idx=b_feat, threshold=b_mean, left = train(left, depth=depth+1, depth_threshold=depth_threshold), right=train(right, depth=depth+1, depth_threshold=depth_threshold),is_Leaf=False)

#Przejdź po drzewie i w liściu sprawdź odp
def classify(node : Node, sample) -> int:
    if node.is_Leaf: #Return label of reached leaf
        return node.label
    else:
        if sample[node.feature_idx] < node.threshold:
            return classify(node.left, sample)
        else:
            return classify(node.right, sample)

def test(tree_root : Node, data : pd.DataFrame, d : int):
    tn, tp = 0, 0
    fn, fp = 0, 0
    for s in data.itertuples(index=False):
        s_class = classify(tree_root, s)
        if s_class==0 and s[BLUE_WINS_INDEX] == 0: 
            tn += 1
        elif s_class == 1 and s[BLUE_WINS_INDEX] == 1:
            tp += 1
        elif s_class == 0 and s[BLUE_WINS_INDEX] == 1:
            fp += 1
        else:
            fn += 1
    with open('results70_30.txt', 'a') as f:
        f.write(f"MAX_DEPTH -> {d}\n")
        if tn!= 0 and fp != 0:
            print(f"Specifity -> {tn/(tn+fp)}")
            f.write(f"Specifity -> {tn/(tn+fp)}\n")
        if tp!=0 and fn != 0:
            print(f"Sensitivity -> {tp/(tp+fn)}")
            f.write(f"Sensitivity -> {tp/(tp+fn)}\n")
        print(f"Accurate -> {tp + tn} from {len(data)}")
        f.write(f"Accurate -> {tp + tn} from {len(data)}\n")
        print(f"Accuracy = {(tp + tn)/len(data)}\n\n")
        f.write(f"Accuracy = {(tp + tn)/len(data)}\n\n")
    return (tp + tn)/len(data)


def print_node(node : Node, depth : int):
    if node is None:
        return
    else:
        print(f"depth {depth}",node.__str__(), )
        print_node(node.left, depth+1)
        print_node(node.right, depth+1)

def read_data(data: pd.DataFrame, s : int):
    t1 = int(s*0.3)
    return data[0:t1], data[t1:s]


data = pd.read_csv('high_diamond_ranked_10min.csv', usecols=['blueWins','blueWardsPlaced','blueWardsDestroyed','blueFirstBlood','blueKills','blueDeaths','blueAssists','blueEliteMonsters','blueDragons','blueHeralds','blueTowersDestroyed','blueTotalMinionsKilled','blueTotalJungleMinionsKilled','blueGoldDiff','blueExperienceDiff','redWardsPlaced','redWardsDestroyed','redAssists','redEliteMonsters','redDragons','redHeralds','redTowersDestroyed','redTotalMinionsKilled','redTotalJungleMinionsKilled'])
size = [SAMPLES]
dept = [1,2,3,4]
index = 10
for i in range(0,10):
    rand_data= data[0].sample(frac=1)
    for s in size:
        test_data, training_data  = read_data(rand_data, s)
        for d in dept:
            print(f"Sample -> {s} Depth -> {d}")
            start = time.perf_counter()
            tree_root = train(training_data, 1, d)
            stop = time.perf_counter()
            #UNCOMMENT TO BUILD DECISION TREE GRAPH
            #Graph.make_graph(tree_root, s, d, index)
            index+=1
            print(f"Training time -> {stop-start}")
            result = test(tree_root, test_data, d)