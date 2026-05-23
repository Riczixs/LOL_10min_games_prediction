
class Node:
    def __init__(self, id=None, feature_idx = None, threshold = None, left = None, right = None, is_Leaf = None, label = None, feat_name = None, samples = 0, gini=0.0):
        self.id = id
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.is_Leaf = is_Leaf
        self.label = label
        self.feat_name = feat_name
        self.samples = samples
        self.gini = gini
    
    def __str__(self):
        return f"feat: {str(self.feat_name)}\n"+ f"gini: {str(self.gini)}\n" + f"threshold: {str(self.threshold)}\n" + f"class: {str(self.label )}\n" + f"samples: {str(self.samples)}\n"
