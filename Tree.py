
class Node:
    def __init__(self, feature_idx = None, threshold = None, left = None, right = None, is_Leaf = None, label = None, feat_name = None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.is_Leaf = is_Leaf
        self.label = label
        self.feat_name = feat_name
    
    def __str__(self):
        return f"feat_id: {str(self.feature_idx)}" + " " + f"threshold: {str(self.threshold)}" + " " + f"isLeaf: {str(self.is_Leaf)}" + f"label: {str(self.label
                                                                                                                                                      )}"