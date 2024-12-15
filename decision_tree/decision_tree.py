Author : enesozeren
https://github.com/enesozeren/machine_learning_from_scratch/commits?author=enesozeren

import inspect
import numpy as np
from collections import Counter
from decision_trees.treenode import TreeNode
#from treenode import TreeNode
import json

from rich import print

class DecisionTree():

    """
    Decision Tree Classifier
    Training: Use "train" function with train set features and labels
    Predicting: Use "predict" function with test set features
    """

    def __init__(self, max_depth=4, min_samples_leaf=1, 
                 min_information_gain=0.0, numb_of_features_splitting=None,
                 amount_of_say=None) -> None:
        """
        Setting the class with hyperparameters
        max_depth: (int) -> max depth of the tree
        min_samples_leaf: (int) -> min # of samples required to be in a leaf to make the splitting possible
        min_information_gain: (float) -> min information gain required to make the splitting possible
        num_of_features_splitting: (str) ->  when splitting if sqrt then sqrt(# of features) features considered, 
                                                            if log then log(# of features) features considered
                                                            else all features are considered
        amount_of_say: (float) -> used for Adaboost algorithm
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.numb_of_features_splitting = numb_of_features_splitting
        self.amount_of_say = amount_of_say

    def _entropy(self, class_probabilities: list) -> float:
 #       [print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::p={p}") for p in class_probabilities]
        return sum([-p * np.log2(p) for p in class_probabilities if p>0])

    def _class_probabilities(self, labels: list) -> list:
        total_count = len(labels)

#        print('labels=')
#        print(labels)

#        print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::", end='')
#        print(f"total_count = {total_count}")

#        print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::", end='')
#        print(json.dumps(Counter(labels), indent=4))

        return [label_count / total_count for label_count in Counter(labels).values()]

    def _data_entropy(self, labels: list) -> float:

        class_probabilities_result = self._class_probabilities(labels)
 #       print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::", end='')
 #       print(f"_class_probablities(labels)= {class_probabilities_result}, type = {type(class_probabilities_result)}, len = {len(class_probabilities_result)}")

        result = self._entropy(class_probabilities_result)
 #       print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::", end='')
 #       print(f"result = {result}, type is {type(result)}")

        return result

    def _partition_entropy(self, subsets: list) -> float:
        """subsets = list of label lists (EX: [[1,0,0], [1,1,1])"""

#        print('subsets[0]')
#        print(subsets[0])

#        print('subsets[1]')
#        print(subsets[1])

#        print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::", end='')
#        print(f"len(subsets) = {len(subsets)}, len(subsets[0])={len(subsets[0])}, len(subsets[1])={len(subsets[1])}")

#        result_a = [len(subset) for subset in subsets]
#        print(f'_partition_entropy:: result_a = {result_a}, type(result_a) = {type(result_a)}, len(result_a)={len(result_a)}')

        total_count = sum([len(subset) for subset in subsets])
        result = [self._data_entropy(subset) * (len(subset) / total_count) for subset in subsets]

#        print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::", end='')
#        print(f"total_count = {total_count}, result[0] = {result[0]}, result[1]={result[1]}, type(result) = {type(result)}, len(result)={len(result)}")

        return sum(result)

    def _split(self, data: np.array, feature_idx: int, feature_val: float) -> tuple:

        mask_below_threshold = data[:, feature_idx] < feature_val
        group1 = data[mask_below_threshold]
        group2 = data[~mask_below_threshold]

#        print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::", end='')
#        print(f"feature_idx = {feature_idx}, feature_val = {feature_val}")

#        print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::", end='')
#        print(f'mask_below_threshold {mask_below_threshold}') # get a boolean array.

#        print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::", end='')
#        print(f"group1 = {group1}")

#        print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::", end='')
#        print(f'group2 = {group2}')

        return group1, group2

    def _select_features_to_use(self, data: np.array) -> list:
        """
        Randomly selects the features to use while splitting w.r.t. hyperparameter numb_of_features_splitting
        """
        feature_idx = list(range(data.shape[1]-1))

#        print('_select_features_to_use:: data.shape', data.shape)
#        print('_select_features_to_use:: data.shape[1]', data.shape[1])
#        print('_select_features_to_use:: range(data.shape[1]-1)', range(data.shape[1]-1))

        if self.numb_of_features_splitting == "sqrt":
            feature_idx_to_use = np.random.choice(feature_idx, size=int(np.sqrt(len(feature_idx))))
        elif self.numb_of_features_splitting == "log":
            feature_idx_to_use = np.random.choice(feature_idx, size=int(np.log2(len(feature_idx))))
        else:
            feature_idx_to_use = feature_idx

        return feature_idx_to_use

    def _find_best_split(self, data: np.array) -> tuple:
        """
        Finds the best split (with the lowest entropy) given data
        Returns 2 splitted groups and split information
        """
        min_part_entropy = 1e9
        feature_idx_to_use = self._select_features_to_use(data)

#        print(f'_find_best_split::feature_idx_to_use: {feature_idx_to_use}') # [0,1,2,3]

        for idx in feature_idx_to_use:

#            print(f'_find_best_split::data[:, {idx}', data[:, idx])
#            print(f'_find_best_split::np.arange(25, 100, 25)=', np.arange(25, 100, 25))
            feature_vals = np.percentile(data[:, idx], q=np.arange(25, 100, 25))

            print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::", end='')
            print(f"feature_idx={idx}, feature_vals={feature_vals}")

            for feature_val in feature_vals:

                g1, g2, = self._split(data, idx, feature_val)
                part_entropy = self._partition_entropy([g1[:, -1], g2[:, -1]])

                print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::", end='')
                print(f"part_entropy={part_entropy}, min_part_entropy={min_part_entropy}")

                if part_entropy < min_part_entropy:

                    print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::", end='')
                    print(f"update min_part_entropy")

                    min_part_entropy = part_entropy
                    min_entropy_feature_idx = idx
                    min_entropy_feature_val = feature_val
                    g1_min, g2_min = g1, g2

        print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::", end='')
        print(f"len(data) = {len(data)}, min_entropy_feature_idx={min_entropy_feature_idx}, min_entropy_feature_val={min_entropy_feature_val}, min_part_entropy={min_part_entropy}")

        return g1_min, g2_min, min_entropy_feature_idx, min_entropy_feature_val, min_part_entropy

    def _find_label_probs(self, data: np.array) -> np.array:

        labels_as_integers = data[:,-1].astype(int)
        # Calculate the total number of labels

        print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::labels_as_integers", end='')
        print(f' len = {len(labels_as_integers)}')

 #       print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::labels_as_integers", end='')
 #       print(f' = {labels_as_integers}')

        total_labels = len(labels_as_integers)
        # Calculate the ratios (probabilities) for each label
        label_probabilities = np.zeros(len(self.labels_in_train), dtype=float)

        # Populate the label_probabilities array based on the specific labels
        for i, label in enumerate(self.labels_in_train):

            label_index = np.where(labels_as_integers == i)[0]

#            print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::labels_probabilities", end='')
#            print(f'i={i}, label={label}, label_index={label_index}')

            if len(label_index) > 0:
                label_probabilities[i] = len(label_index) / total_labels

        print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::labels_probabilities", end='')
        print(f' len = {len(label_probabilities)}, {label_probabilities}')

        return label_probabilities

    def _create_tree(self, data: np.array, current_depth: int) -> TreeNode:
        """
        Recursive, depth first tree creation algorithm
        """

        # Check if the max depth has been reached (stopping criteria)
        if current_depth > self.max_depth:
            return None

        print()
        print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::", end='')
        print(f"<<<<{'-'*40}")

        # Find best split
        split_1_data, split_2_data, split_feature_idx, split_feature_val, split_entropy = self._find_best_split(data)

#        print('_create_tree::split_1_data')
#        print(split_1_data)

 #       print('_create_tree::split_2_data')
 #       print(split_2_data)

        print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::", end='')
        print(f"split_feature_idx={split_feature_idx}, split_feature_val={split_feature_val}, split_entropy={split_entropy}")

        # Find label probs for the node
        label_probabilities = self._find_label_probs(data)

        # Calculate information gain
        node_entropy = self._entropy(label_probabilities)
        information_gain = node_entropy - split_entropy

        print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::", end='')
        print(f"TreeNode Created. node_entropy={node_entropy}, split_entropy={split_entropy}, information_gain={information_gain}, depth={current_depth}")

        # Create node
        node = TreeNode(data, split_feature_idx, split_feature_val, label_probabilities, information_gain)

#        print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::", end='')
#        print(f'node def = {node.node_def()}')

        # Check if the min_samples_leaf has been satisfied (stopping criteria)
        if self.min_samples_leaf > split_1_data.shape[0] or self.min_samples_leaf > split_2_data.shape[0]:

            print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::", end='')
            print(f"STOP! self.min_samples_leaf = {self.min_samples_leaf}, split_1_data.shape[0] = {split_1_data.shape[0]}, split_2_data.shape[0] = {split_2_data.shape[0]}")

            return node
        # Check if the min_information_gain has been satisfied (stopping criteria)
        elif information_gain < self.min_information_gain:

            print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::", end='')
            print(f"STOP! self.min_information_gain = {self.min_information_gain}, information_gain = {information_gain}")

            return node

        current_depth += 1
        node.left = self._create_tree(split_1_data, current_depth)
        node.right = self._create_tree(split_2_data, current_depth)

        print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::", end='')
        print(f'node def = {node.node_def()}')

        return node

    def _predict_one_sample(self, X: np.array) -> np.array:
        """Returns prediction for 1 dim array"""
        node = self.tree

        # Finds the leaf which X belongs
        while node:
            pred_probs = node.prediction_probs
            if X[node.feature_idx] < node.feature_val:
                node = node.left
            else:
                node = node.right

        return pred_probs

    def train(self, X_train: np.array, Y_train: np.array) -> None:
        """
        Trains the model with given X and Y datasets
        """

        # Concat features and labels
        self.labels_in_train = np.unique(Y_train)
        train_data = np.concatenate((X_train, np.reshape(Y_train, (-1, 1))), axis=1)

#        print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::", end='')
#        print('labels_in_train')
#        print(self.labels_in_train)

#        print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::", end='')
#        print('train_data')
#        print(train_data)

        # Start creating the tree
        self.tree = self._create_tree(data=train_data, current_depth=0)

        # Calculate feature importance
        self.feature_importances = dict.fromkeys(range(X_train.shape[1]), 0)
        self._calculate_feature_importance(self.tree)

        print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::", end='')
        print(f'feature_importances = {self.feature_importances}')

        # Normalize the feature importance values
        self.feature_importances = {k: v / total for total in (sum(self.feature_importances.values()),) for k, v in self.feature_importances.items()}

        print(f"{' '*4*(len(inspect. stack(0)))}{inspect.currentframe().f_code.co_name}::", end='')
        print(f'(Normalized) feature_importances = {self.feature_importances}')

    def predict_proba(self, X_set: np.array) -> np.array:
        """Returns the predicted probs for a given data set"""

        pred_probs = np.apply_along_axis(self._predict_one_sample, 1, X_set)

        return pred_probs

    def predict(self, X_set: np.array) -> np.array:
        """Returns the predicted labels for a given data set"""

        pred_probs = self.predict_proba(X_set)
        preds = np.argmax(pred_probs, axis=1)

        return preds

    def _print_recursive(self, node: TreeNode, level=0) -> None:
        if node != None:
            self._print_recursive(node.left, level + 1)
            print('    ' * 4 * level + '-> ' + node.node_def())
            self._print_recursive(node.right, level + 1)

    def print_tree(self) -> None:
        self._print_recursive(node=self.tree)

    def _calculate_feature_importance(self, node):
        """Calculates the feature importance by visiting each node in the tree recursively"""
        if node != None:
            self.feature_importances[node.feature_idx] += node.feature_importance
            self._calculate_feature_importance(node.left)
            self._calculate_feature_importance(node.right)
