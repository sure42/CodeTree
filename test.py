class Solution:
    def tree_build(self, preOrder, middleOrder) :
        if not preOrder or not middleOrder:
            return None
        root_value = preOrder[0]
        root_index_in_middle = middleOrder.index(root_value)

        left_tree = self.tree_build(preOrder[1:1+root_index_in_middle], middleOrder[:root_index_in_middle])
        right_tree = self.tree_build(preOrder[1+root_index_in_middle:], middleOrder[root_index_in_middle+1:])
        return (left_tree,right_tree,root_value)

    def tree_travel(self, tree):
        if tree is None:
            return []
        (left_tree,right_tree,root_value) = tree
        return self.tree_travel(left_tree) + self.tree_travel(right_tree) + [root_value]
    
    def getLRD(self, preOrder, middleOrder) :
        # write code here
        tree = self.tree_build(preOrder, middleOrder)
        ans = self.tree_travel(tree)
        return "".join(ans)

test = Solution()
print(test.getLRD("01234","10324"))