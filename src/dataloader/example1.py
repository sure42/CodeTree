import os
import sys
import javalang  

from dictionary import Dictionary
from dataloader import GPTCoNuTDataLoader
from javalang.ast import Node

GPT_CONUT_TRAINER_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('\\') + 1]
sys.path.append(GPT_CONUT_TRAINER_DIR + '../models/')
sys.path.append(GPT_CONUT_TRAINER_DIR + '../dataloader/')


class JavaCodeParser:
    def __init__(self):
        self.valid_modifiers = {"public", "protected", "private", "abstract", "static", "final", "strictfp", "default"}

    def class_check(self, src):
        """
        判断给定的Java代码字符串是否表示一个类。

        Args:
            src (str): 要检查的Java代码字符串。

        Returns:
            bool: 如果是类，则返回 True；否则返回 False。
        """
        tmp_src = src.strip()

        if tmp_src.startswith("class "):  # 检查代码是否以 "class " 关键字开头
            return True
        else:
            src_without_modifiers = self.modifiers_remove(tmp_src) # 检查修饰符
            # 再次检查class
            if src_without_modifiers.startswith("class "):  # 检查代码是否以 "class " 关键字开头
                return True
            else:
                return False

    def modifiers_remove(self, src):
        """
        从给定的 Java 代码字符串中移除以指定的修饰符开头的部分。

        Args:
            src (str): 要处理的 Java 代码字符串。

        Returns:
            str: 移除修饰符后的 Java 代码字符串。
        """
        src_without_modifiers = None
        tmp_src = src
        while True:
            src_splited = tmp_src.split(" ")
            modifiers = src_splited[0]
            if modifiers.strip() in self.valid_modifiers:
                print("remove", src_splited[0])
                tmp_src = ' '.join(src_splited[1:])
            else:
                src_without_modifiers = tmp_src
                print("modifiers not exit or have been removed!")
                break
        return src_without_modifiers
    
    def parse_java_class(self, src):
        """
        将输入的 Java 代码解析为抽象语法树，并确保代码始终表示一个类定义。

        Args:
            src (str): 要解析的 Java 代码字符串。

        Returns:
            javalang.ast.Node: 表示解析后的类的抽象语法树（AST）对象。
        """
        tmp_src = src.strip()
        if not self.class_check(tmp_src): # 不是一个类
            tmp_src = "public class MyClass {" + tmp_src + "}"
        return javalang.parse.parse(tmp_src)

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []

def ast_to_tree(ast):

    for path, node in ast:  
        if isinstance(node, Node):
            print(node.children)
            if node.children is not None:    
            # 找到函数内部的块（block）节点  
                for block_path, block_node in node.children:  
                    if block_node.type == 'block':  
                        # 遍历块节点内部的语句  
                        for statement_path, statement_node in block_node.children:  
                            # 打印每个语句的类型和内容  
                            print(f"Statement Type: {statement_node.type}, Content: {statement_node.value}")  
  

    # if isinstance(node, javalang.ast.Node):
    #     tree_node = TreeNode(str(node))
    #     if node.children is not None:
    #         for child_name, child_node in node.children or []:
    #             child_tree = ast_to_tree(child_node)
    #             tree_node.children.append((child_name, child_tree))
    #     return tree_node
    # else:
    #     return TreeNode(str(node))

def print_ast(node, depth=0):
    if isinstance(node, Node):  
        print(' ' * depth + str(node))  
        for child in node.children:  
            print_ast(child, depth + 4)  
    elif isinstance(node, list):  
        for item in node:  
            print_ast(item, depth)  
    else:  
        print(' ' * depth + str(node))  

java_code = """
public class Example {  
    public int add(int a, int b) {  
        int sum = 0;  
        sum = a+b;
        return sum;  
    }  
    public int delete(int a, int b) {  
        int sum = a - b;  
        return sum;  
    }  
} 
"""

myparse = JavaCodeParser()
# # tree = ast.parse(src)
# java_ast = myparse.parse_java_class(java_code)
# print_ast(java_ast)


# vocab_file = GPT_CONUT_TRAINER_DIR + '../../data/vocabulary/vocabulary.txt'
# train_file = GPT_CONUT_TRAINER_DIR + '../../data/data/training_src.txt'
# dictionary = Dictionary(vocab_file, min_cnt=0)
# train_loader = GPTCoNuTDataLoader(train_file, dictionary)
# train_loader.load_data(0, 100)
# src = ' '.join(train_loader.src[0]).split("<CTX>")[1]
print(java_code)
java_ast = myparse.parse_java_class(java_code)
# print(java_ast)
print("s"*20)
print_ast(java_ast)














    
