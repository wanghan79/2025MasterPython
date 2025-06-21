from python.maxHeap import *
from enum import Enum


class Color(Enum):
    """定义红黑树节点的颜色"""
    RED = 0
    BLACK = 1


class RBTNode:
    """红黑树节点类"""
    
    def __init__(self, neighbor=-1, color=Color.RED, parent=None, left=None, right=None, heap_ptr=None, weight=0.0):
        self.neighbor = neighbor      # 邻居簇的id
        self.color = color           # 节点的颜色（红或黑）
        self.left = left             # 左子节点指针
        self.right = right           # 右子节点指针
        self.parent = parent         # 父节点指针
        self.heap_ptr = heap_ptr     # 指向元素在最大堆中的位置的指针
        self.weight = weight         # 和data相连的节点的权重


class LinkListNode:
    """单链表节点定义"""
    
    def __init__(self, rbtnode=None, next_node=None):
        self.rbtnode = rbtnode       # 红黑树节点指针
        self.next = next_node        # 下一个节点指针


class RedBlackTree:
    """红黑树类定义"""
    
    def __init__(self):
        """构造函数"""
        self.TNULL = RBTNode()  # 哨兵节点（表示空节点）
        self.TNULL.color = Color.BLACK
        self.TNULL.left = None
        self.TNULL.right = None
        self.root = self.TNULL
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'root'):
            self.destroy_tree(self.root)
    
    def initialize_null_node(self, node, parent):
        """初始化哨兵节点"""
        node.neighbor = -1
        node.parent = parent
        node.left = None
        node.right = None
        node.color = Color.BLACK
    
    def left_rotate(self, x):
        """左旋操作"""
        y = x.right
        x.right = y.left
        if y.left != self.TNULL:
            y.left.parent = x
        
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
    
    def right_rotate(self, x):
        """右旋操作"""
        y = x.left
        x.left = y.right
        if y.right != self.TNULL:
            y.right.parent = x
        
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y
    
    def insert_fix(self, k):
        """插入修复"""
        while k.parent and k.parent.color == Color.RED:
            if k.parent == k.parent.parent.left:
                u = k.parent.parent.right  # uncle
                if u.color == Color.RED:
                    # case 3.1
                    u.color = Color.BLACK
                    k.parent.color = Color.BLACK
                    k.parent.parent.color = Color.RED
                    k = k.parent.parent
                else:
                    if k == k.parent.right:
                        # case 3.2.2
                        k = k.parent
                        self.left_rotate(k)
                    # case 3.2.1
                    k.parent.color = Color.BLACK
                    k.parent.parent.color = Color.RED
                    self.right_rotate(k.parent.parent)
            else:
                u = k.parent.parent.left  # uncle
                
                if u.color == Color.RED:
                    # mirror case 3.1
                    u.color = Color.BLACK
                    k.parent.color = Color.BLACK
                    k.parent.parent.color = Color.RED
                    k = k.parent.parent
                else:
                    if k == k.parent.left:
                        # mirror case 3.2.2
                        k = k.parent
                        self.right_rotate(k)
                    # mirror case 3.2.1
                    k.parent.color = Color.BLACK
                    k.parent.parent.color = Color.RED
                    self.left_rotate(k.parent.parent)
            if k == self.root:
                break
        self.root.color = Color.BLACK
    
    def delete_fix(self, x):
        """删除修复"""
        while x != self.root and x != self.TNULL and x.color == Color.BLACK:
            if x.parent and x == x.parent.left:
                s = x.parent.right
                if s and s.color == Color.RED:
                    # case 3.1
                    s.color = Color.BLACK
                    x.parent.color = Color.RED
                    self.left_rotate(x.parent)
                    s = x.parent.right
                
                if s and s.left and s.right and s.left.color == Color.BLACK and s.right.color == Color.BLACK:
                    # case 3.2
                    s.color = Color.RED
                    x = x.parent
                else:
                    if s and s.right and s.right.color == Color.BLACK:
                        # case 3.3
                        if s.left:
                            s.left.color = Color.BLACK
                        s.color = Color.RED
                        self.right_rotate(s)
                        s = x.parent.right
                    
                    # case 3.4
                    if s:
                        s.color = x.parent.color
                        x.parent.color = Color.BLACK
                        if s.right:
                            s.right.color = Color.BLACK
                        self.left_rotate(x.parent)
                    x = self.root
            else:
                if x.parent:
                    s = x.parent.left
                    if s and s.color == Color.RED:
                        # case 3.1
                        s.color = Color.BLACK
                        x.parent.color = Color.RED
                        self.right_rotate(x.parent)
                        s = x.parent.left
                    
                    if s and s.right and s.left and s.right.color == Color.BLACK and s.left.color == Color.BLACK:
                        # case 3.2
                        s.color = Color.RED
                        x = x.parent
                    else:
                        if s and s.left and s.left.color == Color.BLACK:
                            # case 3.3
                            if s.right:
                                s.right.color = Color.BLACK
                            s.color = Color.RED
                            self.left_rotate(s)
                            s = x.parent.left
                        
                        # case 3.4
                        if s:
                            s.color = x.parent.color
                            x.parent.color = Color.BLACK
                            if s.left:
                                s.left.color = Color.BLACK
                            self.right_rotate(x.parent)
                        x = self.root
                else:
                    break
        if x and x != self.TNULL:
            x.color = Color.BLACK
    
    def delete_node_helper(self, node, key):
        """删除节点的辅助函数"""
        z = self.TNULL
        while node != self.TNULL:
            if node.neighbor == key:
                z = node
            
            if node.neighbor <= key:
                node = node.right
            else:
                node = node.left
        
        if z == self.TNULL:
            print("Couldn't find key in the tree")
            return
        
        y = z
        y_original_color = y.color
        if z.left == self.TNULL:
            x = z.right
            self.rb_transplant(z, z.right)
        elif z.right == self.TNULL:
            x = z.left
            self.rb_transplant(z, z.left)
        else:
            y = self.minimum(z.right)
            y_original_color = y.color
            x = y.right
            if y.parent == z:
                x.parent = y
            else:
                self.rb_transplant(y, y.right)
                y.right = z.right
                y.right.parent = y
            
            self.rb_transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color
        
        if y_original_color == Color.BLACK:
            self.delete_fix(x)
    
    def rb_transplant(self, u, v):
        """替换子树 u 和 v"""
        if u.parent is None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent
    
    def print_helper(self, root, indent, last):
        """递归打印树的辅助函数"""
        if root != self.TNULL:
            print(indent, end="")
            if last:
                print("R----", end="")
                indent += "     "
            else:
                print("L----", end="")
                indent += "|    "
            
            s_color = "RED" if root.color == Color.RED else "BLACK"
            print(f"{root.neighbor}({s_color})")
            self.print_helper(root.left, indent, False)
            self.print_helper(root.right, indent, True)
    
    def minimum(self, node):
        """查找最小值节点"""
        while node.left != self.TNULL:
            node = node.left
        return node
    
    def inorder_to_list(self, node, head):
        """辅助递归函数 - 中序遍历构建链表"""
        if node != self.TNULL:
            self.inorder_to_list(node.left, head)
            
            # 创建新的链表节点
            new_link_node = LinkListNode(node)
            if head[0] is None:
                head[0] = new_link_node
                head[1] = new_link_node  # tail
            else:
                head[1].next = new_link_node
                head[1] = new_link_node
            
            self.inorder_to_list(node.right, head)
    
    def insert(self, node):
        """插入节点"""
        node.parent = None
        node.left = self.TNULL
        node.right = self.TNULL
        node.color = Color.RED  # 新节点必须是红色
        
        y = None
        x = self.root
        
        while x != self.TNULL:
            y = x
            if node.neighbor < x.neighbor:
                x = x.left
            else:
                x = x.right
        
        # y 是 node 的父节点
        node.parent = y
        if y is None:
            self.root = node
        elif node.neighbor < y.neighbor:
            y.left = node
        else:
            y.right = node
        
        # 如果新节点是根节点，直接着色为黑色并返回
        if node.parent is None:
            node.color = Color.BLACK
            return
        
        # 如果祖父节点为空，返回
        if node.parent.parent is None:
            return
        
        # 修复树
        self.insert_fix(node)
    
    def delete_node(self, data):
        """删除节点"""
        self.delete_node_helper(self.root, data)
    
    def destroy_tree(self, node):
        """递归删除树"""
        if node and node != self.TNULL:
            self.destroy_tree(node.left)
            self.destroy_tree(node.right)
    
    def search_tree(self, key):
        """查找节点"""
        node = self.root
        while node != self.TNULL and key != node.neighbor:
            if key < node.neighbor:
                node = node.left
            else:
                node = node.right
        return node if node != self.TNULL else None
    
    def get_root(self):
        """获取根节点"""
        return self.root
    
    def get_link_list(self):
        """中序遍历，将所有节点构成一个单链表，返回头节点"""
        if self.root == self.TNULL:
            return None
        
        head = [None, None]  # [head, tail]
        self.inorder_to_list(self.root, head)
        return head[0]  # 返回头节点
    
    def print_tree(self):
        """打印树的结构"""
        if self.root:
            self.print_helper(self.root, "", True)


# 示例使用
if __name__ == "__main__":
    # 创建红黑树
    rbt = RedBlackTree()
    
    # 创建一些节点并插入
    node1 = RBTNode(neighbor=10, weight=1.5)
    node2 = RBTNode(neighbor=20, weight=2.5)
    node3 = RBTNode(neighbor=5, weight=0.5)
    node4 = RBTNode(neighbor=15, weight=1.8)
    node5 = RBTNode(neighbor=25, weight=3.0)
    
    rbt.insert(node1)
    rbt.insert(node2)
    rbt.insert(node3)
    rbt.insert(node4)
    rbt.insert(node5)
    
    print("红黑树结构:")
    rbt.print_tree()
    
    print("\n搜索节点15:")
    found_node = rbt.search_tree(15)
    if found_node:
        print(f"找到节点: neighbor={found_node.neighbor}, weight={found_node.weight}")
    else:
        print("未找到节点")
    
    print("\n中序遍历链表:")
    link_head = rbt.get_link_list()
    current = link_head
    while current:
        print(f"neighbor: {current.rbtnode.neighbor}, weight: {current.rbtnode.weight}")
        current = current.next
    
    print("\n删除节点20后:")
    rbt.delete_node(20)
    rbt.print_tree()