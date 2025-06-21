#pragma once
#include "maxHeap.h"
#include <string>

// 定义红黑树结点的颜色
enum Color
{
    RED,
    BLACK
};

// 红黑树结点结构体定义
struct RBTNode
{
    int neighbor = -1;          // 邻居簇的id
    Color color = RED;          // 结点的颜色（红或黑）
    RBTNode *left = nullptr;    // 左子结点指针
    RBTNode *right = nullptr;   // 右子结点指针
    RBTNode *parent = nullptr;  // 父结点指针
    MHNode *heap_ptr = nullptr; // 指向元素在最大堆中的位置的指针
    double weight = 0.0;        // 和data相连的节点的权重
    // 构造函数（可选，方便初始化）
    RBTNode(int val = 0, Color c = RED, RBTNode *p = nullptr, RBTNode *l = nullptr, RBTNode *r = nullptr, MHNode *hp = nullptr)
        : neighbor(val), color(c), left(l), right(r), parent(p), heap_ptr(hp) {}
};
// 单链表结点定义
struct LinkListNode
{
    RBTNode *rbtnode;   // 红黑树结点指针
    LinkListNode *next; // 下一个结点指针
    LinkListNode(RBTNode *node = nullptr, LinkListNode *n = nullptr) : rbtnode(node), next(n) {}
};
// 红黑树类定义
class RedBlackTree
{
public:
    RBTNode *root;                                                  // 根结点
    RBTNode *TNULL;                                                 // 哨兵结点（表示空结点）
    RedBlackTree();                                                 // 构造函数
    ~RedBlackTree();                                                // 析构函数
    void initializeNULLNode(RBTNode *node, RBTNode *parent);        // 初始化哨兵结点
    void leftRotate(RBTNode *x);                                    // 左旋操作
    void rightRotate(RBTNode *x);                                   // 右旋操作
    void insertFix(RBTNode *k);                                     // 插入修复
    void deleteFix(RBTNode *x);                                     // 删除修复
    void deleteNodeHelper(RBTNode *node, int key);                  // 删除结点的辅助函数
    void rbTransplant(RBTNode *u, RBTNode *v);                      // 替换子树 u 和 v
    void printHelper(RBTNode *root, std::string indent, bool last); // 递归打印树的辅助函数
    RBTNode *minimum(RBTNode *node);                                // 查找最小值结点
    void inorderToList(RBTNode *node, LinkListNode *&tail);         // 辅助递归函数
    void insert(RBTNode *node);                                     // 插入结点
    void deleteNode(int data);                                      // 删除结点
    void destroyTree(RBTNode *node);                                // 递归删除树
    RBTNode *searchTree(int key);                                   // 查找结点
    RBTNode *getRoot();                                             // 获取根结点
    LinkListNode *getLinkList();                                    // 中序遍历,将所有节点构成一个单链表,头结点返回
    void printTree();                                               // 打印树的结构
};
