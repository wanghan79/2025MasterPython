#include "redBlackTree.h"

// 初始化哨兵结点
void RedBlackTree::initializeNULLNode(RBTNode *node, RBTNode *parent)
{
    node->neighbor = 0; // TNULL 的数据值通常不重要
    node->color = BLACK;
    node->left = nullptr; // 哨兵结点的子结点指向自身或nullptr都可以，这里用nullptr
    node->right = nullptr;
    node->parent = parent;
    node->heap_ptr = nullptr;
}

// 左旋操作
void RedBlackTree::leftRotate(RBTNode *x)
{
    RBTNode *y = x->right;
    x->right = y->left;
    if (y->left != TNULL)
    {
        y->left->parent = x;
    }
    y->parent = x->parent;
    if (x->parent == nullptr) // x 是根结点
    {
        this->root = y;
    }
    else if (x == x->parent->left) // x 是父结点的左孩子
    {
        x->parent->left = y;
    }
    else // x 是父结点的右孩子
    {
        x->parent->right = y;
    }
    y->left = x;
    x->parent = y;
}

// 右旋操作
void RedBlackTree::rightRotate(RBTNode *x)
{
    RBTNode *y = x->left;
    x->left = y->right;
    if (y->right != TNULL)
    {
        y->right->parent = x;
    }
    y->parent = x->parent;
    if (x->parent == nullptr) // x 是根结点
    {
        this->root = y;
    }
    else if (x == x->parent->right) // x 是父结点的右孩子
    {
        x->parent->right = y;
    }
    else // x 是父结点的左孩子
    {
        x->parent->left = y;
    }
    y->right = x;
    x->parent = y;
}

// 插入修复
void RedBlackTree::insertFix(RBTNode *k)
{
    // 当父结点是红色时才需要修复 (违反性质4)
    while (k != root && k->parent->color == RED)
    {
        // 父结点是祖父结点的左孩子
        if (k->parent == k->parent->parent->left)
        {
            RBTNode *uncle = k->parent->parent->right; // 叔叔结点
            // Case 1: 叔叔结点是红色
            if (uncle->color == RED)
            {
                k->parent->color = BLACK;
                uncle->color = BLACK;
                k->parent->parent->color = RED;
                k = k->parent->parent; // 将 k 指向祖父结点，继续向上检查
            }
            // Case 2 & 3: 叔叔结点是黑色
            else
            {
                // Case 2: k 是父结点的右孩子 (形成<形状)
                if (k == k->parent->right)
                {
                    k = k->parent; // 将 k 指向父结点
                    leftRotate(k); // 对父结点进行左旋，转换为 Case 3
                }
                // Case 3: k 是父结点的左孩子 (形成/形状)
                k->parent->color = BLACK;
                k->parent->parent->color = RED;
                rightRotate(k->parent->parent); // 对祖父结点进行右旋
                // 修复完成后，循环条件不再满足，会自动退出
            }
        }
        // 父结点是祖父结点的右孩子 (对称情况)
        else
        {
            RBTNode *uncle = k->parent->parent->left; // 叔叔结点
            // Case 1: 叔叔结点是红色
            if (uncle->color == RED)
            {
                k->parent->color = BLACK;
                uncle->color = BLACK;
                k->parent->parent->color = RED;
                k = k->parent->parent;
            }
            // Case 2 & 3: 叔叔结点是黑色
            else
            {
                // Case 2: k 是父结点的左孩子 (形成>形状)
                if (k == k->parent->left)
                {
                    k = k->parent;
                    rightRotate(k); // 对父结点进行右旋，转换为 Case 3
                }
                // Case 3: k 是父结点的右孩子 (形成\形状)
                k->parent->color = BLACK;
                k->parent->parent->color = RED;
                leftRotate(k->parent->parent); // 对祖父结点进行左旋
            }
        }
    }
    // 循环结束后，确保根结点是黑色 (性质2)
    root->color = BLACK;
}

// 删除修复
void RedBlackTree::deleteFix(RBTNode *x)
{
    // 当 x 不是根且 x 是黑色时，需要修复（删除导致黑色高度不平衡）
    while (x != root && x->color == BLACK)
    {
        // x 是其父结点的左孩子
        if (x == x->parent->left)
        {
            RBTNode *sibling = x->parent->right; // 兄弟结点

            // Case 1: 兄弟结点是红色
            if (sibling->color == RED)
            {
                sibling->color = BLACK;
                x->parent->color = RED;
                leftRotate(x->parent);
                sibling = x->parent->right; // 更新兄弟结点（现在兄弟是黑色的）
            }

            // Case 2: 兄弟结点是黑色，且其两个子结点都是黑色
            if (sibling->left->color == BLACK && sibling->right->color == BLACK)
            {
                sibling->color = RED; // 将兄弟结点设为红色
                x = x->parent;        // 将 x 指向父结点，问题上移
            }
            else
            {
                // Case 3: 兄弟结点是黑色，其左孩子是红色，右孩子是黑色
                if (sibling->right->color == BLACK)
                {
                    sibling->left->color = BLACK;
                    sibling->color = RED;
                    rightRotate(sibling);
                    sibling = x->parent->right; // 更新兄弟结点，转换为 Case 4
                }

                // Case 4: 兄弟结点是黑色，其右孩子是红色
                sibling->color = x->parent->color; // 兄弟结点继承父结点的颜色
                x->parent->color = BLACK;          // 父结点设为黑色
                sibling->right->color = BLACK;     // 兄弟的右孩子设为黑色
                leftRotate(x->parent);             // 对父结点进行左旋
                x = root;                          // 修复完成，将 x 指向根结点以退出循环
            }
        }
        // x 是其父结点的右孩子 (对称情况)
        else
        {
            RBTNode *sibling = x->parent->left; // 兄弟结点

            // Case 1: 兄弟结点是红色
            if (sibling->color == RED)
            {
                sibling->color = BLACK;
                x->parent->color = RED;
                rightRotate(x->parent);
                sibling = x->parent->left;
            }

            // Case 2: 兄弟结点是黑色，且其两个子结点都是黑色
            if (sibling->left->color == BLACK && sibling->right->color == BLACK)
            {
                sibling->color = RED;
                x = x->parent;
            }
            else
            {
                // Case 3: 兄弟结点是黑色，其右孩子是红色，左孩子是黑色
                if (sibling->left->color == BLACK)
                {
                    sibling->right->color = BLACK;
                    sibling->color = RED;
                    leftRotate(sibling);
                    sibling = x->parent->left;
                }

                // Case 4: 兄弟结点是黑色，其左孩子是红色
                sibling->color = x->parent->color;
                x->parent->color = BLACK;
                sibling->left->color = BLACK;
                rightRotate(x->parent);
                x = root;
            }
        }
    }
    // 确保最终的 x (可能是新的根或某个结点) 是黑色
    x->color = BLACK;
}

// 替换子树 u 和 v
void RedBlackTree::rbTransplant(RBTNode *u, RBTNode *v)
{
    if (u->parent == nullptr) // u 是根结点
    {
        root = v;
    }
    else if (u == u->parent->left) // u 是左孩子
    {
        u->parent->left = v;
    }
    else // u 是右孩子
    {
        u->parent->right = v;
    }
    v->parent = u->parent; // 设置 v 的父结点，即使 v 是 TNULL
}

// 查找以 node 为根的子树中的最小值结点
RBTNode *RedBlackTree::minimum(RBTNode *node)
{
    while (node->left != TNULL)
    {
        node = node->left;
    }
    return node;
}

// 实际执行删除的辅助函数
void RedBlackTree::deleteNodeHelper(RBTNode *node, int key)
{
    RBTNode *z = TNULL; // 要删除的结点
    RBTNode *x, *y;     // x 是实际被删除或移动的结点的原始位置的替代者，y 是实际被删除或移动的结点

    // 查找要删除的结点 z
    RBTNode *current = node;
    while (current != TNULL)
    {
        if (current->neighbor == key)
        {
            z = current;
            break; // 找到结点
        }
        if (key < current->neighbor)
        {
            current = current->left;
        }
        else
        {
            current = current->right;
        }
    }

    if (z == TNULL) // 未找到要删除的结点
    {
        std::cout << "Key " << key << " not found in the tree." << std::endl;
        return;
    }

    y = z;                           // y 初始化为要删除的结点 z
    Color yOriginalColor = y->color; // 记录 y 的原始颜色

    // Case 1 & 2: z 最多只有一个孩子
    if (z->left == TNULL)
    {
        x = z->right;              // x 是 z 的右孩子（或 TNULL）
        rbTransplant(z, z->right); // 用 z 的右孩子替换 z
    }
    else if (z->right == TNULL)
    {
        x = z->left;              // x 是 z 的左孩子
        rbTransplant(z, z->left); // 用 z 的左孩子替换 z
    }
    // Case 3: z 有两个孩子
    else
    {
        y = minimum(z->right);     // y 是 z 的右子树中的最小结点（即 z 的后继）
        yOriginalColor = y->color; // 记录后继结点 y 的原始颜色
        x = y->right;              // x 是 y 的右孩子（y 没有左孩子，因为它是最小值）

        if (y->parent == z) // 如果 y 是 z 的直接右孩子
        {
            x->parent = y; // 设置 x 的父结点，即使 x 是 TNULL
        }
        else // 如果 y 不是 z 的直接右孩子
        {
            rbTransplant(y, y->right); // 用 y 的右孩子替换 y
            y->right = z->right;       // 将 z 的右子树挂到 y 的右边
            y->right->parent = y;      // 更新 y 右孩子的父指针
        }

        rbTransplant(z, y);  // 用 y 替换 z
        y->left = z->left;   // 将 z 的左子树挂到 y 的左边
        y->left->parent = y; // 更新 y 左孩子的父指针
        y->color = z->color; // y 继承 z 的颜色
    }

    // 删除结点 z 的内存
    delete z;

    // 如果删除的结点（或其替代者 y）的原始颜色是黑色，则可能破坏红黑树性质，需要修复
    if (yOriginalColor == BLACK)
    {
        deleteFix(x); // 从 x 开始修复
    }
}

// 递归删除树（用于析构函数）
void RedBlackTree::destroyTree(RBTNode *node)
{
    if (node != TNULL)
    {
        destroyTree(node->left);
        destroyTree(node->right);
        delete node;
    }
}

// 构造函数
RedBlackTree::RedBlackTree()
{
    TNULL = new RBTNode();              // 创建哨兵结点
    initializeNULLNode(TNULL, nullptr); // 初始化哨兵结点
    root = TNULL;                       // 初始时根结点指向哨兵
}

// 析构函数
RedBlackTree::~RedBlackTree()
{
    destroyTree(this->root); // 递归删除所有结点
    delete TNULL;            // 删除哨兵结点
}

// 插入结点 (传入已创建的结点指针)
void RedBlackTree::insert(RBTNode *node)
{
    // node 已经创建并传入，设置其初始属性
    node->parent = nullptr;
    node->left = TNULL;
    node->right = TNULL;
    node->color = RED; // 新插入的结点总是红色的

    RBTNode *y = nullptr;    // y 将是 node 的父结点
    RBTNode *x = this->root; // x 用于遍历树

    // 查找插入位置
    while (x != TNULL)
    {
        y = x;
        if (node->neighbor < x->neighbor)
        {
            x = x->left;
        }
        else if (node->neighbor > x->neighbor)
        {
            x = x->right;
        }
        else
        {
            // 处理键值重复的情况
            // 这里简单地打印错误并返回，不插入重复结点
            // 你可以根据需要修改此行为（例如更新结点信息）
            std::cerr << "Error: Duplicate key " << node->neighbor << " found. Insertion aborted." << std::endl;
            // 注意：如果调用者分配了 node，这里不 delete 它，由调用者处理
            return;
        }
    }

    // 将 node 插入树中
    node->parent = y;
    if (y == nullptr) // 树为空
    {
        root = node;
    }
    else if (node->neighbor < y->neighbor)
    {
        y->left = node;
    }
    else
    {
        y->right = node;
    }

    // 如果新结点是根结点，颜色设为黑色并返回
    if (node->parent == nullptr)
    {
        node->color = BLACK;
        return;
    }

    // 如果父结点是根结点，无需修复 (因为根总是黑色，父结点不会是红色)
    if (node->parent->parent == nullptr)
    {
        return;
    }

    // 调用插入修复函数
    insertFix(node);
}

// 删除结点 (根据键值删除)
void RedBlackTree::deleteNode(int data)
{
    deleteNodeHelper(this->root, data);
}

// 查找结点 (根据键值查找)
RBTNode *RedBlackTree::searchTree(int key)
{
    RBTNode *node = root;
    while (node != TNULL)
    {
        if (key == node->neighbor)
        {
            return node; // 找到结点
        }
        if (key < node->neighbor)
        {
            node = node->left;
        }
        else
        {
            node = node->right;
        }
    }
    return nullptr; // 未找到结点 (返回 nullptr 比返回 TNULL 更清晰)
}

// 获取根结点
RBTNode *RedBlackTree::getRoot()
{
    return this->root;
}

// 递归打印树的辅助函数
void RedBlackTree::printHelper(RBTNode *node, std::string indent, bool last)
{
    if (node != TNULL)
    {
        std::cout << indent;
        if (last) // 是右孩子或根
        {
            std::cout << "R----";
            indent += "   ";
        }
        else // 是左孩子
        {
            std::cout << "L----";
            indent += "|  ";
        }

        std::string sColor = node->color == RED ? "RED" : "BLACK";
        std::cout << node->neighbor << "(" << sColor << ")" << std::endl;
        printHelper(node->left, indent, false); // 打印左子树
        printHelper(node->right, indent, true); // 打印右子树
    }
}

// 打印树结构
void RedBlackTree::printTree()
{
    if (root != TNULL)
    {
        printHelper(this->root, "", true);
    }
    else
    {
        std::cout << "Tree is empty." << std::endl;
    }
}

// 中序遍历树，将所有结点构成一个单链表
void RedBlackTree::inorderToList(RBTNode *node, LinkListNode *&tail)
{
    if (node == TNULL)
        return;
    inorderToList(node->left, tail);
    tail->next = new LinkListNode(node);
    tail = tail->next;
    inorderToList(node->right, tail);
}

// 获取中序遍历的单链表,
LinkListNode *RedBlackTree::getLinkList()
{
    LinkListNode *dummyHead = new LinkListNode(); // 虚拟头结点
    LinkListNode *tail = dummyHead;
    inorderToList(root, tail);
    LinkListNode *realHead = dummyHead->next;
    delete dummyHead;
    return realHead;
}
