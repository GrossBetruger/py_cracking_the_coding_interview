from anytree import Node, RenderTree, findall


def in_order_traversal(bin_tree: Node):
    if len(bin_tree.children) == 0:
        print(f'visiting: {bin_tree.name}')

    elif len(bin_tree.children) == 2:
        left, right = bin_tree.children
        in_order_traversal(left)
        print(f'visiting: {bin_tree.name}')
        in_order_traversal(right)

    elif len(bin_tree.children) == 1:
        left, = bin_tree.children
        in_order_traversal(left)
        print(f'visiting: {bin_tree.name}')


if __name__ == '__main__':
    udo = Node("Udo")
    marc = Node("Marc", parent=udo)
    lian = Node("Lian", parent=marc)
    dan = Node("Dan", parent=udo)
    jet = Node("Jet", parent=dan)
    jan = Node("Jan", parent=dan)
    joe = Node("Joe", parent=dan)
    jan2 = Node("Jan", parent=joe)
    root = udo

    for pre, fill, nd in RenderTree(udo):
        print(f'{pre}{nd.name}')

    print(findall(root, filter_=lambda n: n.name == "Jan"))
    print(findall(root, filter_=lambda n: n.name == "Jet"))
    print()

    layer_2_right = [Node(10), Node(15)]
    layer_2_left = [Node(3), Node(9)]
    layer_1 = [Node(8, children=layer_2_left), Node(12, children=layer_2_right), ]
    binary_tree_root = Node(10, children=layer_1)

    for pre, fill, nd in RenderTree(binary_tree_root):
        print(f'{pre}{nd.name}')
    in_order_traversal(binary_tree_root)
