class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

    def __str__(self):
        if self.prev is None:
            prev_str = "None"
        else:
            prev_str = self.prev.get_data()
        if self.next is None:
            next_str = "None"
        else:
            next_str = self.next.get_data()
    
        return str(self.data) + " Prev is " + str(prev_str) + " Next is " + str(next_str)
    def __repr__(self) -> str:
        return str(self.data)
    
    def get_data(self):
        return self.data
    
def reverse_list(n1: Node, n2: Node):
    """
    Reverses the linked list between node1 and node2, inclusive. Node 1 comes before node 2.
    Assumes that node1 and node2 aren't the head or tail of the linked list. Never touching the dummy heads and tails
    Args:
        node1 (RouteStep): The first routestep.
        node2 (RouteStep): The second routestep.

    Returns:
        None
    """
    if n1 == n2:
        return
        
    original_prev = n1.prev
    original_next = n2.next
    current = n1
    final = n2.next

    while current != final:
        old_next = current.next
        old_prev = current.prev

        # don't update extra pointers for n1 and n2
        if current == n1:
            current.prev =  old_next
        elif current == n2:
            current.next = old_prev
        else:
            current.next = old_prev
            current.prev = old_next
        current = old_next
    
    n2.prev = original_prev
    original_prev.next = n2

    n1.next = original_next
    original_next.prev = n1

head = Node(0)
node1 = Node(1)
node2 = Node(2)
node3 = Node(3)
node4 = Node(4)
node5 = Node(5)
node6 = Node(6)
tail = Node(0)

head.next = node1
node1.prev = head
node1.next = node2
node2.prev = node1
node2.next = node3
node3.prev = node2
node3.next = node4
node4.prev = node3
node4.next = node5
node5.prev = node4
node5.next = node6
node6.prev = node5
node6.next = tail
tail.prev = node6

current = head
while current is not None:
    print(current)
    current = current.next

reverse_list(node1, node2)
print("****")

current = head
while current is not None:
    print(current)
    current = current.next

