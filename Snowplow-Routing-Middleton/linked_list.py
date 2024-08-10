class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

    def __str__(self):
        if self.prev is None:
            prev_str = "NONE"
        else:
            prev_str = self.prev.get_data()
        if self.next is None:
            next_str = "NONE"
        else:
            next_str = self.next.get_data()
    
        return str(self.data) + " Prev is " + str(prev_str) + " Next is " + str(next_str)
    def __repr__(self) -> str:
        return self.__str__(self)
    
    def get_data(self):
        return self.data

def two_opt_inter(step1: Node, step2: Node, step1end, step2end):
    old_step1end_next = step1end.next
    old_step2end_next = step2end.next

    step1prev = step1.prev
    step2prev = step2.prev

    step1.prev.next, step2.prev.next = step2, step1

        
    # handle edge cases where one step is end of one route, other is start of the next
    if step2prev == step1end:
        step1.prev = step2end
        step2end.next = step1
        step1end.next = old_step2end_next
        old_step2end_next.prev = step1end
    else:
        step1.prev = step2prev
        step2end.next = old_step1end_next
        old_step1end_next.prev = step2end
    
    if step1prev == step2end:
        step2.prev = step1end
        step1end.next = step2
        step2end.next = old_step1end_next
        old_step1end_next.prev = step2end
    else:
        step2.prev = step1prev
        step1end.next = old_step2end_next
        old_step2end_next.prev = step1end
    

def reverse_list(n1: Node, n2: Node):
    """
    Reverses the linked list between node1A and node2A, inclusive. Node 1 comes before node 2.
    Assumes that node1A and node2A aren't the headA or tailA of the linked list. Never touching the dummy heads and tailAs
    Args:
        node1A (RouteStep): The first routestep.
        node2A (RouteStep): The second routestep.

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

# create list A
headA = Node("0")
node1A = Node("A")
node2A = Node("B")
node3A = Node("C")
node4A = Node("D")
node5A = Node("E")
node6A = Node("F")


# create list B
node1B = Node("G")
node2B = Node("H")
node3B = Node("I")
node4B = Node("J")
node5B = Node("K")
node6B = Node("L")
tailB = Node("0")


headA.next = node1A
node1A.prev = headA
node1A.next = node2A
node2A.prev = node1A
node2A.next = node3A
node3A.prev = node2A
node3A.next = node4A
node4A.prev = node3A
node4A.next = node5A
node5A.prev = node4A
node5A.next = node6A
node6A.prev = node5A
node6A.next = node1B


node1B.prev = node6A
node1B.next = node2B
node2B.prev = node1B
node2B.next = node3B
node3B.prev = node2B
node3B.next = node4B
node4B.prev = node3B
node4B.next = node5B
node5B.prev = node4B
node5B.next = node6B
node6B.prev = node5B
node6B.next = tailB
tailB.prev = node6B


# current = headA
# while current is not None:
#     print(current)
#     current = current.next

# print()

# current = headB
# while current is not None:
#     print(current)
#     current = current.next


two_opt_inter(node2A, node3B, node6A, node6B)
# two_opt_inter(node1B, node3A, node6B, node6A)
current = headA.next
while current is not None:
    print(current)
    current = current.next

print()


