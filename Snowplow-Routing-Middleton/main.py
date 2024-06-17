import initialize
import dual_graphs
import construction
import plotting
from shortest_paths import ShortestPaths
from crossover import crossover_routes


G = None
G_DUAL = None

def create_instance(params:tuple[str, str]=("smalltoy", "genetic"), take_input=False) -> None:
    """
    Creates an instance of the graph and its dual graph based on user input.

    Args:
        params (tuple(str,str)): First item indicates which problem to use. Second item indicates solution method. Defaults to smalltoy genetic
        take_input (bool, optional): Flag indicating whether to manually input problem parameters. Default is False.

    Returns:
        None
    """

    global G
    global G_DUAL

    # 1. Generate Graphs
    if take_input:
        instance = input("Enter the instance name (smalltoy, smallstreets, fullstreets): ")
    else:
        instance = params[0]

    match instance:
        case "smallstreets":
            G = initialize.create_small_streets()
        case "fullstreets":
            G = initialize.create_full_streets()
        case "smalltoy":
            G = initialize.create_small_toy()
        case _:
            print("Invalid instance name")
    if take_input:       
        approach = input("Enter the solution approach (annealing, genetic): ")
    else:
        approach = params[1]

    if approach == "genetic":
        G = initialize.add_multi_edges(G)
        if instance == "smallstreets" or instance == "fullstreets":
            G_DUAL = dual_graphs.create_dual_streets(G)
        else:
            G_DUAL = dual_graphs.create_dual_toy(G)

    elif approach == "annealing":
        if instance == "smallstreets" or instance == "fullstreets":
            G_DUAL = dual_graphs.create_dual_streets(G, False, True)
        else:
            G_DUAL = dual_graphs.create_dual_toy(G, False, True)

# 1. Create primal and dual graphs
create_instance(("smallstreets", "genetic"))
# 2. Generate shortest paths model
shortest_paths = ShortestPaths(G_DUAL, load_data=False, save_data=False)
# 3. Generate initial routes
r1, rreq1 = construction.route_generation(G, shortest_paths)
r2, rreq2 = construction.route_generation(G, shortest_paths)
# 4. Plot initial routes
G_graph = plotting.add_order_attribute(G, rreq1)
plotting.draw_labeled_multigraph(G_graph, 'order')

# 5. Route Improvement Algorithms
rreq0 = crossover_routes(G, rreq1, rreq2, shortest_paths)

print("DONE")