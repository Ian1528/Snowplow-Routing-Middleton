import initialize
import dual_graphs
import construction
import plotting
import sectioning_streets.sectioning as sectioning
import params
from shortest_paths import ShortestPaths
from solution import Solution
from crossover import apply_crossover
from genetic import run_genetic
from routes_representations import full_routes, full_routes_with_returns
import folium
import costs


def create_instance(params:tuple[str, str]=("smalltoy", "genetic"), take_input=False) -> None:
    """
    Creates an instance of the graph and its dual graph based on user input.

    Args:
        params (tuple(str,str)): First item indicates which problem to use. Second item indicates solution method. Defaults to smalltoy genetic
        take_input (bool, optional): Flag indicating whether to manually input problem parameters. Default is False.

    Returns:
        None
    """
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

    return G, G_DUAL

def solve_section(polygon_path: str, label_color: str, path_color: str, required_parts: bool = True, m: folium.Map | None = None) -> tuple[folium.Map, Solution, list]:
    """
    Solves the sectioning problem for a given polygon path.
    Args:
        polygon_path (str): The path to the polygon file.
        label_color (str): The color of the labels for the plotted routes.
        path_color (str): The color of the paths for the plotted routes.
        required_parts (bool, optional): Flag indicating whether the polygon has non-required and required parts. Defaults to True.
        m (folium.Map | None, optional): The folium map object. Defaults to None.
    Returns:
        folium.Map: The folium map object with the plotted routes.
        Solution: The solution object.
        list: The full route.
    Raises:
        None
    Example:
        solve_section('/path/to/polygon_file', 'red', 'blue', m)
    """
    
    G = sectioning.section_component(polygon_path, required_parts)
    params.DEPOT = params.find_depot(G)[0]
    DEPOT = params.DEPOT
    print("Depot found, sections are done")

    G = initialize.add_multi_edges(G)
    G_DUAL = dual_graphs.create_dual_streets(G, DEPOT)
    print("Graphs created. Depot is at", params.DEPOT)
    shortest_paths = ShortestPaths(G_DUAL, False, True)

    print("Shortest paths created, running genetic algorithm")
    sol = run_genetic(G, shortest_paths, DEPOT)

    full_route = full_routes_with_returns(G, shortest_paths, sol.routes, DEPOT)

    time_seconds = costs.route_travel_time(G, full_route, DEPOT)
    # Display costs and travel time
    print("Routes cost", sol.cost)
    print("Travel time hours", time_seconds/3600)

    # plot
    G_graph = plotting.add_order_attribute(G, sol.routes)
    plotting.draw_labeled_multigraph(G_graph, 'order', size=(75,75), plotDepot=True)

    m = plotting.plot_routes_folium(G, full_route, m, label_color, path_color)
    return m, sol, full_route
if __name__ == "__main__":
    # 1. Create primal and dual graphs
    G, G_DUAL = create_instance(("smalltoy", "genetic"))
    # 2. Generate shortest paths model
    shortest_paths = ShortestPaths(G_DUAL, load_data=False, save_data=False)
    # 3. Generate initial routes
    r1, rreq1 = construction.route_generation(G, shortest_paths)
    r2, rreq2 = construction.route_generation(G, shortest_paths)

    # 4. Route Improvement Algorithms
    sol = run_genetic(G, shortest_paths)

    for route in sol.routes:
        for edge in route:
            print(edge)
        print("***")


    # 5. Plot final routes
    G_graph = plotting.add_order_attribute(G, sol.routes)
    plotting.draw_labeled_multigraph(G_graph, 'order')

    print("DONE")
    print("Routes cost", sol.cost)