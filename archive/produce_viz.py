from utils.viz import visualize_example, visualize_example_with_energy, plot_y_with_initial_conditions, visualize_loss, print_summary


def produce_viz(exp_n, vars):
    visualize_example()
    visualize_example_with_energy()
    plot_y_with_initial_conditions()
    visualize_loss()
    print_summary()
    return