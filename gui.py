import dearpygui.dearpygui as dpg
import numpy as np

# Global variable for storing the line series ID
line_series_tag = "gaussian_series"

# Callback for creating a link
def link_callback(sender, app_data):
    input_attr, output_attr = app_data
    print(f"Link created between {input_attr} -> {output_attr}")
    dpg.add_node_link(input_attr, output_attr, parent=sender)

# Callback for removing a link
def delink_callback(sender, app_data):
    print(f"Link deleted: {app_data}")
    dpg.delete_item(app_data)

# Callback to update the Gaussian curve
def update_gaussian_curve(sender, app_data, user_data):
    std_dev = app_data
    x = np.linspace(-5, 5, 100)
    y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x / std_dev) ** 2))
    dpg.set_value(line_series_tag, [x.tolist(), y.tolist()])

# Create context
dpg.create_context()
dpg.create_viewport(title='Node editor', width=1250, height=700)
dpg.setup_dearpygui()

# Main node editor window
with dpg.window(label="Main window", width=900, height=650):
    with dpg.node_editor(callback=link_callback, delink_callback=delink_callback, tag="node_editor"):
        # Input node
        with dpg.node(label="Input IMG", pos=[20, 20]):
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, tag="A_output"):
                dpg.add_text("")

        # Output node
        with dpg.node(label="Output IMG", pos=[300, 150]):
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, tag="B_input"):
                dpg.add_text("")

        # Gaussian Blur node
        with dpg.node(label="Gaussian Blur", pos=[150, 300]):
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, tag="Gauss_input"):
                dpg.add_text("Input")
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, tag="Gauss_output"):
                dpg.add_text("Output")

        # Median Filter node
        with dpg.node(label="Median filter", pos=[170, 300]):
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, tag="Median_input"):
                dpg.add_text("Input")
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, tag="Median_output"):
                dpg.add_text("Output")

# Control panel window
with dpg.window(label="Panel", pos=[910, 20], width=300):
    dpg.add_text("Control Panel")
    dpg.add_button(label="Load Image", callback=lambda: print("Load Image Clicked"))

    # Gaussian curve std dev slider
    dpg.add_slider_float(label="Std Dev", default_value=1.0, min_value=0.1, max_value=5.0,
                         callback=update_gaussian_curve)

    # Plot
    with dpg.plot(label="Gaussian Curve", width=280, height=200):
        dpg.add_plot_legend()

        with dpg.plot_axis(dpg.mvXAxis, label="X") as x_axis:
            pass

        with dpg.plot_axis(dpg.mvYAxis, label="Y") as y_axis:
            x = np.linspace(-5, 5, 100)
            std_dev = 1.0
            y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x / std_dev) ** 2))
            dpg.add_line_series(x, y, label="Gaussian", parent=y_axis, tag=line_series_tag)

# Show and run
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
