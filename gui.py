import dearpygui.dearpygui as dpg
import numpy as np

dpg.create_context()


node_counter = [0]
link_counter = [0]
link_ids = []

#Callbacks
def link_callback(sender, app_data):
    input_attr, output_attr = app_data
    link_id = f"link_{link_counter[0]}"
    link_counter[0] += 1
    dpg.add_node_link(input_attr, output_attr, parent="node_editor", tag=link_id)
    link_ids.append(link_id)

def delink_callback(sender, app_data):
    if app_data in link_ids:
        print(f"Link deleted: {app_data}")
        dpg.delete_item(app_data)
        link_ids.remove(app_data)

def delete_node_callback(sender, app_data, user_data):
    dpg.delete_item(user_data)

#Node creation
def create_node(label, inputs=1, outputs=1, pos=(100, 100)):
    node_id = f"node_{node_counter[0]}"
    node_counter[0] += 1

    with dpg.node(label=label, parent="node_editor", pos=pos, tag=node_id):
        for _ in range(inputs):
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                dpg.add_text("Input")
        for _ in range(outputs):
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_text("Output")

    # Right-click popup to delete node
    with dpg.popup(parent=node_id, mousebutton=dpg.mvMouseButton_Right):
        dpg.add_button(label="Delete Node", callback=delete_node_callback, user_data=node_id)

#Node menu 
def add_input_node(): create_node("Input IMG", inputs=0, outputs=1, pos=(20, 30))
def add_output_node(): create_node("Output IMG", inputs=1, outputs=0, pos=(300, 30))
def add_gaussian_node(): create_node("Gaussian Blur", inputs=1, outputs=1, pos=(150, 60))
def add_median_node(): create_node("Median Filter", inputs=1, outputs=1, pos=(150, 120))
def add_histogram_node(): create_node("Histogram Equalization", inputs=1, outputs=1, pos=(150, 180))

#Setup
dpg.create_viewport(title='Node Editor', width=1250, height=700)
dpg.setup_dearpygui()

#Main Node Editor Window
with dpg.window(label="Main window", width=900, height=650):
    with dpg.menu_bar():
        with dpg.menu(label="Add Node"):
            dpg.add_menu_item(label="Input IMG", callback=add_input_node)
            dpg.add_menu_item(label="Output IMG", callback=add_output_node)
            dpg.add_menu_item(label="Gaussian Blur", callback=add_gaussian_node)
            dpg.add_menu_item(label="Median Filter", callback=add_median_node)
            dpg.add_menu_item(label="Histogram Equalization", callback=add_histogram_node)

    with dpg.node_editor(callback=link_callback, delink_callback=delink_callback, tag="node_editor"):
        pass

#Control Panel
with dpg.window(label="Panel", pos=[910, 20], width=300):
    dpg.add_text("Control Panel")
    dpg.add_button(label="Load Image", callback=lambda: print("Load Image Clicked"))

    dpg.add_slider_float(label="Std Dev", default_value=1.0, min_value=0.1, max_value=5.0)

    

#Right-click to delete links
def check_for_link_right_click():
    if dpg.is_item_hovered("node_editor") and dpg.is_mouse_button_released(dpg.mvMouseButton_Right):
        for link_id in link_ids:
            if dpg.is_item_hovered(link_id):
                dpg.delete_item(link_id)
                link_ids.remove(link_id)
                break

with dpg.handler_registry():
    dpg.add_mouse_click_handler(callback=check_for_link_right_click)

#Launch
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
