# -*- coding: utf-8 -*-
import dearpygui.dearpygui as dpg 
import numpy as np
from PIL import Image
import os

from image_filters import ImageFilters

# Global dictionaries for application state
node_registry = {} 
node_outputs = {} 
links = {} 
global_texture_counter = 0 

_current_dialog_node_id = None

def load_image_to_dpg_texture(image_path):
    # Loads an image and converts it to RGBA numpy array.
    try:
        img = Image.open(image_path).convert("RGBA")
        width, height = img.size
        data_normalized = np.array(img, dtype=np.float32) / 255.0
        flattened_dpg_data = data_normalized.flatten()
        return width, height, flattened_dpg_data, data_normalized
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None, None, None

def numpy_to_dpg_texture_data(numpy_image):
    # Converts a NumPy array to a flattened RGBA texture format.
    if numpy_image is None: 
        return None

    if numpy_image.dtype != np.float32:
        if numpy_image.max() > 1.01:
            numpy_image = numpy_image.astype(np.float32) / 255.0
        else:
            numpy_image = numpy_image.astype(np.float32)

    numpy_image = np.clip(numpy_image, 0.0, 1.0)

    if numpy_image.ndim == 2:
        numpy_image = np.stack([numpy_image, numpy_image, numpy_image], axis=-1)
        
    if numpy_image.shape[-1] == 3:
        alpha_channel = np.ones((*numpy_image.shape[:-1], 1), dtype=np.float32)
        numpy_image = np.concatenate((numpy_image, alpha_channel), axis=-1)
    elif numpy_image.shape[-1] != 4:
        print(f"Unsupported number of image channels: {numpy_image.shape[-1]}")
        return None

    return numpy_image.flatten()

def update_dpg_texture(base_texture_name, width, height, data):
    # Creates a new dynamic texture with a unique tag.
    global global_texture_counter
    global_texture_counter += 1
    
    new_texture_tag = f"{base_texture_name}_v{global_texture_counter}"
    
    with dpg.texture_registry(show=False):
        if dpg.does_item_exist(new_texture_tag):
            dpg.delete_item(new_texture_tag)
        dpg.add_dynamic_texture(width=width, height=height, default_value=data, tag=new_texture_tag)
    
    return new_texture_tag


def apply_gaussian_blur(image_data, size, sigma):
    # Applies Gaussian blur to the image data.
    if image_data is None: return None
    
    image_data_uint8 = (image_data * 255).astype(np.uint8)
    
    blurred_image_uint8 = np.zeros_like(image_data_uint8)
    
    if image_data_uint8.ndim == 3: 
        for c in range(image_data_uint8.shape[2]):
            blurred_image_uint8[:, :, c] = ImageFilters.gaussian_blur(image_data_uint8[:, :, c], size=size, sigma=sigma)
    else: 
        blurred_image_uint8 = ImageFilters.gaussian_blur(image_data_uint8, size=size, sigma=sigma)
        
    return blurred_image_uint8.astype(np.float32) / 255.0

def apply_median_filter(image_data, size):
    # Applies median filter to the image data.
    if image_data is None: return None
    
    image_data_uint8 = (image_data * 255).astype(np.uint8)
    
    filtered_image_uint8 = np.zeros_like(image_data_uint8)
    
    if image_data_uint8.ndim == 3: 
        for c in range(image_data_uint8.shape[2]):
            filtered_image_uint8[:, :, c] = ImageFilters.median_filter(image_data_uint8[:, :, c], size=size)
    else: 
        filtered_image_uint8 = ImageFilters.median_filter(image_data_uint8, size=size)
        
    return filtered_image_uint8.astype(np.float32) / 255.0

def apply_histogram_equalization(image_data):
    # Applies histogram equalization to the image data.
    if image_data is None: return None
    
    image_data_uint8 = (image_data * 255).astype(np.uint8)
    
    equalized_image_uint8 = np.zeros_like(image_data_uint8)
    
    if image_data_uint8.ndim == 3: 
        for c in range(image_data_uint8.shape[2]):
            equalized_image_uint8[:, :, c] = ImageFilters.histogram_equalization(image_data_uint8[:, :, c])
    else: 
        equalized_image_uint8 = ImageFilters.histogram_equalization(image_data_uint8)
        
    return equalized_image_uint8.astype(np.float32) / 255.0

def apply_edge_detection(image_data, force):
    # Applies edge detection to the image data.
    if image_data is None: return None
    
    image_data_uint8 = (image_data * 255).astype(np.uint8)
    
    edged_image_uint8 = np.zeros_like(image_data_uint8)
    
    if image_data_uint8.ndim == 3: 
        for c in range(image_data_uint8.shape[2]):
            edged_image_uint8[:, :, c] = ImageFilters.edge_detection(image_data_uint8[:, :, c], force=force)
    else: 
        edged_image_uint8 = ImageFilters.edge_detection(image_data_uint8, force=force)
        
    return edged_image_uint8.astype(np.float32) / 255.0

def get_node_input_image(node_id, current_node_outputs):
    # Retrieves the input image for a given node.
    target_input_pin_id = node_registry[node_id].get('input_pin_tag')

    if not (target_input_pin_id and dpg.does_item_exist(target_input_pin_id)):
        return None

    for link_id, (source_pin, target_pin_from_link) in links.items():
        if target_pin_from_link == target_input_pin_id:
            source_node_id = dpg.get_item_parent(source_pin)
            source_output_pin_id = node_registry.get(source_node_id, {}).get('output_pin_tag')
            
            if source_pin == source_output_pin_id:
                retrieved_image = current_node_outputs.get(source_node_id)
                return retrieved_image
            else:
                print(f"WARNING: Link {link_id} has an invalid source pin.")
                return None

    return None

def process_node(node_id, current_node_outputs):
    # Processes the given node based on its type and updates its output.
    node_type = node_registry[node_id]['type']
    
    effective_input_image = None
    if node_type == "input_node":
        effective_input_image = current_node_outputs.get(node_id) 
    else:
        effective_input_image = get_node_input_image(node_id, current_node_outputs)

    processed_output_image = None

    if node_type == "input_node":
        processed_output_image = effective_input_image
    elif node_type == "gaussian_blur_node":
        if effective_input_image is not None:
            sigma = dpg.get_value(node_registry[node_id]['sigma_slider'])
            size = dpg.get_value(node_registry[node_id]['size_slider'])
            processed_output_image = apply_gaussian_blur(effective_input_image, size, sigma)
    elif node_type == "median_filter_node":
        if effective_input_image is not None:
            size = dpg.get_value(node_registry[node_id]['size_slider'])
            processed_output_image = apply_median_filter(effective_input_image, size)
    elif node_type == "hist_equalization_node":
        if effective_input_image is not None:
            processed_output_image = apply_histogram_equalization(effective_input_image)
    elif node_type == "edge_detection_node": 
        if effective_input_image is not None:
            force = dpg.get_value(node_registry[node_id]['force_slider'])
            processed_output_image = apply_edge_detection(effective_input_image, force)
    elif node_type == "output_node":
        processed_output_image = effective_input_image 
        
        pass
    
    current_node_outputs[node_id] = processed_output_image


def re_process_graph(start_node_id=None):
    # Reprocesses the graph, performing a topological sort to determine processing order.
    in_degree = {node_id: 0 for node_id in node_registry}
    graph = {node_id: [] for node_id in node_registry} 

    if dpg.does_item_exist("node_editor"):
        for link_id, (source_pin, target_pin) in links.items():
            source_node = dpg.get_item_parent(source_pin)
            target_node = dpg.get_item_parent(target_pin)
            
            if source_node in node_registry and target_node in node_registry:
                graph[source_node].append(target_node)
                in_degree[target_node] += 1
            else:
                print(f"WARNING: Link {link_id} involves non-existent node(s). Skipping.")

    queue = [node_id for node_id in node_registry if in_degree[node_id] == 0]
    
    if start_node_id is not None and start_node_id not in queue and in_degree.get(start_node_id, 0) == 0:
        queue.insert(0, start_node_id) 

    processed_nodes_in_order = []

    while queue:
        u = queue.pop(0)
        processed_nodes_in_order.append(u)

        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    if len(processed_nodes_in_order) != len(node_registry):
        print(f"WARNING: Topological sort did not process all nodes. Possible cycle or disconnected nodes.")
        remaining_nodes = [node_id for node_id in node_registry if node_id not in processed_nodes_in_order]
        processed_nodes_in_order.extend(remaining_nodes)

    for node_id in processed_nodes_in_order:
        if not dpg.does_item_exist(node_id):
            print(f"WARNING: Node {node_id} no longer exists in DPG. Skipping processing.")
            continue
        
        try:
            process_node(node_id, node_outputs)
        except Exception as e:
            print(f"Error processing node {node_id}: {e}")

def link_callback(sender, app_data):
    # Handles new link creation in the node editor.
    source_pin_id, target_pin_id = app_data[0], app_data[1]
    
    link_id = dpg.add_node_link(source_pin_id, target_pin_id, parent=sender)
    links[link_id] = (source_pin_id, target_pin_id)

    source_node_id = dpg.get_item_parent(source_pin_id)
    re_process_graph(source_node_id)

def delink_callback(sender, app_data):
    # Handles link deletion from the node editor.
    link_id = app_data 

    if link_id not in links:
        print(f"WARNING: Attempted to delink {link_id} which is not in our registry. Skipping.")
        dpg.delete_item(link_id) 
        return
        
    source_pin_id, target_pin_id = links[link_id]
    target_node_id = dpg.get_item_parent(target_pin_id)

    del links[link_id]
    dpg.delete_item(link_id)

    re_process_graph(target_node_id)

def open_file_dialog_callback(sender, app_data):
    # Callback for file dialog to load an image.
    global _current_dialog_node_id
    
    file_path = app_data['file_path_name']
    node_id = _current_dialog_node_id 
    _current_dialog_node_id = None 

    if not file_path:
        print("INFO: File selection cancelled or no file chosen.")
        unload_image_from_input_node(node_id)
        return
    
    width, height, dpg_data, numpy_data = load_image_to_dpg_texture(file_path)

    if dpg_data is not None and numpy_data is not None:
        node_outputs[node_id] = numpy_data 

        base_texture_name = f"input_texture_{node_id}"
        if node_id is not None and node_id in node_registry:
            pass 
        else:
            print(f"ERROR: node_id {node_id} not in node_registry.")

        re_process_graph(node_id)
    else:
        print(f"ERROR: Failed to load image data for {file_path}.")
        if node_id is not None:
            unload_image_from_input_node(node_id)
        re_process_graph(node_id) 

def save_output_file_dialog_callback(sender, app_data):
    # Callback for file dialog to save an image.
    global _current_dialog_node_id 

    node_id = _current_dialog_node_id 
    _current_dialog_node_id = None 

    if not app_data['file_path_name']:
        print("INFO: File save cancelled or no path chosen.")
        return

    file_path = app_data['file_path_name']

    output_image = node_outputs.get(node_id) 
    if output_image is not None:
        try:
            img_to_save = None
            if output_image.ndim == 2:
                img_to_save = Image.fromarray((output_image * 255).astype(np.uint8), mode='L') 
            elif output_image.ndim == 3:
                if output_image.shape[-1] == 3:
                    img_to_save = Image.fromarray((output_image * 255).astype(np.uint8), mode='RGB') 
                elif output_image.shape[-1] == 4:
                    img_to_save = Image.fromarray((output_image * 255).astype(np.uint8), mode='RGBA') 
                else:
                    print(f"ERROR: Unsupported image channel count for saving: {output_image.shape[-1]} channels.")
                    return
            else:
                print(f"ERROR: Unsupported image dimension for saving: {output_image.ndim} dimensions.")
                return

            if img_to_save:
                img_to_save.save(file_path)
                print(f"Image saved successfully to: {file_path}")
            else:
                print("ERROR: Image could not be prepared for saving.")

        except Exception as e:
            print(f"ERROR: Error saving image: {e}")
    else:
        print("INFO: No image available to save in the output node.")


def node_parameter_changed(sender, app_data, user_data):
    # Triggers graph reprocessing when a node parameter changes.
    node_id = user_data 
    re_process_graph(node_id)

def delete_selected_blocks(sender, app_data):
    # Deletes all currently selected nodes and their associated data.
    selected_nodes = dpg.get_selected_nodes("node_editor")
    
    if not selected_nodes:
        print("INFO: No nodes selected for deletion.")
        return

    nodes_to_reprocess_from = set() 

    for node_id in selected_nodes:
        if not dpg.does_item_exist(node_id):
            print(f"WARNING: Attempted to delete node {node_id} but it no longer exists in DPG.")
            continue 
        
        node_output_pin = node_registry.get(node_id, {}).get('output_pin_tag')
        if node_output_pin and dpg.does_item_exist(node_output_pin):
            for link_id, (source_pin, target_pin) in list(links.items()): 
                if source_pin == node_output_pin:
                    target_node = dpg.get_item_parent(target_pin)
                    if target_node and target_node not in selected_nodes and dpg.does_item_exist(target_node): 
                        nodes_to_reprocess_from.add(target_node)

        if node_id in node_registry:
            node_info = node_registry[node_id]

            del node_registry[node_id]
            if node_id in node_outputs: 
                del node_outputs[node_id]

        dpg.delete_item(node_id)
    
    keys_to_delete = []
    for link_id, (source_pin, target_pin) in list(links.items()): 
        if not (dpg.does_item_exist(source_pin) and dpg.does_item_exist(target_pin)):
            keys_to_delete.append(link_id)

    for link_id in keys_to_delete:
        del links[link_id]

    if nodes_to_reprocess_from:
        for node_id_to_reprocess in nodes_to_reprocess_from:
            if dpg.does_item_exist(node_id_to_reprocess):
                re_process_graph(node_id_to_reprocess)
            else:
                print(f"INFO: Downstream node {node_id_to_reprocess} no longer exists, skipping re-processing.")
    else:
        re_process_graph()

def unload_image_from_input_node(node_id):
    # Unloads the image data from the specified input node.
    if node_id in node_outputs:
        node_outputs[node_id] = None  
        print(f"INFO: Image data for node {node_id} cleared.")
    
    re_process_graph(node_id)

def delete_selected_links(sender, app_data):
    # Deletes all currently selected links and triggers graph reprocessing.
    selected_links = dpg.get_selected_links("node_editor")

    if not selected_links:
        print("INFO: No links selected for deletion.")
        return

    nodes_to_reprocess_from = set()

    for link_id in selected_links:
        if link_id in links:
            source_pin_id, target_pin_id = links[link_id]
            target_node_id = dpg.get_item_parent(target_pin_id)
            
            if target_node_id and dpg.does_item_exist(target_node_id):
                nodes_to_reprocess_from.add(target_node_id)
            
            del links[link_id]
            dpg.delete_item(link_id)
        else:
            print(f"WARNING: Attempted to delete link {link_id} not found in internal registry. Deleting from DPG if exists.")
            if dpg.does_item_exist(link_id):
                dpg.delete_item(link_id)

    if nodes_to_reprocess_from:
        for node_id_to_reprocess in nodes_to_reprocess_from:
            if dpg.does_item_exist(node_id_to_reprocess):
                re_process_graph(node_id_to_reprocess)
    else:
        re_process_graph()

def create_input_node(node_editor_id):
    # Creates an image input node with load/unload buttons.
    node_id = dpg.generate_uuid() 
    node_registry[node_id] = {'type': 'input_node'}

    with dpg.node(parent=node_editor_id, label="Input Image", tag=node_id):
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as output_pin_tag: 
            node_registry[node_id]['output_pin_tag'] = output_pin_tag 

            def _load_button_callback(s, a, node_id_to_pass):
                global _current_dialog_node_id
                _current_dialog_node_id = node_id_to_pass
                dpg.show_item("file_dialog_id")

            def _unload_button_callback(s, a, node_id_to_pass):
                unload_image_from_input_node(node_id_to_pass)
            
            dpg.add_button(label="Load Image", callback=lambda s, a: _load_button_callback(s, a, node_id))
            dpg.add_button(label="Unload Image", callback=lambda s, a: _unload_button_callback(s, a, node_id))
            
    return node_id

def create_gaussian_blur_node(node_editor_id):
    # Creates a Gaussian blur filter node.
    node_id = dpg.generate_uuid()
    node_registry[node_id] = {'type': 'gaussian_blur_node'}

    with dpg.node(parent=node_editor_id, label="Gaussian Blur", tag=node_id):
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as input_pin_tag:
            node_registry[node_id]['input_pin_tag'] = input_pin_tag 
            dpg.add_text("Image Input")
        
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
            sigma_slider = dpg.add_slider_float(label="Sigma", default_value=3.0, min_value=0.1, max_value=10.0,
                                                 callback=node_parameter_changed, user_data=node_id, width=150)
            node_registry[node_id]['sigma_slider'] = sigma_slider
            size_slider = dpg.add_slider_int(label="Kernel Size (odd)", default_value=5, min_value=1, max_value=21,
                                                 callback=node_parameter_changed, user_data=node_id, width=150)
            node_registry[node_id]['size_slider'] = size_slider
        
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as output_pin_tag:
            node_registry[node_id]['output_pin_tag'] = output_pin_tag 
            dpg.add_text("Image Output")
    return node_id

def create_median_filter_node(node_editor_id):
    # Creates a median filter node.
    node_id = dpg.generate_uuid()
    node_registry[node_id] = {'type': 'median_filter_node'}

    with dpg.node(parent=node_editor_id, label="Median Filter", tag=node_id):
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as input_pin_tag:
            node_registry[node_id]['input_pin_tag'] = input_pin_tag 
            dpg.add_text("Image Input")
        
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
            size_slider = dpg.add_slider_int(label="Size (odd)", default_value=7, min_value=1, max_value=21,
                                                 callback=node_parameter_changed, user_data=node_id, width=150)
            node_registry[node_id]['size_slider'] = size_slider
        
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as output_pin_tag:
            node_registry[node_id]['output_pin_tag'] = output_pin_tag 
            dpg.add_text("Image Output")
    return node_id

def create_hist_equalization_node(node_editor_id):
    # Creates a histogram equalization node.
    node_id = dpg.generate_uuid()
    node_registry[node_id] = {'type': 'hist_equalization_node'}

    with dpg.node(parent=node_editor_id, label="Histogram Equalization", tag=node_id):
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as input_pin_tag:
            node_registry[node_id]['input_pin_tag'] = input_pin_tag 
            dpg.add_text("Image Input")
        
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as output_pin_tag:
            node_registry[node_id]['output_pin_tag'] = output_pin_tag 
            dpg.add_text("Image Output")
    return node_id

def create_edge_detection_node(node_editor_id):
    # Creates an Edge Detection node.
    node_id = dpg.generate_uuid()
    node_registry[node_id] = {'type': 'edge_detection_node'}

    with dpg.node(parent=node_editor_id, label="Edge Detection", tag=node_id):
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as input_pin_tag:
            node_registry[node_id]['input_pin_tag'] = input_pin_tag 
            dpg.add_text("Image Input")
        
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
            force_slider = dpg.add_slider_float(label="Force", default_value=1.0, min_value=0.1, max_value=5.0,
                                                 callback=node_parameter_changed, user_data=node_id, width=150)
            node_registry[node_id]['force_slider'] = force_slider
        
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as output_pin_tag:
            node_registry[node_id]['output_pin_tag'] = output_pin_tag 
            dpg.add_text("Image Output")
    return node_id


def create_output_node(node_editor_id):
    # Creates an image output node with a save button.
    node_id = dpg.generate_uuid()
    node_registry[node_id] = {'type': 'output_node'}

    with dpg.node(parent=node_editor_id, label="Output Image", tag=node_id):
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as input_pin_tag:
            node_registry[node_id]['input_pin_tag'] = input_pin_tag 
            dpg.add_text("Image Input")
            
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
            def _save_button_callback(s, a, node_id_to_pass):
                global _current_dialog_node_id
                _current_dialog_node_id = node_id_to_pass
                dpg.show_item("save_file_dialog_id")

            dpg.add_button(label="Save Image", callback=lambda s, a: _save_button_callback(s, a, node_id))
    return node_id

def run_gui():
    # Sets up and runs the DearPyGui application for the image processing editor.
    dpg.create_context()

    with dpg.file_dialog(
        directory_selector=False, show=False, id="file_dialog_id", width=700, height=400,
        label="Select Image File", default_path=os.getcwd(), callback=open_file_dialog_callback
    ):
        dpg.add_file_extension(".png", custom_text="[PNG Image]")
        dpg.add_file_extension(".jpg", custom_text="[JPG Image]")
        dpg.add_file_extension(".jpeg", custom_text="[JPEG Image]")
        dpg.add_file_extension(".bmp", custom_text="[BMP Image]") 
        dpg.add_file_extension(".*", custom_text="[All Files]") 

    with dpg.file_dialog(
        directory_selector=False, show=False, id="save_file_dialog_id", width=700, height=400,
        label="Save Image As", default_path=os.getcwd(), callback=save_output_file_dialog_callback
    ):
        dpg.add_file_extension(".png", custom_text="[PNG Image]")
        dpg.add_file_extension(".jpg", custom_text="[JPG Image]")
        dpg.add_file_extension(".jpeg", custom_text="[JPEG Image]")
        dpg.add_file_extension(".bmp", custom_text="[BMP Image]")
        dpg.add_file_extension(".*", custom_text="[All Files]")

    with dpg.window(label="Graph-based Image Editor", tag="main_window", width=1200, height=800):
        with dpg.group(horizontal=True):
            with dpg.child_window(width=200, height=-1):
                dpg.add_text("Processing Blocks")
                dpg.add_separator()
                dpg.add_button(label="Input Image", callback=lambda: create_input_node("node_editor"), width=-1)
                dpg.add_button(label="Gaussian Blur", callback=lambda: create_gaussian_blur_node("node_editor"), width=-1)
                dpg.add_button(label="Median Filter", callback=lambda: create_median_filter_node("node_editor"), width=-1)
                dpg.add_button(label="Histogram Equalization", callback=lambda: create_hist_equalization_node("node_editor"), width=-1)
                dpg.add_button(label="Edge Detection", callback=lambda: create_edge_detection_node("node_editor"), width=-1)
                dpg.add_button(label="Output Image", callback=lambda: create_output_node("node_editor"), width=-1)
                dpg.add_separator()
                dpg.add_button(label="Delete Selected Blocks", callback=delete_selected_blocks, width=-1)
                dpg.add_button(label="Delete Selected Links", callback=delete_selected_links, width=-1) 
            
            with dpg.node_editor(callback=link_callback, delink_callback=delink_callback, tag="node_editor", width=-1, height=-1):
                pass

    dpg.create_viewport(title='Image Editor', width=1200, height=800)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    run_gui()
