# -*- coding: utf-8 -*-
import dearpygui.dearpygui as dpg 
import numpy as np
from PIL import Image
import os

# Import the ImageFilters class from your separate file
from image_filters import ImageFilters

# Global dictionaries to store application state
node_registry = {}  # Stores information about nodes (type, parameters, DPG tags)
node_outputs = {}   # Stores the resulting image data (numpy array) for each node
links = {}          # Stores connections between node pins (link_id -> (source_pin, target_pin))
global_texture_counter = 0 # Global counter for generating unique texture tags

# Global variable to temporarily hold the node_id for file dialogs
_current_dialog_node_id = None

# --- Helper functions for image processing (integrated with ImageFilters) ---

def load_image_to_dpg_texture(image_path):
    """
    Loads an image from a file and prepares it for display in DearPyGui as a texture.
    Converts the image to RGBA format and normalizes pixels to the [0, 1] range.
    """
    try:
        img = Image.open(image_path).convert("RGBA")
        width, height = img.size
        data_normalized = np.array(img, dtype=np.float32) / 255.0
        flattened_dpg_data = data_normalized.flatten()
        return width, height, flattened_dpg_data, data_normalized
    except Exception as e:
        print(f"Error loading image in load_image_to_dpg_texture: {e}")
        return None, None, None, None

def numpy_to_dpg_texture_data(numpy_image):
    """
    Converts a numpy array (image) to a flattened RGBA texture format accepted by DearPyGui.
    Assumes the numpy image is in the [0, 1] range.
    """
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
        print(f"Unsupported number of image channels for DPG texture conversion: {numpy_image.shape[-1]}")
        return None

    return numpy_image.flatten()

def update_dpg_texture(base_texture_name, width, height, data):
    """
    Creates a new dynamic texture with a unique tag each time to avoid "Alias already exists" errors.
    """
    global global_texture_counter
    global_texture_counter += 1
    
    new_texture_tag = f"{base_texture_name}_v{global_texture_counter}"
    
    with dpg.texture_registry(show=False):
        if dpg.does_item_exist(new_texture_tag):
            dpg.delete_item(new_texture_tag)
            print(f"DEBUG: Deleted existing texture {new_texture_tag} before recreation.")
        dpg.add_dynamic_texture(width=width, height=height, default_value=data, tag=new_texture_tag)
        print(f"DEBUG: Created NEW dynamic texture {new_texture_tag} with dimensions {width}x{height}.")
    
    return new_texture_tag


def apply_gaussian_blur(image_data, size, sigma):
    """
    Applies Gaussian blur to the image using the custom ImageFilters class.
    Handles both grayscale and color images by processing channels independently.
    Input image_data is float32 [0,1], output is float32 [0,1].
    """
    if image_data is None: return None
    
    # Convert input from float32 [0,1] to uint8 [0,255] for custom filters
    image_data_uint8 = (image_data * 255).astype(np.uint8)
    
    blurred_image_uint8 = np.zeros_like(image_data_uint8)
    
    if image_data_uint8.ndim == 3: # Color image (H, W, C)
        for c in range(image_data_uint8.shape[2]):
            blurred_image_uint8[:, :, c] = ImageFilters.gaussian_blur(image_data_uint8[:, :, c], size=size, sigma=sigma)
    else: # Grayscale image (H, W) or single channel
        blurred_image_uint8 = ImageFilters.gaussian_blur(image_data_uint8, size=size, sigma=sigma)
        
    # Convert output back to float32 [0,1]
    return blurred_image_uint8.astype(np.float32) / 255.0

def apply_median_filter(image_data, size):
    """
    Applies median filter to the image using the custom ImageFilters class.
    Handles both grayscale and color images by processing channels independently.
    Input image_data is float32 [0,1], output is float32 [0,1].
    """
    if image_data is None: return None
    
    # Convert input from float32 [0,1] to uint8 [0,255] for custom filters
    image_data_uint8 = (image_data * 255).astype(np.uint8)
    
    filtered_image_uint8 = np.zeros_like(image_data_uint8)
    
    if image_data_uint8.ndim == 3: # Color image (H, W, C)
        for c in range(image_data_uint8.shape[2]):
            filtered_image_uint8[:, :, c] = ImageFilters.median_filter(image_data_uint8[:, :, c], size=size)
    else: # Grayscale image (H, W) or single channel
        filtered_image_uint8 = ImageFilters.median_filter(image_data_uint8, size=size)
        
    # Convert output back to float32 [0,1]
    return filtered_image_uint8.astype(np.float32) / 255.0

def apply_histogram_equalization(image_data):
    """
    Applies histogram equalization to the image using the custom ImageFilters class.
    Handles both grayscale and color images by processing channels independently.
    Input image_data is float32 [0,1], output is float32 [0,1].
    """
    if image_data is None: return None
    
    # Convert input from float32 [0,1] to uint8 [0,255] for custom filters
    image_data_uint8 = (image_data * 255).astype(np.uint8)
    
    equalized_image_uint8 = np.zeros_like(image_data_uint8)
    
    if image_data_uint8.ndim == 3: # Color image (H, W, C)
        for c in range(image_data_uint8.shape[2]):
            equalized_image_uint8[:, :, c] = ImageFilters.histogram_equalization(image_data_uint8[:, :, c])
    else: # Grayscale image (H, W) or single channel
        equalized_image_uint8 = ImageFilters.histogram_equalization(image_data_uint8)
        
    # Convert output back to float32 [0,1]
    return equalized_image_uint8.astype(np.float32) / 255.0

def apply_edge_detection(image_data, force):
    """
    Applies edge detection to the image using the custom ImageFilters class.
    Handles both grayscale and color images by processing channels independently.
    Input image_data is float32 [0,1], output is float32 [0,1].
    """
    if image_data is None: return None
    
    # Convert input from float32 [0,1] to uint8 [0,255] for custom filters
    image_data_uint8 = (image_data * 255).astype(np.uint8)
    
    edged_image_uint8 = np.zeros_like(image_data_uint8)
    
    if image_data_uint8.ndim == 3: # Color image (H, W, C)
        for c in range(image_data_uint8.shape[2]):
            edged_image_uint8[:, :, c] = ImageFilters.edge_detection(image_data_uint8[:, :, c], force=force)
    else: # Grayscale image (H, W) or single channel
        edged_image_uint8 = ImageFilters.edge_detection(image_data_uint8, force=force)
        
    # Convert output back to float32 [0,1]
    return edged_image_uint8.astype(np.float32) / 255.0

# --- Graph logic and processing ---

def get_node_input_image(node_id, current_node_outputs):
    """Retrieves the input image for a given node by looking for a connected input pin."""
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
                print(f"WARNING: Link {link_id} has source pin {source_pin} that does not match "
                      f"expected output pin {source_output_pin_id} for source node {source_node_id}. Invalid link source.")
                return None

    return None

def process_node(node_id, current_node_outputs):
    """Processes the given node and updates its output."""
    node_type = node_registry[node_id]['type']
    
    effective_input_image = None
    if node_type == "input_node":
        effective_input_image = current_node_outputs.get(node_id) 
        print(f"DEBUG: Node {node_id} ({node_type}) input is from source. Shape: {effective_input_image.shape if effective_input_image is not None else 'None'}, Min: {np.min(effective_input_image) if effective_input_image is not None else 'N/A'}, Max: {np.max(effective_input_image) if effective_input_image is not None else 'N/A'}")
    else:
        effective_input_image = get_node_input_image(node_id, current_node_outputs)
        print(f"DEBUG: Node {node_id} ({node_type}) input from upstream. Shape: {effective_input_image.shape if effective_input_image is not None else 'None'}, Min: {np.min(effective_input_image) if effective_input_image is not None else 'N/A'}, Max: {np.max(effective_input_image) if effective_input_image is not None else 'N/A'}")

    processed_output_image = None

    if node_type == "input_node":
        processed_output_image = effective_input_image
    elif node_type == "gaussian_blur_node":
        if effective_input_image is not None:
            sigma = dpg.get_value(node_registry[node_id]['sigma_slider'])
            size = dpg.get_value(node_registry[node_id]['size_slider'])
            processed_output_image = apply_gaussian_blur(effective_input_image, size, sigma)
            print(f"DEBUG: Gaussian Blur applied. Output shape: {processed_output_image.shape if processed_output_image is not None else 'None'}, Min: {np.min(processed_output_image) if processed_output_image is not None else 'N/A'}, Max: {np.max(processed_output_image) if processed_output_image is not None else 'N/A'}")
    elif node_type == "median_filter_node":
        if effective_input_image is not None:
            size = dpg.get_value(node_registry[node_id]['size_slider'])
            processed_output_image = apply_median_filter(effective_input_image, size)
            print(f"DEBUG: Median Filter applied. Output shape: {processed_output_image.shape if processed_output_image is not None else 'None'}, Min: {np.min(processed_output_image) if processed_output_image is not None else 'N/A'}, Max: {np.max(processed_output_image) if processed_output_image is not None else 'N/A'}")
    elif node_type == "hist_equalization_node":
        if effective_input_image is not None:
            processed_output_image = apply_histogram_equalization(effective_input_image)
            print(f"DEBUG: Hist Equalization applied. Output shape: {processed_output_image.shape if processed_output_image is not None else 'None'}, Min: {np.min(processed_output_image) if processed_output_image is not None else 'N/A'}, Max: {np.max(processed_output_image) if processed_output_image is not None else 'N/A'}")
    elif node_type == "edge_detection_node": # New edge detection node type
        if effective_input_image is not None:
            force = dpg.get_value(node_registry[node_id]['force_slider'])
            processed_output_image = apply_edge_detection(effective_input_image, force)
            print(f"DEBUG: Edge Detection applied. Output shape: {processed_output_image.shape if processed_output_image is not None else 'None'}, Min: {np.min(processed_output_image) if processed_output_image is not None else 'N/A'}, Max: {np.max(processed_output_image) if processed_output_image is not None else 'N/A'}")
    elif node_type == "output_node":
        processed_output_image = effective_input_image 
        
        # Special handling for output node: update its displayed image
        image_item_tag = node_registry[node_id].get('image_item_tag')
        base_texture_name = f"output_texture_{node_id}" # Base name for new texture tag
        image_container_tag = node_registry[node_id].get('image_container_tag') 

        if not (image_container_tag and dpg.does_item_exist(image_container_tag)):
            print(f"CRITICAL ERROR: Image container {image_container_tag} for output node {node_id} is missing. "
                  "This should not happen if node creation is correct. Cannot update display.")
            return
        
        if not (image_item_tag and dpg.does_item_exist(image_item_tag)):
            print(f"DEBUG: Output image item {image_item_tag} for node {node_id} does not exist. Recreating.")
            try:
                temp_texture_tag = f"temp_placeholder_texture_{node_id}"
                with dpg.texture_registry(show=False):
                    if not dpg.does_item_exist(temp_texture_tag):
                        dpg.add_dynamic_texture(width=1, height=1, default_value=np.zeros(4, dtype=np.float32), tag=temp_texture_tag)

                image_item_tag = dpg.add_image(temp_texture_tag, width=200, height=200, parent=image_container_tag)
                node_registry[node_id]['image_item_tag'] = image_item_tag 
            except Exception as e:
                print(f"CRITICAL ERROR during output image item recreation for node {node_id}: {e}")
                return 

        if processed_output_image is not None:
            height, width = processed_output_image.shape[0], processed_output_image.shape[1]
            dpg_data = numpy_to_dpg_texture_data(processed_output_image)
            
            if dpg_data is not None:
                new_actual_texture_tag = update_dpg_texture(base_texture_name, width, height, dpg_data)
                dpg.set_item_width(image_item_tag, width)
                dpg.set_item_height(image_item_tag, height)
                dpg.set_item_source(image_item_tag, new_actual_texture_tag)
                
                dpg.set_item_width(image_container_tag, width)
                dpg.set_item_height(image_container_tag, height)
                node_registry[node_id]['texture_tag'] = new_actual_texture_tag
            else:
                print(f"WARNING: Failed to convert processed image to DPG texture data for node {node_id}.")
                new_actual_texture_tag = update_dpg_texture(base_texture_name, 1, 1, np.zeros(4, dtype=np.float32))
                dpg.set_item_width(image_item_tag, 200) 
                dpg.set_item_height(image_item_tag, 200)
                dpg.set_item_source(image_item_tag, new_actual_texture_tag)
                dpg.set_item_width(image_container_tag, 200)
                dpg.set_item_height(image_container_tag, 200)
                node_registry[node_id]['texture_tag'] = new_actual_texture_tag

        else:
            print(f"DEBUG: Output node {node_id} received None image. Clearing display.")
            new_actual_texture_tag = update_dpg_texture(base_texture_name, 1, 1, np.zeros(4, dtype=np.float32)) 
            dpg.set_item_width(image_item_tag, 200) 
            dpg.set_item_height(image_item_tag, 200)
            dpg.set_item_source(image_item_tag, new_actual_texture_tag)
            dpg.set_item_width(image_container_tag, 200)
            dpg.set_item_height(image_container_tag, 200)
            node_registry[node_id]['texture_tag'] = new_actual_texture_tag
    
    current_node_outputs[node_id] = processed_output_image


def re_process_graph(start_node_id=None):
    """
    Reprocesses the graph, starting from a specified node (or all if none).
    Performs a full topological sort to ensure correct processing order.
    """
    print(f"DEBUG: Calling re_process_graph. Triggered by: {start_node_id if start_node_id else 'None (full graph)'}")

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
                print(f"WARNING: Link {link_id} involves non-existent node(s) (source: {source_node}, target: {target_node}). Skipping.")

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
        print(f"WARNING: Topological sort did not process all nodes. Possible cycle or disconnected nodes. "
              f"Processed: {len(processed_nodes_in_order)}, Total: {len(node_registry)}")
        remaining_nodes = [node_id for node_id in node_registry if node_id not in processed_nodes_in_order]
        processed_nodes_in_order.extend(remaining_nodes)

    print(f"DEBUG: Final processing order: {processed_nodes_in_order}")
    for node_id in processed_nodes_in_order:
        if not dpg.does_item_exist(node_id):
            print(f"WARNING: Node {node_id} no longer exists in DPG. Skipping processing.")
            continue
        
        try:
            process_node(node_id, node_outputs)
        except Exception as e:
            print(f"Error processing node {node_id}: {e}")

# --- DearPyGui Callbacks ---

def link_callback(sender, app_data):
    """Called when a new link is added between pins."""
    source_pin_id, target_pin_id = app_data[0], app_data[1]
    
    link_id = dpg.add_node_link(source_pin_id, target_pin_id, parent=sender)
    links[link_id] = (source_pin_id, target_pin_id)

    source_node_id = dpg.get_item_parent(source_pin_id)
    re_process_graph(source_node_id)

def delink_callback(sender, app_data):
    """Called when a link is removed between pins."""
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
    """
    Called after a file is selected in the file dialog for an input node.
    Loads the image and triggers graph reprocessing.
    """
    global _current_dialog_node_id
    
    file_path = app_data['file_path_name']
    node_id = _current_dialog_node_id 
    _current_dialog_node_id = None 

    if not file_path:
        print("INFO: File selection cancelled or no file chosen.")
        if node_id is not None and node_id in node_outputs:
            node_outputs[node_id] = None
            if node_id in node_registry and 'image_item_tag' in node_registry[node_id]:
                image_item_tag = node_registry[node_id]['image_item_tag']
                base_texture_name = f"input_texture_{node_id}"
                if dpg.does_item_exist(image_item_tag):
                    new_actual_texture_tag = update_dpg_texture(base_texture_name, 1, 1, np.zeros(4, dtype=np.float32))
                    dpg.set_item_width(image_item_tag, 200) 
                    dpg.set_item_height(image_item_tag, 200)
                    dpg.set_item_source(image_item_tag, new_actual_texture_tag)
                    node_registry[node_id]['texture_tag'] = new_actual_texture_tag
            re_process_graph(node_id)
        return

    print(f"DEBUG: open_file_dialog_callback called. sender: {sender}, app_data: {app_data}, node_id: {node_id}")
    print(f"DEBUG: Attempting to load file: {file_path} for input node {node_id}")
    
    width, height, dpg_data, numpy_data = load_image_to_dpg_texture(file_path)

    if dpg_data is not None and numpy_data is not None:
        print(f"DEBUG: Image {file_path} loaded successfully. Dims: {width}x{height}, numpy_data shape: {numpy_data.shape}")
        node_outputs[node_id] = numpy_data 

        base_texture_name = f"input_texture_{node_id}"
        if node_id is not None and node_id in node_registry and 'image_item_tag' in node_registry[node_id]:
            image_item_tag = node_registry[node_id]['image_item_tag']
            
            if dpg.does_item_exist(image_item_tag):
                new_actual_texture_tag = update_dpg_texture(base_texture_name, width, height, dpg_data)
                dpg.set_item_width(image_item_tag, width)
                dpg.set_item_height(image_item_tag, height)
                dpg.set_item_source(image_item_tag, new_actual_texture_tag) 
                node_registry[node_id]['texture_tag'] = new_actual_texture_tag
            else:
                print(f"ERROR: Input node {node_id} image item {image_item_tag} does not exist when trying to update display. "
                      "This indicates a GUI creation issue or item deletion.")
        else:
            print(f"ERROR: node_id {node_id} not in node_registry or missing 'image_item_tag'. Cannot update input node display.")

        re_process_graph(node_id)
    else:
        print(f"ERROR: Failed to load image data for {file_path}. `load_image_to_dpg_texture` returned None values "
              "(likely due to unsupported file format or corruption).")
        if node_id is not None:
            node_outputs[node_id] = None
            if node_id in node_registry and 'image_item_tag' in node_registry[node_id]:
                image_item_tag = node_registry[node_id]['image_item_tag']
                base_texture_name = f"input_texture_{node_id}"
                if dpg.does_item_exist(image_item_tag):
                    new_actual_texture_tag = update_dpg_texture(base_texture_name, 1, 1, np.zeros(4, dtype=np.float32))
                    dpg.set_item_width(image_item_tag, 200) 
                    dpg.set_item_height(image_item_tag, 200)
                    dpg.set_item_source(image_item_tag, new_actual_texture_tag)
                    node_registry[node_id]['texture_tag'] = new_actual_texture_tag
        re_process_graph(node_id) 

def save_output_file_dialog_callback(sender, app_data):
    """
    Called after a file save path is selected in the file dialog.
    Saves the processed image from the specified output node.
    """
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
                print("ERROR: Image could not be prepared for saving (reason unknown).")

        except Exception as e:
            print(f"ERROR: Error saving image: {e}")
    else:
        print("INFO: No image available to save in the output node. Connect an image source.")


def node_parameter_changed(sender, app_data, user_data):
    """
    Called when a node parameter (e.g., slider) is changed.
    Triggers graph reprocessing from the node whose parameter changed.
    """
    node_id = user_data 
    re_process_graph(node_id)

# --- Node Creation Functions ---

def create_input_node(node_editor_id):
    """
    Creates an image input node. This node allows users to load an image
    from a file and outputs the image data.
    """
    node_id = dpg.generate_uuid() 
    node_registry[node_id] = {'type': 'input_node'}

    with dpg.node(parent=node_editor_id, label="Obraz wejściowy", tag=node_id):
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as output_pin_tag: 
            node_registry[node_id]['output_pin_tag'] = output_pin_tag 

            def _load_button_callback(s, a, node_id_to_pass):
                global _current_dialog_node_id
                _current_dialog_node_id = node_id_to_pass
                dpg.show_item("file_dialog_id")
            
            dpg.add_button(label="Wczytaj obraz", callback=lambda s, a: _load_button_callback(s, a, node_id))
            
            base_texture_name = f"input_texture_{node_id}"
            placeholder_data = np.zeros(4, dtype=np.float32) 
            initial_texture_id = update_dpg_texture(base_texture_name, 1, 1, placeholder_data)

            image_item_tag = dpg.add_image(initial_texture_id, width=200, height=200) 
            node_registry[node_id]['image_item_tag'] = image_item_tag
            node_registry[node_id]['texture_tag'] = initial_texture_id 
    return node_id

def create_gaussian_blur_node(node_editor_id):
    """
    Creates a Gaussian blur node. This node applies a Gaussian blur filter
    to the input image based on the 'sigma' and 'size' parameters.
    """
    node_id = dpg.generate_uuid()
    node_registry[node_id] = {'type': 'gaussian_blur_node'}

    with dpg.node(parent=node_editor_id, label="Rozmycie Gaussa", tag=node_id):
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as input_pin_tag:
            node_registry[node_id]['input_pin_tag'] = input_pin_tag 
            dpg.add_text("Wejście obrazu")
        
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
            sigma_slider = dpg.add_slider_float(label="Sigma", default_value=3.0, min_value=0.1, max_value=10.0,
                                                 callback=node_parameter_changed, user_data=node_id, width=150)
            node_registry[node_id]['sigma_slider'] = sigma_slider
            size_slider = dpg.add_slider_int(label="Rozmiar jądra (nieparzysty)", default_value=5, min_value=1, max_value=21,
                                                 callback=node_parameter_changed, user_data=node_id, width=150)
            node_registry[node_id]['size_slider'] = size_slider
        
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as output_pin_tag:
            node_registry[node_id]['output_pin_tag'] = output_pin_tag 
            dpg.add_text("Wyjście obrazu")
    return node_id

def create_median_filter_node(node_editor_id):
    """
    Creates a median filter node. This node applies a median filter
    to the input image based on the 'size' parameter.
    """
    node_id = dpg.generate_uuid()
    node_registry[node_id] = {'type': 'median_filter_node'}

    with dpg.node(parent=node_editor_id, label="Filtr Medianowy", tag=node_id):
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as input_pin_tag:
            node_registry[node_id]['input_pin_tag'] = input_pin_tag 
            dpg.add_text("Wejście obrazu")
        
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
            size_slider = dpg.add_slider_int(label="Rozmiar (nieparzysty)", default_value=7, min_value=1, max_value=21,
                                             callback=node_parameter_changed, user_data=node_id, width=150)
            node_registry[node_id]['size_slider'] = size_slider
        
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as output_pin_tag:
            node_registry[node_id]['output_pin_tag'] = output_pin_tag 
            dpg.add_text("Wyjście obrazu")
    return node_id

def create_hist_equalization_node(node_editor_id):
    """
    Creates a histogram equalization node. This node enhances the contrast
    of the input image by distributing the intensity values.
    """
    node_id = dpg.generate_uuid()
    node_registry[node_id] = {'type': 'hist_equalization_node'}

    with dpg.node(parent=node_editor_id, label="Wyrównanie Histogramu", tag=node_id):
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as input_pin_tag:
            node_registry[node_id]['input_pin_tag'] = input_pin_tag 
            dpg.add_text("Wejście obrazu")
        
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as output_pin_tag:
            node_registry[node_id]['output_pin_tag'] = output_pin_tag 
            dpg.add_text("Wyjście obrazu")
    return node_id

def create_edge_detection_node(node_editor_id):
    """
    Creates an Edge Detection node. This node applies an edge detection filter
    to the input image based on the 'force' parameter.
    """
    node_id = dpg.generate_uuid()
    node_registry[node_id] = {'type': 'edge_detection_node'}

    with dpg.node(parent=node_editor_id, label="Wykrywanie Krawędzi", tag=node_id):
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as input_pin_tag:
            node_registry[node_id]['input_pin_tag'] = input_pin_tag 
            dpg.add_text("Wejście obrazu")
        
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
            # Slider for the 'force' parameter
            force_slider = dpg.add_slider_float(label="Siła", default_value=1.0, min_value=0.1, max_value=5.0,
                                                 callback=node_parameter_changed, user_data=node_id, width=150)
            node_registry[node_id]['force_slider'] = force_slider
        
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as output_pin_tag:
            node_registry[node_id]['output_pin_tag'] = output_pin_tag 
            dpg.add_text("Wyjście obrazu")
    return node_id


def create_output_node(node_editor_id):
    """
    Creates an image output node. This node displays the final processed image
    and allows saving it to a file.
    """
    node_id = dpg.generate_uuid()
    node_registry[node_id] = {'type': 'output_node'}

    with dpg.node(parent=node_editor_id, label="Obraz wyjściowy", tag=node_id):
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as input_pin_tag:
            node_registry[node_id]['input_pin_tag'] = input_pin_tag 
            dpg.add_text("Wejście obrazu")
            
            image_container_tag = dpg.add_child_window(width=200, height=200, border=False, parent=input_pin_tag)
            node_registry[node_id]['image_container_tag'] = image_container_tag 

            base_texture_name = f"output_texture_{node_id}" 
            placeholder_data = np.zeros(4, dtype=np.float32) 
            initial_texture_id = update_dpg_texture(base_texture_name, 1, 1, placeholder_data)

            image_item_tag = dpg.add_image(initial_texture_id, width=200, height=200, parent=image_container_tag)
            node_registry[node_id]['image_item_tag'] = image_item_tag
            node_registry[node_id]['texture_tag'] = initial_texture_id 

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
            def _save_button_callback(s, a, node_id_to_pass):
                global _current_dialog_node_id
                _current_dialog_node_id = node_id_to_pass
                dpg.show_item("save_file_dialog_id")

            dpg.add_button(label="Zapisz obraz", callback=lambda s, a: _save_button_callback(s, a, node_id))
    return node_id

# --- Main GUI Configuration ---

def run_gui():
    """
    Sets up and runs the DearPyGui application for the image processing editor.
    """
    dpg.create_context()

    with dpg.file_dialog(
        directory_selector=False, show=False, id="file_dialog_id", width=700, height=400,
        label="Wybierz plik obrazu", default_path=os.getcwd(), callback=open_file_dialog_callback
    ):
        dpg.add_file_extension(".png", custom_text="[Obraz PNG]")
        dpg.add_file_extension(".jpg", custom_text="[Obraz JPG]")
        dpg.add_file_extension(".jpeg", custom_text="[Obraz JPEG]")
        dpg.add_file_extension(".bmp", custom_text="[Obraz BMP]") 
        dpg.add_file_extension(".*", custom_text="[Wszystkie pliki]") 

    with dpg.file_dialog(
        directory_selector=False, show=False, id="save_file_dialog_id", width=700, height=400,
        label="Zapisz obraz jako", default_path=os.getcwd(), callback=save_output_file_dialog_callback
    ):
        dpg.add_file_extension(".png", custom_text="[Obraz PNG]")
        dpg.add_file_extension(".jpg", custom_text="[Obraz JPG]")
        dpg.add_file_extension(".jpeg", custom_text="[Obraz JPEG]")
        dpg.add_file_extension(".bmp", custom_text="[Obraz BMP]")
        dpg.add_file_extension(".*", custom_text="[Wszystkie pliki]")

    with dpg.window(label="Edytor obrazów sterowany grafem", tag="main_window", width=1200, height=800):
        with dpg.group(horizontal=True):
            with dpg.child_window(width=200, height=-1):
                dpg.add_text("Bloczki przetwarzania")
                dpg.add_separator()
                dpg.add_button(label="Obraz wejściowy", callback=lambda: create_input_node("node_editor"), width=-1)
                dpg.add_button(label="Rozmycie Gaussa", callback=lambda: create_gaussian_blur_node("node_editor"), width=-1)
                dpg.add_button(label="Filtr Medianowy", callback=lambda: create_median_filter_node("node_editor"), width=-1)
                dpg.add_button(label="Wyrównanie Histogramu", callback=lambda: create_hist_equalization_node("node_editor"), width=-1)
                # New button for Edge Detection node
                dpg.add_button(label="Wykrywanie Krawędzi", callback=lambda: create_edge_detection_node("node_editor"), width=-1)
                dpg.add_button(label="Obraz wyjściowy", callback=lambda: create_output_node("node_editor"), width=-1)
            
            with dpg.node_editor(callback=link_callback, delink_callback=delink_callback, tag="node_editor", width=-1, height=-1):
                pass

    dpg.create_viewport(title='Edytor obrazów', width=1200, height=800)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    dpg.show_debug()
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    run_gui()
