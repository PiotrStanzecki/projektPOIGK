import dearpygui.dearpygui as dpg
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import os
import platform 

# --- Kontekst DPG ---
dpg.create_context()

# --- Wypisanie informacji o wersji DPG i systemie operacyjnym (dla debugowania) ---
print(f"OS: {platform.platform()}")

# --- Liczniki i zmienne globalne ---
node_counter = [0]
link_counter = [0]
link_ids = []
node_data_store = {}
node_outputs = {}
node_inputs_status = {}

current_input_node_for_load = None 

# --- Funkcje przetwarzania obrazu (przystosowane do Dear PyGui) ---

def process_gaussian_blur(image_data, std_dev=1.0):
    if image_data is None: return None
    if not np.issubdtype(image_data.dtype, np.uint8):
        image_data = (image_data.clip(0, 255)).astype(np.uint8)
    if image_data.ndim == 2:
        image_data = np.stack([image_data]*3, axis=-1) 
    img_pil = Image.fromarray(image_data)
    blurred_img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=std_dev))
    return np.array(blurred_img_pil)

def process_median_filter(image_data):
    if image_data is None: return None
    if not np.issubdtype(image_data.dtype, np.uint8):
        image_data = (image_data.clip(0, 255)).astype(np.uint8)
    if image_data.ndim == 2:
        image_data = np.stack([image_data]*3, axis=-1) 
    img_pil = Image.fromarray(image_data)
    median_img_pil = img_pil.filter(ImageFilter.MedianFilter(size=3))
    return np.array(median_img_pil)

def process_histogram_equalization(image_data):
    if image_data is None: return None
    if not np.issubdtype(image_data.dtype, np.uint8):
        image_data = (image_data.clip(0, 255)).astype(np.uint8)
    img_pil = Image.fromarray(image_data)
    if img_pil.mode != 'L':
        img_pil = img_pil.convert('L')
    equalized_img_pil = ImageOps.equalize(img_pil)
    # Ensure 3 channels for display consistency if it was originally grayscale
    return np.stack([np.array(equalized_img_pil)]*3, axis=-1) 

def process_grayscale(image_data):
    if image_data is None: return None
    if not np.issubdtype(image_data.dtype, np.uint8):
        image_data = (image_data.clip(0, 255)).astype(np.uint8)
    img_pil = Image.fromarray(image_data)
    gray_img_pil = img_pil.convert('L')
    # Return as 3 channels for consistent display in DPG
    return np.stack([np.array(gray_img_pil)]*3, axis=-1) 

def process_invert(image_data):
    if image_data is None: return None
    if not np.issubdtype(image_data.dtype, np.uint8):
        image_data = (image_data.clip(0, 255)).astype(np.uint8)
    if image_data.ndim == 2:
        image_data = np.stack([image_data]*3, axis=-1) 
    img_pil = Image.fromarray(image_data)
    inverted_img_pil = ImageOps.invert(img_pil)
    return np.array(inverted_img_pil)

# --- Poprawiona funkcja ładowania tekstury z danych numpy ---
def load_texture_from_image_data(np_image_data, target_node_id):
    print(f"DEBUG: Entering load_texture_from_image_data for node {target_node_id}")
    if np_image_data is None or np_image_data.size == 0:
        print(f"DEBUG: load_texture_from_image_data - np_image_data is None or empty for node {target_node_id}")
        return None, 0, 0
    
    try:
        if not np.issubdtype(np_image_data.dtype, np.uint8):
            np_image_data = (np_image_data.clip(0, 255)).astype(np.uint8)

        # Ensure image has enough channels for RGBA conversion if it's grayscale
        if np_image_data.ndim == 2:
            img_pil = Image.fromarray(np_image_data, mode='L')
        elif np_image_data.ndim == 3 and np_image_data.shape[2] == 3:
            img_pil = Image.fromarray(np_image_data, mode='RGB')
        elif np_image_data.ndim == 3 and np_image_data.shape[2] == 4:
            img_pil = Image.fromarray(np_image_data, mode='RGBA')
        else:
            print(f"ERROR: Unsupported numpy image format for node {target_node_id}. Shape: {np_image_data.shape}")
            return None, 0, 0

        if img_pil.mode != 'RGBA':
            img_pil = img_pil.convert('RGBA')

        data = np.array(img_pil, dtype=np.float32) / 255.0
        data = data.flatten().tolist() 

        width, height = img_pil.size

        texture_tag = f"texture_{target_node_id}"
        
        if dpg.does_item_exist(texture_tag):
            print(f"DEBUG: Deleting existing texture '{texture_tag}' for node {target_node_id}")
            dpg.delete_item(texture_tag)

        with dpg.texture_registry(show=False):
            dpg.add_static_texture(width=width, height=height, default_value=data, tag=texture_tag)
        
        print(f"DEBUG: Texture '{texture_tag}' created/updated successfully for node {target_node_id}. Size: {width}x{height}")
        return texture_tag, width, height
    except Exception as e:
        print(f"ERROR: load_texture_from_image_data for node {target_node_id}: {e}")
        return None, 0, 0

def display_image_in_node(node_id, np_image_data):
    print(f"DEBUG: Entering display_image_in_node for node {node_id}")
    image_attr_id = f"{node_id}_image_attr"
    
    # Always try to delete the old image display and placeholder first
    if dpg.does_item_exist(f"{node_id}_image_display"):
        print(f"DEBUG: Deleting old image display for node {node_id}")
        dpg.delete_item(f"{node_id}_image_display")
    
    if dpg.does_item_exist(f"{node_id}_image_display_placeholder"):
        print(f"DEBUG: Deleting old placeholder for node {node_id}")
        dpg.delete_item(f"{node_id}_image_display_placeholder")

    if np_image_data is None or np_image_data.size == 0: # Added np_image_data.size == 0 check
        print(f"DEBUG: display_image_in_node - np_image_data is None or empty for node {node_id}. Clearing display.")
        texture_tag_to_delete = f"texture_{node_id}"
        if dpg.does_item_exist(texture_tag_to_delete):
            print(f"DEBUG: Deleting associated texture '{texture_tag_to_delete}' for node {node_id}")
            dpg.delete_item(texture_tag_to_delete)
        
        if dpg.does_item_exist(image_attr_id):
            dpg.add_spacer(height=100, tag=f"{node_id}_image_display_placeholder", parent=image_attr_id)
            print(f"DEBUG: Added placeholder for node {node_id}")
        return

    texture_tag, width, height = load_texture_from_image_data(np_image_data, node_id)

    if not texture_tag:
        print(f"DEBUG: display_image_in_node - Failed to create texture for node '{node_id}'. Not displaying image.")
        if dpg.does_item_exist(image_attr_id):
            dpg.add_spacer(height=100, tag=f"{node_id}_image_display_placeholder", parent=image_attr_id)
            print(f"DEBUG: Added placeholder after texture creation failure for node {node_id}")
        return

    max_node_width = 180
    max_node_height = 130

    original_height, original_width = np_image_data.shape[0], np_image_data.shape[1]
    
    scale_factor_w = max_node_width / original_width
    scale_factor_h = max_node_height / original_height
    scale_factor = min(scale_factor_w, scale_factor_h)

    display_width = int(original_width * scale_factor)
    display_height = int(original_height * scale_factor)

    if dpg.does_item_exist(image_attr_id):
        dpg.add_image(texture_tag, width=display_width, height=display_height, parent=image_attr_id, tag=f"{node_id}_image_display")
        print(f"DEBUG: Image displayed for node {node_id} using texture {texture_tag}. Size: {display_width}x{display_height}")
    else:
        print(f"ERROR: display_image_in_node - Image attribute '{image_attr_id}' not found for node '{node_id}'.")


# --- Callbacks dla Dear PyGui ---

def node_editor_callback(sender, app_data):
    print(f"DEBUG: node_editor_callback received app_data: {app_data}, type: {type(app_data)}")
    if isinstance(app_data, tuple) and len(app_data) == 2: # Link created
        input_attr_id, output_attr_id = app_data
        
        link_id = f"link_{link_counter[0]}"
        link_counter[0] += 1
        dpg.add_node_link(input_attr_id, output_attr_id, parent="node_editor", tag=link_id)
        link_ids.append(link_id)
        print(f"DEBUG: Link created: {output_attr_id} -> {input_attr_id}")
        start_processing_graph() 
    elif isinstance(app_data, list): 
        node_selection_callback(sender, app_data) 

def delink_callback(sender, app_data):
    print(f"DEBUG: delink_callback called for link: {app_data}")
    if app_data in link_ids and dpg.does_item_exist(app_data):
        print(f"DEBUG: Deleting link: {app_data}")
        link_details = dpg.get_item_configuration(app_data)
        input_attr_id = link_details.get('target') 
        dpg.delete_item(app_data)
        link_ids.remove(app_data)
        
        if input_attr_id: 
            parent_node_id = dpg.get_item_parent(input_attr_id)
            if isinstance(parent_node_id, int): 
                parent_node_id = dpg.get_item_alias(parent_node_id)
            if parent_node_id and parent_node_id in node_inputs_status:
                node_inputs_status[parent_node_id] = False 
                clear_downstream_data(parent_node_id) 
        start_processing_graph() 

def delete_node_callback(sender, app_data, user_data):
    node_id_to_delete = user_data
    print(f"DEBUG: delete_node_callback called for node: {node_id_to_delete}")
    
    links_to_remove_on_delete = []
    for link_id in link_ids:
        if dpg.does_item_exist(link_id):
            link_details = dpg.get_item_configuration(link_id)
            input_attr_id = link_details.get('target') 
            output_attr_id = link_details.get('source') 
            
            input_node_from_attr = dpg.get_item_parent(input_attr_id) if dpg.does_item_exist(input_attr_id) else None
            output_node_from_attr = dpg.get_item_parent(output_attr_id) if dpg.does_item_exist(output_attr_id) else None
            
            if isinstance(input_node_from_attr, int):
                input_node_from_attr = dpg.get_item_alias(input_node_from_attr)
            if isinstance(output_node_from_attr, int):
                output_node_from_attr = dpg.get_item_alias(output_node_from_attr)

            if input_attr_id and output_attr_id and dpg.does_item_exist(input_attr_id) and dpg.does_item_exist(output_attr_id):
                if input_node_from_attr == node_id_to_delete or \
                   output_node_from_attr == node_id_to_delete:
                    links_to_remove_on_delete.append(link_id)
    
    for link_id in links_to_remove_on_delete:
        delink_callback(None, link_id) 

    if node_id_to_delete in node_data_store:
        texture_tag = f"texture_{node_id_to_delete}"
        if dpg.does_item_exist(texture_tag):
            print(f"DEBUG: Deleting texture '{texture_tag}' associated with deleted node {node_id_to_delete}")
            dpg.delete_item(texture_tag)
        print(f"DEBUG: Deleting node data for {node_id_to_delete} from node_data_store.")
        del node_data_store[node_id_to_delete]
    if node_id_to_delete in node_outputs:
        print(f"DEBUG: Deleting node output for {node_id_to_delete}.")
        del node_outputs[node_id_to_delete]
    if node_id_to_delete in node_inputs_status:
        print(f"DEBUG: Deleting node input status for {node_id_to_delete}.")
        del node_inputs_status[node_id_to_delete]

    print(f"DEBUG: Deleting DPG item for node: {node_id_to_delete}")
    dpg.delete_item(node_id_to_delete)
    print(f"Node deleted: {node_id_to_delete}")
    
    if dpg.does_item_exist("gaussian_std_dev_slider_group"):
        dpg.hide_item("gaussian_std_dev_slider_group")
    
    start_processing_graph() 

def node_selection_callback(sender, app_data):
    global current_input_node_for_load
    selected_nodes_dpg_ids = app_data 
    print(f"DEBUG: node_selection_callback called with selected_nodes_dpg_ids: {selected_nodes_dpg_ids}")
    
    if dpg.does_item_exist("gaussian_std_dev_slider_group"):
        dpg.hide_item("gaussian_std_dev_slider_group")

    current_input_node_for_load = None 

    if selected_nodes_dpg_ids:
        selected_dpg_id = selected_nodes_dpg_ids[0] 
        selected_node_tag = dpg.get_item_alias(selected_dpg_id) 
        print(f"DEBUG: First selected node DPG ID: {selected_dpg_id}, Tag: {selected_node_tag}")

        if selected_node_tag in node_data_store:
            node_label = node_data_store[selected_node_tag]["label"]
            print(f"DEBUG: Selected node label: {node_label} (Tag: {selected_node_tag})")
            
            if node_label == "Gaussian Blur":
                if dpg.does_item_exist("gaussian_std_dev_slider_group"):
                    dpg.show_item("gaussian_std_dev_slider_group")
                    current_std_dev = node_data_store[selected_node_tag].get("gaussian_std_dev", 1.0)
                    dpg.set_value("gaussian_std_dev_slider", current_std_dev)
            elif node_label == "Input IMG":
                current_input_node_for_load = selected_node_tag
                print(f"DEBUG: current_input_node_for_load set to: {current_input_node_for_load}")
        else:
            print(f"ERROR: Selected node tag '{selected_node_tag}' (DPG ID: {selected_dpg_id}) not found in node_data_store. This shouldn't happen.")
    else:
        print("DEBUG: No nodes selected. current_input_node_for_load is None.")

def create_node(label, inputs=1, outputs=1, pos=(100, 100), node_type="filter"):
    node_id = f"node_{node_counter[0]}"
    node_counter[0] += 1
    
    node_data_store[node_id] = {
        "label": label,
        "type": node_type,
        "input_attr_id": [],
        "output_attr_id": [],
        "image_data": None, 
        "filter_func": None, 
        "slider_value": 1.0 
    }
    node_inputs_status[node_id] = False 

    with dpg.node(label=label, parent="node_editor", pos=pos, tag=node_id):
        for i in range(inputs):
            attr_id = f"{node_id}_input_{i}"
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, tag=attr_id):
                dpg.add_text(f"Input {i+1}" if inputs > 1 else "") 
            node_data_store[node_id]["input_attr_id"].append(attr_id)
        
        if node_type == "image":
            image_attr_id = f"{node_id}_image_attr"
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static, tag=image_attr_id): 
                dpg.add_spacer(height=5) 
                dpg.add_spacer(height=100, tag=f"{node_id}_image_display_placeholder")
        
        for i in range(outputs):
            attr_id = f"{node_id}_output_{i}"
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, tag=attr_id):
                dpg.add_text(f"Output {i+1}" if outputs > 1 else "")
            node_data_store[node_id]["output_attr_id"].append(attr_id)

    with dpg.popup(parent=node_id, mousebutton=dpg.mvMouseButton_Right):
        dpg.add_button(label="Delete Node", callback=delete_node_callback, user_data=node_id)
    
    print(f"DEBUG: Created node: {label} (ID: {node_id}). node_data_store check: {node_id in node_data_store}. Current node_data_store keys: {list(node_data_store.keys())}")
    return node_id

# --- Funkcje dodawania konkretnych węzłów (używają create_node) ---

def add_input_node_callback(sender, app_data):
    print("DEBUG: add_input_node_callback called.")
    existing_input_node_id = get_input_node_id()
    if existing_input_node_id:
        print(f"INFO: Węzeł 'Input IMG' już istnieje: {existing_input_node_id}. Nie dodaję nowego.")
        return 
        
    node_id = create_node("Input IMG", inputs=0, outputs=1, pos=(20, 30), node_type="image")
    print(f"INFO: Węzeł 'Input IMG' ({node_id}) został dodany. PROSZĘ KLIKNĄĆ NA TEN WĘZEŁ W EDYTORZE, ABY GO ZAZNACZYĆ PRZED ZAŁADOWANIEM OBRAZU.")

def add_output_node_callback(sender, app_data):
    print("DEBUG: add_output_node_callback called.")
    node_id = create_node("Output IMG", inputs=1, outputs=0, pos=(300, 30), node_type="image")

def add_gaussian_node_callback(sender, app_data):
    print("DEBUG: add_gaussian_node_callback called.")
    node_id = create_node("Gaussian Blur", inputs=1, outputs=1, pos=(150, 60), node_type="filter")
    node_data_store[node_id]["filter_func"] = process_gaussian_blur
    node_data_store[node_id]["gaussian_std_dev"] = 1.0 

def add_median_node_callback(sender, app_data):
    print("DEBUG: add_median_node_callback called.")
    node_id = create_node("Median Filter", inputs=1, outputs=1, pos=(150, 120), node_type="filter")
    node_data_store[node_id]["filter_func"] = process_median_filter

def add_histogram_node_callback(sender, app_data):
    print("DEBUG: add_histogram_node_callback called.")
    node_id = create_node("Histogram Equalization", inputs=1, outputs=1, pos=(150, 180), node_type="filter")
    node_data_store[node_id]["filter_func"] = process_histogram_equalization

def add_grayscale_node_callback(sender, app_data):
    print("DEBUG: add_grayscale_node_callback called.")
    node_id = create_node("Grayscale", inputs=1, outputs=1, pos=(150, 240), node_type="filter")
    node_data_store[node_id]["filter_func"] = process_grayscale

def add_invert_node_callback(sender, app_data):
    print("DEBUG: add_invert_node_callback called.")
    node_id = create_node("Invert", inputs=1, outputs=1, pos=(150, 300), node_type="filter")
    node_data_store[node_id]["filter_func"] = process_invert

# --- Zmodyfikowany callback dla okna dialogowego ładowania pliku ---
def file_dialog_image_load_callback(sender, app_data, user_data):
    global current_input_node_for_load 
    print(f"DEBUG: file_dialog_image_load_callback - current_input_node_for_load (on entry): {current_input_node_for_load}")

    if not app_data or 'file_path_name' not in app_data or not app_data['file_path_name']:
        print("DEBUG: Nie wybrano pliku (anulowano lub brak ścieżki).")
        dpg.hide_item(sender) 
        current_input_node_for_load = None 
        return

    file_path = app_data['file_path_name']
    print(f"DEBUG: File path received from dialog: '{file_path}'")
    dpg.hide_item(sender) 

    if current_input_node_for_load is None or not dpg.does_item_exist(current_input_node_for_load):
        print(f"ERROR: Nie wybrano węzła 'Input IMG' do załadowania obrazu lub jest on niepoprawny. current_input_node_for_load jest {current_input_node_for_load}.")
        current_input_node_for_load = None 
        return

    target_node_id = current_input_node_for_load
    print(f"DEBUG: Attempting to load image to target_node_id: {target_node_id}")

    if not os.path.isfile(file_path):
        print(f"ERROR: Wybrana ścieżka '{file_path}' nie jest prawidłowym plikiem lub nie istnieje. Proszę wybrać konkretny plik obrazu.")
        node_data_store[target_node_id]["image_data"] = None
        node_inputs_status[target_node_id] = False
        display_image_in_node(target_node_id, None)
        current_input_node_for_load = None
        return

    try:
        print(f"DEBUG: Attempting Image.open('{file_path}')")
        img_pil = Image.open(file_path)
        print(f"DEBUG: Image opened successfully. Mode: {img_pil.mode}, Size: {img_pil.size}")
        
        # Convert to RGB if palette or grayscale to ensure 3 channels for consistency
        if img_pil.mode == 'P':
            print("DEBUG: Converting palette image to RGB.")
            img_pil = img_pil.convert('RGB')
        elif img_pil.mode == 'L':
            print("DEBUG: Converting grayscale image to RGB.")
            img_pil = img_pil.convert('RGB')
        
        # Convert to RGBA for consistent texture loading
        if img_pil.mode != 'RGBA':
            print(f"DEBUG: Converting image to RGBA from {img_pil.mode}.")
            img_pil = img_pil.convert('RGBA')

        np_image_data = np.array(img_pil)
        print(f"DEBUG: Converted to NumPy array. Shape: {np_image_data.shape}, Dtype: {np_image_data.dtype}")

        if not np.issubdtype(np_image_data.dtype, np.uint8):
            print("DEBUG: Clipping and converting image data to uint8.")
            np_image_data = (np_image_data.clip(0, 255)).astype(np.uint8)

        node_data_store[target_node_id]["image_data"] = np_image_data
        node_inputs_status[target_node_id] = True 
        
        print(f"DEBUG: Stored image data in node_data_store[{target_node_id}]. Image data present: {node_data_store[target_node_id]['image_data'] is not None}")
        display_image_in_node(target_node_id, np_image_data)
        
        print(f"DEBUG: Loaded image to {target_node_id}: {file_path}, Shape: {np_image_data.shape}, Dtype: {np_image_data.dtype}")
        
        start_processing_graph() 
    except Exception as e:
        print(f"ERROR: Error processing loaded image for NumPy: {e}")
        node_data_store[target_node_id]["image_data"] = None
        node_inputs_status[target_node_id] = False
        display_image_in_node(target_node_id, None)
    finally:
        current_input_node_for_load = None 
        print(f"DEBUG: current_input_node_for_load reset to: {current_input_node_for_load} (finally block)")

# Callback dla przycisku "Save Processed Image"
def _save_image_dialog_callback(sender, app_data, user_data):
    image_to_save = user_data 
    
    if not app_data or 'file_path_name' not in app_data or not app_data['file_path_name']:
        print("DEBUG: Nie wybrano pliku do zapisu (anulowano lub brak ścieżki).")
        dpg.hide_item("save_file_dialog_id")
        return

    file_path = app_data['file_path_name']
    dpg.hide_item("save_file_dialog_id")

    try:
        if image_to_save is None:
            print("INFO: Brak obrazu do zapisania.")
            return

        if not np.issubdtype(image_to_save.dtype, np.uint8):
            image_to_save = (image_to_save.clip(0, 255)).astype(np.uint8)
        
        if image_to_save.ndim == 2:
            img_pil = Image.fromarray(image_to_save, mode='L') 
        elif image_to_save.ndim == 3 and image_to_save.shape[2] == 3:
            img_pil = Image.fromarray(image_to_save, mode='RGB')
        elif image_to_save.ndim == 3 and image_to_save.shape[2] == 4: 
            img_pil = Image.fromarray(image_to_save, mode='RGBA')
        else:
            raise ValueError(f"Unsupported image format for saving. Shape: {image_to_save.shape}")

        img_pil.save(file_path)
        print(f"INFO: Processed image saved to: {file_path}")
    except Exception as e:
        print(f"ERROR: Error saving image: {e}")

initial_dir = os.path.expanduser("~") 
if not os.path.isdir(initial_dir):
    initial_dir = os.getcwd() 

print(f"DEBUG: Initial file dialog directory set to: {initial_dir}")

with dpg.file_dialog(directory_selector=False, show=False, callback=file_dialog_image_load_callback, tag="file_dialog_id", width=700, height=400):
    dpg.add_file_extension(".*", custom_text="[All Files]")
    dpg.add_file_extension(".png", color=(0, 255, 0, 255))
    dpg.add_file_extension(".jpg", color=(0, 255, 0, 255))
    dpg.add_file_extension(".jpeg", color=(0, 255, 0, 255))
    dpg.add_file_extension(".bmp", color=(0, 255, 0, 255))

with dpg.file_dialog(directory_selector=False, show=False, callback=_save_image_dialog_callback, tag="save_file_dialog_id", default_filename="processed_image.png", width=700, height=400):
    dpg.add_file_extension(".*", custom_text="[All Files]") 
    dpg.add_file_extension(".png", color=(0, 255, 0, 255))
    dpg.add_file_extension(".jpg", color=(0, 255, 0, 255))
    dpg.add_file_extension(".bmp", color=(0, 255, 0, 255))


def load_image_to_input_node_trigger(sender, app_data, user_data):
    global current_input_node_for_load 
    print("DEBUG: load_image_to_input_node_trigger called.")
    
    selected_nodes_dpg_ids = dpg.get_selected_nodes("node_editor") 
    print(f"DEBUG: Nodes currently selected in editor (DPG IDs): {selected_nodes_dpg_ids}")
    
    target_input_node_tag = None
    if selected_nodes_dpg_ids:
        selected_dpg_id = selected_nodes_dpg_ids[0] 
        selected_node_tag = dpg.get_item_alias(selected_dpg_id) 
        
        print(f"DEBUG: Selected DPG ID: {selected_dpg_id}, corresponding Tag: {selected_node_tag}")

        if selected_node_tag in node_data_store:
            node_label = node_data_store[selected_node_tag]["label"]
            print(f"DEBUG: Selected node label from node_data_store: {node_label}")
            if node_label == "Input IMG":
                target_input_node_tag = selected_node_tag 
                print(f"DEBUG: load_image_to_input_node_trigger - Identified selected Input IMG node (Tag): {target_input_node_tag}")
            else:
                print(f"INFO: Selected node ('{selected_node_tag}') is not an 'Input IMG' node. It's '{node_label}'.")
        else:
            print(f"ERROR: Selected node tag '{selected_node_tag}' (DPG ID: {selected_dpg_id}) not found in node_data_store. This indicates a data consistency issue.")
    else:
        print("INFO: No nodes selected in editor when 'Load Image' button was pressed.")

    if target_input_node_tag is None:
        print("ERROR: Proszę zaznaczyć węzeł 'Input IMG' w edytorze przed kliknięciem 'Load Image'.")
        return

    current_input_node_for_load = target_input_node_tag 
    print(f"DEBUG: current_input_node_for_load set for dialog callback: {current_input_node_for_load}")

    if not os.path.isdir(initial_dir):
        print(f"WARNING: Domyślny katalog '{initial_dir}' nie istnieje. Używam bieżącego katalogu roboczego.")
        dpg.configure_item("file_dialog_id", default_path=os.getcwd())
    else:
        dpg.configure_item("file_dialog_id", default_path=initial_dir)
        
    dpg.show_item("file_dialog_id")

# ZMODYFIKOWANA FUNKCJA
def save_processed_image_callback_trigger(sender, app_data):
    print("DEBUG: save_processed_image_callback_trigger called.")
    
    # 1. Zapewnij, że graf zostanie przetworzony przed próbą zapisu
    start_processing_graph() 

    output_node_id = get_output_node_id()
    
    # 2. Dodatkowe sprawdzenie, czy węzeł wyjściowy istnieje i czy ma dane obrazu
    if output_node_id == None: # Changed from 'is None' for consistency, though 'is None' is Pythonic
        print("INFO: Brak węzła 'Output IMG'. Proszę dodać węzeł wyjściowy.")
        return

    processed_image_data = node_data_store.get(output_node_id, {}).get("image_data")

    if processed_image_data is None:
        print(f"INFO: Węzeł 'Output IMG' ({output_node_id}) nie zawiera przetworzonego obrazu do zapisania. Upewnij się, że graf jest prawidłowo połączony i przetworzony.")
        return

    if not os.path.isdir(initial_dir):
        print(f"WARNING: Domyślny katalog '{initial_dir}' nie istnieje. Używam bieżącego katalogu roboczego.")
        dpg.configure_item("save_file_dialog_id", user_data=processed_image_data, default_path=os.getcwd())
    else:
        dpg.configure_item("save_file_dialog_id", user_data=processed_image_data, default_path=initial_dir)
        
    dpg.show_item("save_file_dialog_id")

# --- Logika przetwarzania grafu ---

def get_input_node_id():
    """Pobiera ID pierwszego węzła 'Input IMG' z node_data_store."""
    print("DEBUG: get_input_node_id() called. Checking node_data_store...")
    for node_id, data in node_data_store.items():
        if data["label"] == "Input IMG":
            print(f"DEBUG: get_input_node_id() found 'Input IMG' with ID: {node_id}")
            return node_id
    print("INFO: get_input_node_id() - No 'Input IMG' node found in node_data_store.")
    return None

def get_output_node_id():
    print("DEBUG: get_output_node_id() called. Checking node_data_store...")
    for node_id, data in node_data_store.items():
        if data["label"] == "Output IMG":
            print(f"DEBUG: get_output_node_id() found 'Output IMG' with ID: {node_id}")
            return node_id
    print("INFO: get_output_node_id() - No 'Output IMG' node found in node_data_store.")
    return None

def get_connected_node_and_attr(output_attr_id):
    print(f"DEBUG: get_connected_node_and_attr called for output_attr_id: {output_attr_id}")
    for link_id in link_ids:
        if dpg.does_item_exist(link_id):
            link_details = dpg.get_item_configuration(link_id)
            source_attr_dpg_id = link_details.get('source') 
            target_attr_dpg_id = link_details.get('target') 

            # WAŻNA POPRAWKA: Konwertuj UUID na aliasy przed porównaniem
            source_attr_tag = dpg.get_item_alias(source_attr_dpg_id) if dpg.does_item_exist(source_attr_dpg_id) else None
            target_attr_tag = dpg.get_item_alias(target_attr_dpg_id) if dpg.does_item_exist(target_attr_dpg_id) else None
            
            print(f"DEBUG: Checking link {link_id}: Source {source_attr_tag} -> Target {target_attr_tag}")

            if source_attr_tag == output_attr_id:
                if target_attr_tag: # Teraz target_attr_tag to już alias
                    # Nadal używamy UUID (target_attr_dpg_id) do pobrania rodzica, ale potem konwertujemy na alias
                    node_id_from_attr = dpg.get_item_alias(dpg.get_item_parent(target_attr_dpg_id)) 
                    print(f"DEBUG: Found link from {output_attr_id} to node {node_id_from_attr} via input {target_attr_tag}")
                    if node_id_from_attr in node_data_store:
                        return node_id_from_attr, target_attr_tag
    print(f"DEBUG: No connected node found for output_attr_id: {output_attr_id}")
    return None, None

def clear_downstream_data(start_node_id):
    print(f"DEBUG: clear_downstream_data called for node: {start_node_id}")
    q = [start_node_id]
    visited = set()

    while q:
        current_node_id = q.pop(0)
        if current_node_id in visited:
            continue
        visited.add(current_node_id)

        if current_node_id in node_data_store:
            node_data_store[current_node_id]["image_data"] = None
            display_image_in_node(current_node_id, None) 
            
            if node_data_store[current_node_id]["label"] != "Input IMG":
                node_inputs_status[current_node_id] = False 
            
            node_outputs[current_node_id] = None 
            print(f"DEBUG: Cleared data for node {current_node_id}. Image data: {node_data_store[current_node_id]['image_data'] is None}, Outputs: {node_outputs[current_node_id] is None}")


        for output_attr_id in node_data_store[current_node_id]["output_attr_id"]:
            connected_node_id, _ = get_connected_node_and_attr(output_attr_id)
            if connected_node_id and connected_node_id not in visited:
                q.append(connected_node_id)


def start_processing_graph(sender=None, app_data=None):
    print("\n--- start_processing_graph called ---")
    input_node_id = get_input_node_id()
    output_node_id = get_output_node_id()

    if input_node_id is None:
        print("INFO: Brak węzła 'Input IMG'. Dodaj go, aby rozpocząć przetwarzanie.")
        return
    if output_node_id is None:
        print("INFO: Brak węzła 'Output IMG'. Dodaj go, aby zobaczyć wynik.")
        return
    
    if node_data_store[input_node_id].get("image_data") is None:
        print(f"INFO: Węzeł 'Input IMG' ({input_node_id}) nie ma załadowanego obrazu. Nie rozpoczynam przetwarzania.")
        print(f"DEBUG: Value of node_data_store[{input_node_id}]['image_data'] is {node_data_store[input_node_id].get('image_data')}")
        return


    print("--- Rozpoczynanie przetwarzania grafu ---")

    for node_id in node_inputs_status:
        if node_id != input_node_id:
            node_inputs_status[node_id] = False
    
    node_outputs[input_node_id] = node_data_store[input_node_id]["image_data"]
    print(f"DEBUG: Input node {input_node_id} output data status: {node_outputs[input_node_id] is not None}")
    node_inputs_status[input_node_id] = True 
    print(f"DEBUG: Input node {input_node_id} initialized. Image data present: {node_outputs[input_node_id] is not None}")


    process_queue = [input_node_id] 
    visited_nodes = set() 
    visited_nodes.add(input_node_id)


    while process_queue:
        current_node_id = process_queue.pop(0)
        current_node_info = node_data_store[current_node_id]
        
        print(f"DEBUG: Processing node: {current_node_info['label']} ({current_node_id})")

        input_image_data = None
        if current_node_info["label"] == "Input IMG":
            input_image_data = node_data_store[current_node_id]["image_data"]
        else: 
            if len(current_node_info["input_attr_id"]) > 0:
                target_input_attr_id = current_node_info["input_attr_id"][0] 
                source_output_attr_id = None
                
                found_link = False
                for link_id in link_ids:
                    if dpg.does_item_exist(link_id): 
                        link_details = dpg.get_item_configuration(link_id)
                        # Ensure we get the alias for comparison
                        if dpg.get_item_alias(link_details.get('target')) == target_input_attr_id:
                            source_output_attr_id_dpg_uuid = link_details.get('source') # UUID
                            source_output_attr_id = dpg.get_item_alias(source_output_attr_id_dpg_uuid) # Alias
                            found_link = True
                            print(f"DEBUG: Link found for {current_node_id} input from {source_output_attr_id}")
                            break
                
                if found_link and source_output_attr_id:
                    # Need to get parent from the actual DPG UUID of the source output attribute, then its alias
                    source_node_dpg_uuid = dpg.get_item_parent(dpg.get_item_uuid(source_output_attr_id))
                    source_node_id = dpg.get_item_alias(source_node_dpg_uuid)

                    if source_node_id in node_outputs and node_outputs[source_node_id] is not None:
                        input_image_data = node_outputs[source_node_id]
                        print(f"DEBUG: Retrieved input data for {current_node_id} from {source_node_id}. Data present: {input_image_data is not None}")
                    else:
                        print(f"INFO: Missing input data from previous node ({source_node_id}) for node {current_node_info['label']} ({current_node_id}). Clearing downstream data.")
                        clear_downstream_data(current_node_id) 
                        continue 
                else:
                    print(f"INFO: Node {current_node_info['label']} ({current_node_id}) is not connected. Clearing data.")
                    clear_downstream_data(current_node_id) 
                    continue

        processed_data = None
        if current_node_info["type"] == "filter" and current_node_info["filter_func"] is not None:
            if input_image_data is not None:
                try:
                    if current_node_info["label"] == "Gaussian Blur":
                        current_std_dev = node_data_store[current_node_id].get("gaussian_std_dev", 1.0)
                        
                        selected_nodes_dpg_ids = dpg.get_selected_nodes("node_editor") 
                        if selected_nodes_dpg_ids: 
                            selected_dpg_id = selected_nodes_dpg_ids[0]
                            selected_node_tag = dpg.get_item_alias(selected_dpg_id)
                            if selected_node_tag == current_node_id and dpg.does_item_exist("gaussian_std_dev_slider"):
                                current_std_dev = dpg.get_value("gaussian_std_dev_slider")
                                node_data_store[current_node_id]["gaussian_std_dev"] = current_std_dev 
                                print(f"DEBUG: Gaussian Blur std_dev updated to {current_std_dev} for node {current_node_id}")
                        
                        processed_data = current_node_info["filter_func"](input_image_data, current_std_dev)
                    else:
                        processed_data = current_node_info["filter_func"](input_image_data)
                    print(f"DEBUG: Filter '{current_node_info['label']}' applied successfully to node {current_node_id}. Output data present: {processed_data is not None}")
                except Exception as e:
                    print(f"ERROR: Error applying filter {current_node_info['label']} to node {current_node_id}: {e}")
                    processed_data = None 
            else:
                print(f"INFO: No input data for filter {current_node_info['label']} ({current_node_id}).")
                processed_data = None

            node_data_store[current_node_id]["image_data"] = processed_data
            node_outputs[current_node_id] = processed_data 
            display_image_in_node(current_node_id, processed_data)

        elif current_node_info["label"] == "Output IMG":
            node_data_store[current_node_id]["image_data"] = input_image_data
            node_outputs[current_node_id] = input_image_data
            display_image_in_node(current_node_id, input_image_data) 
            print(f"DEBUG: Output IMG node {current_node_id} updated. Image data present: {input_image_data is not None}")
            
        for output_attr_id in current_node_info["output_attr_id"]:
            print(f"DEBUG: Checking output attribute: {output_attr_id} from node {current_node_id}")
            connected_node_id, connected_input_attr_id = get_connected_node_and_attr(output_attr_id)
            if connected_node_id and connected_node_id not in visited_nodes:
                if node_outputs[current_node_id] is not None:
                    node_inputs_status[connected_node_id] = True 
                    process_queue.append(connected_node_id)
                    visited_nodes.add(connected_node_id) 
                    print(f"DEBUG: Added connected node {connected_node_id} to process queue from {current_node_id}. New queue: {process_queue}")
                else:
                    print(f"INFO: Output data from node {current_node_id} is empty. Not adding connected node {connected_node_id} to queue.")

    print("--- Zakończono przetwarzanie grafu ---\n")


## Konfiguracja i uruchomienie DPG


dpg.create_viewport(title='Node Editor Obrazów', width=1250, height=700)
dpg.setup_dearpygui()

with dpg.window(label="Main window", width=900, height=650):
    with dpg.menu_bar():
        with dpg.menu(label="Add Node"):
            dpg.add_menu_item(label="Input IMG", callback=add_input_node_callback)
            dpg.add_menu_item(label="Output IMG", callback=add_output_node_callback)
            dpg.add_menu_item(label="Gaussian Blur", callback=add_gaussian_node_callback)
            dpg.add_menu_item(label="Median Filter", callback=add_median_node_callback)
            dpg.add_menu_item(label="Histogram Equalization", callback=add_histogram_node_callback)
            dpg.add_menu_item(label="Grayscale", callback=add_grayscale_node_callback)
            dpg.add_menu_item(label="Invert", callback=add_invert_node_callback)

    with dpg.node_editor(callback=node_editor_callback, delink_callback=delink_callback, tag="node_editor" 
                         # Consider adding this for visual debugging:
                         # minimap=True, minimap_location=dpg.mvNodeEditorMiniMap_Location_BottomRight
                            ):
        pass

with dpg.window(label="Control Panel", pos=[910, 20], width=300):
    dpg.add_text("Operations", color=(255, 255, 0)) 
    dpg.add_button(label="Load Image", tag="load_image_button", callback=load_image_to_input_node_trigger)
    dpg.add_button(label="Process Graph", callback=start_processing_graph) 
    dpg.add_button(label="Save Processed Image", callback=save_processed_image_callback_trigger)

    dpg.add_separator()
    dpg.add_text("Filter Parameters", color=(255, 255, 0)) 
    
    with dpg.group(tag="gaussian_std_dev_slider_group", show=False):
        dpg.add_slider_float(label="Gaussian Std Dev", default_value=1.0, min_value=0.1, max_value=5.0, tag="gaussian_std_dev_slider", format="%.1f",
                               callback=start_processing_graph) 
    
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()