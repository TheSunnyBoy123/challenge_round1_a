import os
import json
import re
from collections import Counter, defaultdict
from PIL import Image
from ultralytics import YOLO
import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# ==============================================================================
#  PART 0 - PDF TO IMAGE CONVERSION (UNCHANGED)
# ==============================================================================
def convert_pdfs_to_images(pdf_dir: str, output_dir: str):
    if not os.path.exists(pdf_dir):
        print(f"  [Error] PDF input directory '{pdf_dir}' not found.")
        return
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, filename)
            pdf_name_without_ext = os.path.splitext(filename)[0]
            image_subdir = os.path.join(output_dir, pdf_name_without_ext)
            os.makedirs(image_subdir, exist_ok=True)
            print(f"  - Converting '{filename}'...")
            try:
                images = convert_from_path(pdf_path, dpi=300)
                for i, image in enumerate(images):
                    image_path = os.path.join(image_subdir, f'page_{i+1:03d}.jpg')
                    image.save(image_path, 'JPEG')
                print(f"    - Saved {len(images)} pages to '{image_subdir}'")
            except Exception as e:
                print(f"    [Error] Could not convert {filename}: {e}")

# ==============================================================================
#  PART 1: DOCUMENT PROCESSING CLASS (CORE LOGIC UNCHANGED)
# ==============================================================================
class DocumentProcessor:
    def __init__(self, input_dir: str, model_path: str, output_dir: str):
        self.input_dir = input_dir
        self.model_path = model_path
        self.output_dir = output_dir
        self.annotated_image_dir_name = 'annotated_images'
        self.output_json_path = os.path.join(output_dir, 'document_layout.json')
        self.images = []
        self.img_list = []
        self.page_map = {}
        self.docseg_model = None
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_and_sort_images(self):
        if not os.path.isdir(self.input_dir): return False
        image_files = sorted([f for f in os.listdir(self.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not image_files: return False
        for i, filename in enumerate(image_files):
            image_path = os.path.join(self.input_dir, filename)
            self.img_list.append(image_path)
            self.images.append(Image.open(image_path))
            self.page_map[i + 1] = i
        return True

    def _load_model(self):
        try:
            self.docseg_model = YOLO(self.model_path)
            return True
        except Exception: return False

    def _run_yolo_model(self):
        if not self.docseg_model or not self.img_list: return None
        return self.docseg_model(source=self.img_list, save=True, project=self.output_dir, name=self.annotated_image_dir_name, exist_ok=True, verbose=False)

    def _extract_components_with_ocr(self, results):
        document_components = []
        class_names = self.docseg_model.names
        for i, entry in enumerate(results):
            page_image = self.images[i]
            page_width, page_height = page_image.size
            for box in entry.boxes:
                component_type = class_names[int(box.cls[0])]
                if component_type not in ['Title', 'Section-header']: continue
                box_coords = box.xyxy[0].numpy()
                padding = 5
                x1, y1, x2, y2 = max(0, box_coords[0]-padding), max(0, box_coords[1]-padding), min(page_width, box_coords[2]+padding), min(page_height, box_coords[3]+padding)
                cropped_image_pil = page_image.crop((x1, y1, x2, y2))
                cropped_image_cv = cv2.cvtColor(np.array(cropped_image_pil), cv2.COLOR_RGB2GRAY)
                _, processed_image = cv2.threshold(cropped_image_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                config = '--psm 7'
                extracted_text = pytesseract.image_to_string(processed_image, config=config).strip()
                document_components.append({"page": i + 1, "type": component_type, "bbox": [round(c, 2) for c in box_coords.tolist()], "confidence": round(float(box.conf[0]), 4), "text": re.sub(r'\s+', ' ', extracted_text)})
        return document_components

    def _identify_main_title(self, components: list) -> list:
        page_one_titles = [c for c in components if c['page'] == 1 and c['type'] == 'Title']
        if len(page_one_titles) > 1:
            page_one_titles.sort(key=lambda c: c['bbox'][1])
            for comp in page_one_titles[1:]: comp['type'] = 'Section-header'
        elif not page_one_titles:
            page_one_headers = [c for c in components if c['page'] == 1 and c['type'] == 'Section-header']
            if page_one_headers:
                page_one_headers.sort(key=lambda c: c['bbox'][1])
                page_one_headers[0]['type'] = 'Title'
        return components
        
    def _extract_hierarchical_features(self, components: list) -> list:
        for i, c in enumerate(components):
            if 'bbox' not in c: continue
            page_w, page_h = self.images[self.page_map.get(c['page'])].size
            font_size = c['bbox'][3] - c['bbox'][1]
            space_before = c['bbox'][1] - components[i-1]['bbox'][3] if i>0 and components[i-1]['page']==c['page'] else c['bbox'][1]
            c.update({'font_size':round(font_size,2), 'norm_x0':round(c['bbox'][0]/page_w,4), 'norm_y0':round(c['bbox'][1]/page_h,4), 'is_centered':1 if abs(((c['bbox'][0]+c['bbox'][2])/2)-(page_w/2))<(page_w*0.05) else 0})
        return components

    def _assign_hierarchy_hybrid(self, components: list) -> list:
        print("  - Final Engine: Using Hybrid Rule-Based Classification...")
        headers = [c for c in components if c.get('type') == 'Section-header']
        if not headers: return components
        rule_assigned_indices, unassigned_headers = set(), []
        for i, header in enumerate(headers):
            match = re.match(r'^(\d+(\.\d+)*)\.?', header.get('text', '').strip())
            if match:
                level = match.group(1).count('.') + 1
                if level <= 4:
                    header['hierarchy_level'] = f'h{level}'
                    rule_assigned_indices.add(i)
        
        unassigned_headers = [h for i, h in enumerate(headers) if i not in rule_assigned_indices]
        if not unassigned_headers:
            print("    - All headers assigned by numbering rules.")
            return components

        features_to_use = ['font_size', 'norm_x0', 'is_centered']
        assigned_by_rule = [h for i, h in enumerate(headers) if i in rule_assigned_indices]
        if assigned_by_rule:
            print("    - Creating style profiles from numbered headings...")
            all_header_features = np.array([[h[key] for key in features_to_use] for h in headers])
            scaler = StandardScaler()
            scaled_all_features = scaler.fit_transform(all_header_features)
            header_map = {id(h): i for i, h in enumerate(headers)}
            grouped_features = defaultdict(list)
            for header in assigned_by_rule:
                grouped_features[header['hierarchy_level']].append(scaled_all_features[header_map[id(header)]])
            
            style_centroids = {level: np.mean(feature_list, axis=0) for level, feature_list in grouped_features.items()}
            print(f"    - Classifying {len(unassigned_headers)} un-numbered headers...")
            for header in unassigned_headers:
                if not style_centroids: break
                scaled_feature_vector = scaled_all_features[header_map[id(header)]]
                distances = {level: np.linalg.norm(scaled_feature_vector - centroid) for level, centroid in style_centroids.items()}
                header['hierarchy_level'] = min(distances, key=distances.get)
        else:
            print("    - No numbered headings found. Falling back to style clustering.")
            if len(unassigned_headers) < 2: return components
            feature_matrix = np.array([[h[key] for key in features_to_use] for h in unassigned_headers])
            scaled_features = StandardScaler().fit_transform(feature_matrix)
            n_clusters = min(4, len(unassigned_headers))
            labels = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(scaled_features)
            font_size_index = features_to_use.index('font_size')
            cluster_avg_font_size = [feature_matrix[labels == i, font_size_index].mean() for i in range(n_clusters)]
            sorted_cluster_indices = sorted(range(n_clusters), key=lambda i: cluster_avg_font_size[i], reverse=True)
            hierarchy_map = {cluster_id: f'h{rank + 1}' for rank, cluster_id in enumerate(sorted_cluster_indices)}
            for i, component in enumerate(unassigned_headers):
                component['hierarchy_level'] = hierarchy_map[labels[i]]
        return components

    def process_document(self):
        print("-" * 50)
        print(f"Processing Document: {os.path.basename(self.input_dir)}")
        if not self._load_and_sort_images() or not self._load_model(): return False
        results = self._run_yolo_model()
        if not results: return False
        document_components = self._extract_components_with_ocr(results)
        document_components.sort(key=lambda c: (c['page'], c['bbox'][1], c['bbox'][0]))
        document_components = self._identify_main_title(document_components)
        document_components = self._extract_hierarchical_features(document_components)
        document_components = self._assign_hierarchy_hybrid(document_components)
        for i, component in enumerate(document_components): component['line_id'] = i + 1
        print(f"  - Saving layout analysis to: '{self.output_json_path}'")
        with open(self.output_json_path, 'w', encoding='utf-8') as f:
            json.dump(document_components, f, indent=4, ensure_ascii=False)
        return True

# ==============================================================================
#  PART 2: ORCHESTRATORS (REFINED)
# ==============================================================================
def run_full_pipeline(base_image_dir, intermediate_layout_dir, model_path):
    """
    Processes all document image folders, saving intermediate layout files.
    """
    if not os.path.exists(model_path):
        print(f"FATAL ERROR: Model file not found at '{model_path}'")
        return
    if not os.path.exists(base_image_dir):
        print(f"FATAL ERROR: Base image directory '{base_image_dir}' not found.")
        return
    doc_names = [d for d in os.listdir(base_image_dir) if os.path.isdir(os.path.join(base_image_dir, d))]
    for doc_name in doc_names:
        input_image_dir = os.path.join(base_image_dir, doc_name)
        # Each document gets its own subdirectory for intermediate files
        doc_layout_dir = os.path.join(intermediate_layout_dir, doc_name)
        processor = DocumentProcessor(input_dir=input_image_dir, model_path=model_path, output_dir=doc_layout_dir)
        if not processor.process_document():
            print(f"  [Error] Document processing failed for {doc_name}.")

def format_to_final_outline(layout_data: list) -> dict:
    """
    Formats the detailed layout data into the final, clean outline structure.
    """
    title_text = ""
    outline = []
    layout_data.sort(key=lambda x: x.get('line_id', float('inf')))
    for component in layout_data:
        if component.get('type') == 'Title' and not title_text:
            title_text = component.get('text', '').strip()
        if 'hierarchy_level' in component:
            outline.append({
                "level": component['hierarchy_level'].upper(), 
                "text": component.get('text', '').strip(), 
                "page": component.get('page')
            })
    return {"title": title_text, "outline": outline}

def run_final_conversion(intermediate_dir: str, final_dir: str):
    """
    UPDATED: Reads intermediate layouts and saves the final JSON outline to the
    top-level final output directory with the correct name.
    """
    if not os.path.exists(intermediate_dir):
        print(f"  [Error] Intermediate directory '{intermediate_dir}' not found.")
        return
    doc_names = [d for d in os.listdir(intermediate_dir) if os.path.isdir(os.path.join(intermediate_dir, d))]
    for doc_name in doc_names:
        layout_json_path = os.path.join(intermediate_dir, doc_name, 'document_layout.json')
        # The final output is named after the document and placed in the root of the final directory
        final_outline_path = os.path.join(final_dir, f"{doc_name}.json")
        if os.path.exists(layout_json_path):
            print(f"  - Creating final output for '{doc_name}'...")
            try:
                with open(layout_json_path, 'r', encoding='utf-8') as f: layout_data = json.load(f)
                final_data = format_to_final_outline(layout_data)
                with open(final_outline_path, 'w', encoding='utf-8') as f: json.dump(final_data, f, indent=4, ensure_ascii=False)
                print(f"    - Successfully saved final outline to '{final_outline_path}'")
            except Exception as e:
                print(f"    [Error] Could not create final output for {doc_name}: {e}")

# ==============================================================================
#  MAIN EXECUTION BLOCK (REFINED FOR CLARITY AND ROBUSTNESS)
# ==============================================================================
if __name__ == "__main__":
    # --- Define fixed paths for the environment ---
    PDF_INPUT_DIR = '/app/input'
    FINAL_OUTPUT_DIR = '/app/output'
    MODEL_PATH = '/app/yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt'

    # --- Define temporary directories for all intermediate files ---
    TEMP_BASE_DIR = '/tmp/doc_processing'
    TEMP_IMAGE_DIR = os.path.join(TEMP_BASE_DIR, 'images')
    TEMP_LAYOUT_DIR = os.path.join(TEMP_BASE_DIR, 'layouts')

    # --- Setup directories ---
    os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)
    os.makedirs(TEMP_LAYOUT_DIR, exist_ok=True)

    print("--- Document Outline Extraction Pipeline Started ---")
    
    if not os.path.exists(MODEL_PATH):
        print(f"FATAL ERROR: Model file not found at '{MODEL_PATH}'")
        exit()

    # --- STEP 1: Convert all PDFs to images into a temporary location ---
    print("\n--- STEP 1: Converting PDFs to Images ---")
    convert_pdfs_to_images(pdf_dir=PDF_INPUT_DIR, output_dir=TEMP_IMAGE_DIR)

    # --- STEP 2: Run layout analysis and save intermediate files to a temporary location ---
    print("\n--- STEP 2: Running Document Layout Analysis ---")
    run_full_pipeline(
        base_image_dir=TEMP_IMAGE_DIR, 
        intermediate_layout_dir=TEMP_LAYOUT_DIR,
        model_path=MODEL_PATH
    )

    # --- STEP 3: Generate the final, clean JSON outputs in the required directory ---
    print("\n--- STEP 3: Generating Final Output JSONs ---")
    run_final_conversion(
        intermediate_dir=TEMP_LAYOUT_DIR,
        final_dir=FINAL_OUTPUT_DIR
    )

    print("\n--- All PDFs processed successfully. ---")