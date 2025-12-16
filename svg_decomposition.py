import base64
import copy
import io
import json
import os
import re
import xml.etree.ElementTree as ET
import concurrent.futures

from bs4 import BeautifulSoup

from prompts.vlm_strategies import *
from prompts.llm_templates import PARSER_SYSTEM_PROMPT, PARSER_PROMPT, PARSER_GROUP_PROMPT

if os.name == 'nt':
    # Cairosvg requires an additional dlls directory on Windows.
    # https://stackoverflow.com/questions/46265677/get-cairosvg-working-in-windows
    # https://sk1project.net/uc2/download/ <-- Download and install UniConvertor
    # Add the path to the dlls directory to the PATH environment variable
    os.environ['path'] += r";C:\\Program Files\\UniConvertor-2.0rc5\dlls"
import cairosvg
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image

#### Parser Utilities ####

def pretty_print_html(markup_string, parser='xml'):
    """Pretty-print HTML or SVG string using BeautifulSoup.
    
    Args:
        markup_string: The HTML or SVG string to format
        is_svg: Set to True for SVG, False for HTML
        
    Returns:
        A formatted string with proper indentation
    """
    # For SVG, use the 'xml' parser instead of 'html.parser'
    soup = BeautifulSoup(markup_string, parser)
    pretty_string = soup.prettify()

    return pretty_string

def svg_to_png(svg_file, resize_to=1024):
    """Renders an SVG file to a PNG file and resizes it."""
    png_bytes = cairosvg.svg2png(url=svg_file)
    
    # Open the PNG image from the bytes and resize the image to the desired dimensions.
    image = Image.open(io.BytesIO(png_bytes))
    image = image.resize((resize_to, resize_to), Image.LANCZOS)

    # Encode the image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return img_str

def svg_to_base64(svg_element):
    """Convert SVG element to base64 PNG."""
    svg_string = ET.tostring(svg_element, encoding='utf-8').decode('utf-8')
    svg_base64 = base64.b64encode(svg_string.encode('utf-8')).decode('utf-8')
    data_uri = f"data:image/svg+xml;base64,{svg_base64}"
    
    # You'll need to implement svg_to_png function or use cairosvg
    return svg_to_png(data_uri)

class SVGParser:
    def __init__(self, vlm, logger=None):
        self.logger = logger or print
        self.vlm = vlm

        self.system_template = PARSER_SYSTEM_PROMPT
        self.prompt_template = PARSER_PROMPT
        self.group_template = PARSER_GROUP_PROMPT

    def set_parser(self, svg_file, plans):
        self.set_element_tree(svg_file)
        self.set_svg_file(svg_file)
        self.set_plans(plans)

    def set_plans(self, plans):
        self.plans = plans
        self.semantic_groups = {plan: [] for plan in self.plans.keys()}
        self.semantic_groups['others'] = []

    def set_svg_file(self, svg_file):
        self.svg_file = svg_file
        tree = ET.parse(svg_file)
        root = tree.getroot()
        
        svg_string = ET.tostring(root, encoding='utf-8').decode('utf-8')
        svg_base64 = base64.b64encode(svg_string.encode('utf-8')).decode('utf-8')
        data_uri = f"data:image/svg+xml;base64,{svg_base64}"
        self.full_svg_base64 = svg_to_png(data_uri)

    def set_element_tree(self, svg_file):
        self.svg_file = svg_file
        self.tree = ET.parse(svg_file)
        self.root = self.tree.getroot()
        ns_match = re.match(r'{(.*)}', self.root.tag)
        self.ns = ns_match.group(1) if ns_match else 'http://www.w3.org/2000/svg'
        ET.register_namespace('', self.ns)
        
        self.drawing_elements = []
        for elem in ['path', 'rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon', 'text', 'image']:
            self.drawing_elements.append(f"{{{self.ns}}}{elem}")
            self.drawing_elements.append(elem)

    def analyze_element_with_vlm(self, element):
        """Analyze an element using the VLM."""
        # Render the element highlighted
        svg_copy = copy.deepcopy(self.root)
        self.highlight_target(svg_copy, element)
        highlighted_svg_base64 = svg_to_base64(svg_copy)
        
        # Create prompt
        formatted_prompt = self.prompt_template.format(
            semantic_categories="\n".join(self.plans.keys()),
            group=self.group_template,
        )
        
        # Create messages for VLM
        messages = [
            SystemMessage(content=self.system_template),
            HumanMessage(content=[
                {'type': 'text', 'text': formatted_prompt},
                {'type': 'image_url', 'image_url': {"url": f"data:image/jpeg;base64,{self.full_svg_base64}"}},
                {'type': 'image_url', 'image_url': {"url": f"data:image/jpeg;base64,{highlighted_svg_base64}"}},
            ])
        ]
        
        # Get response
        response = self.vlm.invoke(messages)
        response_text = response.content.strip().replace('```', '').replace('json', '').strip()

        try:
            result = json.loads(response_text)
            return result

        except json.JSONDecodeError:
            return self.analyze_element_with_vlm(element)

    def is_same_element(self, elem1, elem2):
        """Check if two elements are the same by comparing key attributes."""
        # Compare tag
        if elem1.tag != elem2.tag:
            return False
        
        # Compare key identifying attributes
        key_attrs = ['id', 'd', 'x', 'y', 'cx', 'cy', 'r', 'width', 'height', 'points']
        
        for attr in key_attrs:
            val1 = elem1.get(attr)
            val2 = elem2.get(attr)
            if val1 != val2:
                return False
        
        # Compare text content
        if elem1.text != elem2.text:
            return False
        
        return True

    def is_drawable_element(self, element):
        """Check if element is a drawable SVG element."""
        drawable_tags = ['path', 'rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon', 'text', 'image']
        tag_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag
        return tag_name in drawable_tags

    def traverse_and_tag(self, element):
        """
        Recursively traverse and tag elements.
        This is the core function for semantic tagging and also very easy to break.
        Do not modify lightly.
        """
        if element.tag.endswith('defs') or element.tag.endswith('style'):
            return # Skip defs and style sections
        
        # If it's a group or drawable element, analyze it
        tag_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag
        
        if tag_name == 'g' or self.is_drawable_element(element):
            result = self.analyze_element_with_vlm(element)
            category = result['category']
            
            # Add category as class
            existing_class = element.get('class', '')
            if existing_class:
                element.set('class', f"{existing_class} {category}")
            else:
                element.set('class', category)
            
            # If it's a group and categorized as "group", traverse children
            if tag_name == 'g' and category == 'group':
                for child in element:
                    self.traverse_and_tag(child)
        else:
            # For non-group, non-drawable elements, just traverse children
            # Not sure what this would be, but let's keep it safe
            for child in element:
                self.traverse_and_tag(child)

    def tag_semantics(self, is_burn_in=False):
        """Main function to tag SVG elements with semantic categories.""" 
        drawable_children = []
        for child in self.root:
            tag_name = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if tag_name == 'g' or self.is_drawable_element(child):
                drawable_children.append(child)
        
        # If there's only one group, skip it and analyze its children directly
        if len(drawable_children) == 1 and drawable_children[0].tag.endswith('g'):
            top_group = drawable_children.pop()
            top_group.set('class', 'group')  # Tag it as group
            for child in top_group:
                self.traverse_and_tag(child)
        else: # Normal traversal
            for child in self.root:
                self.traverse_and_tag(child)
        
        if not is_burn_in:
            self.save_all_svg_groups()
            self.save_tagged_svg()
        svg_string = ET.tostring(self.root, encoding='utf-8').decode('utf-8')
        return svg_string

    def save_all_svg_groups(self):
        """Save each semantic group to separate PNG files."""
        log_dir = self.logger.exp_dir
        for i, class_name in enumerate(self.plans.keys()):
            img_str = self.render_by_class(class_name)
            img_data = base64.b64decode(img_str)
            with open(os.path.join(log_dir, f"{i:02d}_{class_name}.png"), 'wb') as f:
                f.write(img_data)

    def load_tagged_svg(self, tagged_svg_path=None):
        """Load previously tagged SVG if exists."""
        if tagged_svg_path is None:
            tagged_svg_path = os.path.join('./logs', os.path.basename(self.logger.exp_name)+'.svg')
        if os.path.exists(tagged_svg_path):
            with open(tagged_svg_path, 'r', encoding='utf-8') as f:
                svg_string = f.read()
            self.set_svg_file(tagged_svg_path)
            self.set_element_tree(tagged_svg_path)
            return svg_string
        return None
    
    def save_tagged_svg(self):
        """Save the tagged SVG to a specified path."""
        svg_string = ET.tostring(self.root, encoding='utf-8').decode('utf-8')
        tagged_svg_path = os.path.join('./logs', os.path.basename(self.logger.exp_name)+'.svg')
        with open(tagged_svg_path, 'w', encoding='utf-8') as f:
            f.write(svg_string)

    def render_by_class(self, class_name):
        """Render SVG showing only elements with specified class."""
        def _filter_by_class(element, target_class):
            """Recursively filter elements, keeping only those with target class."""
            # Get the clean tag name (without namespace)
            tag_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag
            if tag_name in ['defs', 'style']:
                return True

            # Check if current element has the target class
            element_classes = element.get('class', '').split()
            has_target_class = target_class in element_classes
            has_visible_children = False

            # If it's a group with the target class, keep it and all its children
            if tag_name == 'g' and has_target_class:
                return True  # Keep the group and all its children, regardless of their class
            
            children_to_remove = []
            for child in element:
                if _filter_by_class(child, target_class):
                    has_visible_children = True
                else:
                    children_to_remove.append(child)
            
            # Remove children that don't have the class
            for child in children_to_remove:
                element.remove(child)
            
            # If it's a group and has relevant children, preserve it
            if tag_name == 'g' and has_visible_children:
                return True  # Keep the group if it has relevant children
            
            # Keep the element if it has the target class or if it has visible children
            return has_target_class or has_visible_children
        
        # Create a copy of the SVG
        svg_copy = copy.deepcopy(self.root)
        
        # Remove elements that don't have the specified class
        # The root <svg> element itself is processed here, and its children are filtered.
        _filter_by_class(svg_copy, class_name)
    
        return svg_to_base64(svg_copy)

class SingleSVGParser(SVGParser):
    def __init__(self, vlm, logger):
        """
        Initialize the SVG Semantic Parser with a VLM and logger.
        This version is used to test single strategies and for evaluation.
        """
        super().__init__(vlm, logger)

        # self.vlm_strategy = Highlight()
        # self.vlm_strategy = BoundingBox()
        self.vlm_strategy = ZoomInHighlight()
        # self.vlm_strategy = VanillaRender()

    def set_element_tree(self, svg_file):
        self.svg_file = svg_file
        self.tree = ET.parse(svg_file)
        self.root = self.tree.getroot()
        ns_match = re.match(r'{(.*)}', self.root.tag)
        self.ns = ns_match.group(1) if ns_match else 'http://www.w3.org/2000/svg'
        self.vlm_strategy.set_namespace(self.ns)
        ET.register_namespace('', self.ns)
        
        # Add both namespaced and non-namespaced versions
        self.drawing_elements = []
        for elem in ['path', 'rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon', 'text', 'image']:
            self.drawing_elements.append(f"{{{self.ns}}}{elem}")
            self.drawing_elements.append(elem)

    def analyze_element_with_vlm(self, element):
        """Analyze an element using the VLM."""
        # Render the element highlighted
        svg_copy = copy.deepcopy(self.root)
        svg_copy = self.vlm_strategy.analyze(svg_copy, element)
        
        highlighted_svg_base64 = svg_to_base64(svg_copy)
        
        # Create prompt
        prompt_template = self.vlm_strategy.get_prompt_template()
        group_template = self.vlm_strategy.get_group_template()
        formatted_prompt = prompt_template.format(
            semantic_categories="\n".join(self.plans.keys()),
            group=group_template,
        )
        
        # Create messages for VLM
        messages = [
            SystemMessage(content=self.system_template),
            HumanMessage(content=[
                {'type': 'text', 'text': formatted_prompt},
                {'type': 'image_url', 'image_url': {"url": f"data:image/jpeg;base64,{self.full_svg_base64}"}},
                {'type': 'image_url', 'image_url': {"url": f"data:image/jpeg;base64,{highlighted_svg_base64}"}},
            ])
        ]
        response = self.vlm.invoke(messages)
        response_text = response.content.strip().replace('```', '').replace('json', '').strip()

        try:
            result = json.loads(response_text)
            return result

        except json.JSONDecodeError:
            return self.analyze_element_with_vlm(element)

class MultiSVGParser(SVGParser):
    def __init__(self, vlm, logger):
        """
        Initialize the Spectral Consensus Parser with a VLM and logger.
        We implement multiple strategies for analyzing SVG elements.
        Each element will be analyzed using multiple strategies,
        and the results will be combined to form a consensus.
        """
        super().__init__(vlm, logger)

        self.vlm_strategies = [
            VanillaRender(),
            Highlight(),
            BoundingBox(),
            ZoomInHighlight(),
            Outline(),
            # Add more strategies if needed
        ]
        self.agreement2decision = AgreementMatrix(
            len(self.vlm_strategies), 
            logger
        )

    def set_parser(self, svg_file, plans):
        self.set_element_tree(svg_file)
        self.set_svg_file(svg_file)
        self.set_plans(plans)
        self.agreement2decision.set_classes(len(plans.keys()))

    def set_element_tree(self, svg_file):
        self.svg_file = svg_file
        self.tree = ET.parse(svg_file)
        self.root = self.tree.getroot()
        ns_match = re.match(r'{(.*)}', self.root.tag)
        self.ns = ns_match.group(1) if ns_match else 'http://www.w3.org/2000/svg'
        for strategy in self.vlm_strategies:
            strategy.set_namespace(self.ns)
        ET.register_namespace('', self.ns)
        
        # Add both namespaced and non-namespaced versions
        self.drawing_elements = []
        for elem in ['path', 'rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon', 'text', 'image']:
            self.drawing_elements.append(f"{{{self.ns}}}{elem}")
            self.drawing_elements.append(elem)

    def _process_single_strategy(self, vlm_strategy, element):
        """Process a single VLM strategy - extracted from the original loop body."""
        svg_copy = copy.deepcopy(self.root)
        svg_copy = vlm_strategy.analyze(svg_copy, element)
        highlighted_svg_base64 = svg_to_base64(svg_copy)
        
        # Create prompt
        prompt_template = vlm_strategy.get_prompt_template()
        group_template = vlm_strategy.get_group_template()
        formatted_prompt = prompt_template.format(
            semantic_categories="\n".join(self.plans.keys()),
            group=group_template if element.tag.endswith('g') else "",
        )
        messages = [
            SystemMessage(content=self.system_template),
            HumanMessage(content=[
                {'type': 'text', 'text': formatted_prompt},
                {'type': 'image_url', 'image_url': {"url": f"data:image/jpeg;base64,{self.full_svg_base64}"}},
                {'type': 'image_url', 'image_url': {"url": f"data:image/jpeg;base64,{highlighted_svg_base64}"}},
            ])
        ]
        
        # Get response
        response = self.vlm.invoke(messages)
        response_text = response.content.strip().replace('```', '').replace('json', '').strip()
        self.logger.debug(f"[{vlm_strategy}] Response: {response_text}")
        
        try:
            result = json.loads(response_text)
            return result['category']
        except json.JSONDecodeError:
            self.logger.warning(f"JSON parse error for element {element.get('id', 'unknown')}, retrying...")
            return self._process_single_strategy(vlm_strategy, element)
    
    def analyze_element_with_vlm(self, element):
        """Analyze an element using the VLM."""
        # Parallelize the VLM strategy loop
        vlm_predictions = [None] * len(self.vlm_strategies)  # To preserve order
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_index = {
                executor.submit(self._process_single_strategy, strategy, element): idx 
                for idx, strategy in enumerate(self.vlm_strategies)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                result = future.result()
                if result is not None:
                    vlm_predictions[idx] = result

        self.logger.info(f"[MultiSVGParser] VLM Predictions: {vlm_predictions}")
        map_prediction = self.agreement2decision.update_and_decide(vlm_predictions)
        self.logger.info(f"[MultiSVGParser] Multi VLM Prediction: {map_prediction}")

        return {
            'category': map_prediction,
        }
    
    def burn_in(self, burn_in_count):
        """Burn-in method to stabilize the parser.""" 
        iteration = 0
        original_root = copy.deepcopy(self.root)
        origianl_tree = copy.deepcopy(self.tree)
        while iteration < burn_in_count:
            self.logger.info(f"[SVG Parser] Burn-in Iteration {iteration + 1:02d}/{burn_in_count:02d}")
            self.tag_semantics(is_burn_in=True)

            self.root = copy.deepcopy(original_root)
            self.tree = copy.deepcopy(origianl_tree)
            iteration += 1
        return 


# Module is intended to be imported and used by the application.