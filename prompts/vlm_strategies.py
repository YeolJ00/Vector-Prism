import os
import tempfile
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
from utils.setup import setup_browser
from prompts.llm_templates import (
    RENDER_SYSTEM_PROMPT,
    RENDER_VANILLA_PROMPT,
    RENDER_GROUP_PROMPT,
    RENDER_HIGHLIGHT_PROMPT,
    RENDER_BBOX_PROMPT,
    RENDER_ZOOM_PROMPT,
    RENDER_ZOOMHIGHLIGHT_PROMPT,
    RENDER_OUTLINE_PROMPT,
    RENDER_HTML_SCRIPT,
    RENDER_JS_SCRIPT,
)

def np_print(arr):
    return ", ".join([f"{x:.3f}" for x in arr])

class AgreementMatrix:
    def __init__(self, num_vlms, logger):
        self.num_vlms = num_vlms
        self.A = np.zeros((num_vlms, num_vlms))
        self.total_rounds = 0  # Track number of decision rounds, not pairwise comparisons
        self.logger = logger
   
    def set_classes(self, classes):
        self.K = classes

    def update_and_decide(self, vlm_predictions):
        """
        vlm_predictions: list of labels from each VLM (length = n).
        Returns the chosen class label (weighted by multiclass log-odds).
        """
        if len(vlm_predictions) != self.num_vlms:
            raise ValueError("Number of predictions must match the number of VLMs.")

        N, K, eps = self.num_vlms, self.K, 1e-6

        # Update pairwise agreement counts
        for i in range(N):
            for j in range(i + 1, N):
                if vlm_predictions[i] == vlm_predictions[j]:
                    self.A[i, j] += 1.0
                    self.A[j, i] += 1.0
        self.total_rounds += 1

        B = self.A / self.total_rounds
        off = ~np.eye(N, dtype=bool)
        B[off] -= 1.0 / K
        B[~off] = 0.0

        eigenvalues, eigenvectors = np.linalg.eigh(B)
        # self.logger.info(f"[Eigenvalues] Matrix B: " + np_print(eigenvalues))
        x = eigenvectors[:, np.argmax(eigenvalues)]
        x = -x if np.sum(x) < 0 else x  # Ensure positive sum

        # Eigen vector -> Probabilities (p_hat = 1/K + scale * x)
        positive_mask, negative_mask = x > 0, x < 0
        pos_scale_cap = np.min((1 - eps - 1/K) / x[positive_mask]) if positive_mask.any() else np.inf
        neg_scale_cap = np.min((1/K - eps) / (-x[negative_mask])) if negative_mask.any() else np.inf
        scale_cap = min(pos_scale_cap, neg_scale_cap)
        scale = 0.95 * scale_cap if np.isfinite(scale_cap) and scale_cap > 0 else 0.0
        p_hat = np.clip(1/K + scale * x, eps, 1 - eps)

        # Log-odds weights -> Weighted voting
        w = np.log(((K - 1) * p_hat) / (1 - p_hat + eps))
        self.logger.info(f"[Weights] VLM weights: " + np_print(w))
        scores = defaultdict(float)
        for i, y in enumerate(vlm_predictions):
            scores[y] += w[i]

        return max(scores, key=scores.get)

class BaseAnalysis(ABC):
    """Base class for SVG element analysis strategies."""
    
    def __init__(self):
        pass

    def set_namespace(self, ns):
        """Set the SVG namespace for this strategy."""
        self.ns = ns
    
    def __str__(self):
        return self.__class__.__name__

    @abstractmethod
    def get_system_message(self):
        """Return the system message for this strategy."""
        pass
    
    @abstractmethod
    def get_prompt_template(self):
        """Return the prompt template for this strategy."""
        pass

    @abstractmethod
    def get_group_template(self):
        """Return the group template for this strategy."""
        pass

    @abstractmethod
    def analyze(self, svg_copy, svg_element=None):
        """Main analysis method that coordinates the strategy."""
        pass

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

    def add_grid_to_svg(self, svg_element):
        """Add a crisp grid background to an SVG element."""
        # Get SVG dimensions from viewBox or width/height
        if 'viewBox' in svg_element.attrib:
            viewbox = svg_element.attrib['viewBox'].split()
            min_x, min_y = float(viewbox[0]), float(viewbox[1])
            width, height = float(viewbox[2]), float(viewbox[3])
        else:
            min_x, min_y = 0, 0
            width = float(svg_element.attrib.get('width', '800').rstrip('px'))
            height = float(svg_element.attrib.get('height', '600').rstrip('px'))
    
        # Use proper namespace
        ns = self.ns if hasattr(self, 'ns') and self.ns else 'http://www.w3.org/2000/svg'
    
        # Create a defs element if it doesn't exist
        defs_element = None
        for child in svg_element:
            if child.tag.endswith('defs'):
                defs_element = child
                break
        if defs_element is None:
            defs_element = ET.SubElement(svg_element, f"{{{ns}}}defs")
        
        # Use a nice round number for grid spacing
        grid_interval = max(2.0, min(width, height) / 10)  # Ensure minimum size
    
        # Create a pattern for the grid
        pattern_id = "grid_pattern"
        pattern = ET.SubElement(defs_element, f"{{{ns}}}pattern")
        pattern.attrib.update({
            'id': pattern_id,
            'width': str(grid_interval),
            'height': str(grid_interval),
            'patternUnits': 'userSpaceOnUse'
        })
    
        # Add vertical line (back to whole pixel coordinates for patterns)
        ET.SubElement(pattern, f"{{{ns}}}line", {
            'x1': '0',
            'y1': '0',
            'x2': '0',
            'y2': str(grid_interval),
            'stroke': '#CCCCCC',
            'stroke-width': '0.5',  # Thinner stroke to reduce blur
            'shape-rendering': 'crispEdges'
        })
    
        # Add horizontal line
        ET.SubElement(pattern, f"{{{ns}}}line", {
            'x1': '0',
            'y1': '0',
            'x2': str(grid_interval),
            'y2': '0',
            'stroke': '#CCCCCC',
            'stroke-width': '0.5',  # Thinner stroke to reduce blur
            'shape-rendering': 'crispEdges'
        })
        
        # Alternative approach: Use a rect with stroke instead of fill pattern
        # This often renders more crisply
        grid_group = ET.Element(f"{{{ns}}}g", {
            'stroke': '#CCCCCC',
            'stroke-width': '0.5',
            'shape-rendering': 'crispEdges'
        })
        
        # Draw vertical lines
        x = min_x
        while x <= min_x + width:
            ET.SubElement(grid_group, f"{{{ns}}}line", {
                'x1': str(x),
                'y1': str(min_y),
                'x2': str(x),
                'y2': str(min_y + height)
            })
            x += grid_interval
        
        # Draw horizontal lines
        y = min_y
        while y <= min_y + height:
            ET.SubElement(grid_group, f"{{{ns}}}line", {
                'x1': str(min_x),
                'y1': str(y),
                'x2': str(min_x + width),
                'y2': str(y)
            })
            y += grid_interval
    
        # Insert the grid group as the first child (instead of pattern)
        svg_element.insert(0, grid_group)
        return svg_element

class VanillaRender(BaseAnalysis):
    """Strategy to render a single SVG element in isolation."""
    
    def __init__(self):
        super().__init__()
    
    def get_system_message(self):
        return RENDER_SYSTEM_PROMPT
    
    def get_prompt_template(self):
        return RENDER_VANILLA_PROMPT
    
    def get_group_template(self):
        return RENDER_GROUP_PROMPT
    
    def analyze(self, svg_copy, svg_element=None):
        """
        Render the specified SVG element in isolation.
        Recreate the SVG structure with just this element, including its nested groups.
        """
        # Find the path from root to target element
        path_to_target = self.find_path_to_element(svg_copy, svg_element)
        
        if not path_to_target:
            return svg_copy
        
        return self.create_svg_on_path(svg_copy, path_to_target)
    
    def find_path_to_element(self, root, target):
        """Find the path from root to target element, preserving hierarchy."""
        def search(element, path):
            # Check if this is our target
            if self.is_same_element(element, target):
                return path + [element]
            
            # Search in children
            for child in element:
                result = search(child, path + [element])
                if result:
                    return result
            return None
        
        return search(root, [])
    
    def create_svg_on_path(self, original_svg, path_to_target):
        """Create a new SVG containing only the target element and its necessary ancestors."""
        if not path_to_target:
            return original_svg
        
        # Create new root (copy of original SVG root attributes)
        new_root = ET.Element(original_svg.tag, dict(original_svg.attrib))
        for child in original_svg:
            if not self.is_in_path(child, path_to_target[1:]) and not self.is_drawable_element(child):
                self.deep_copy_element(child, new_root)
        
        current_new = new_root
        for i in range(1, len(path_to_target)):
            element = path_to_target[i]
            
            # Copy the element with its attributes
            new_element = ET.SubElement(current_new, element.tag, dict(element.attrib))
            if element.text:
                new_element.text = element.text
            if element.tail:
                new_element.tail = element.tail
            
            # If this is not the target element (i.e., it's an intermediate group)
            if i < len(path_to_target) - 1:
                for child in element:
                    if (not self.is_in_path(child, path_to_target[i+1:]) and 
                        not self.is_drawable_element(child)):
                        self.deep_copy_element(child, new_element)
            
            current_new = new_element
        
        return new_root
    
    def is_in_path(self, element, path_elements):
        """Check if an element is in the given the path as a list."""
        return any(self.is_same_element(element, path_elem) for path_elem in path_elements)
    
    def deep_copy_element(self, source, parent):
        """Recursively copy an element and all its children."""
        # Create new element
        new_element = ET.SubElement(parent, source.tag, dict(source.attrib))
        if source.text:
            new_element.text = source.text
        if source.tail:
            new_element.tail = source.tail
        
        # Recursively copy all children
        for child in source:
            self.deep_copy_element(child, new_element)
        
        return new_element

class Highlight(BaseAnalysis):
    """Strategy to highlight a specific SVG element."""
    
    def __init__(self):
        super().__init__()
    
    def get_system_message(self):
        return RENDER_SYSTEM_PROMPT
    
    def get_prompt_template(self):
        return RENDER_HIGHLIGHT_PROMPT
    
    def get_group_template(self):
        return RENDER_GROUP_PROMPT
    
    def analyze(self, svg_copy, svg_element=None):
        """Highlight the specified SVG element."""
        self.highlight_target(svg_copy, svg_element)
        
        return svg_copy

    def dim_element(self, element):
        """Dim a single element while preserving its structure."""
        # Parse existing style into a dictionary
        style = element.get('style', '')
        style_dict = {}
        if style:
            for part in style.split(';'):
                if ':' in part:
                    prop, value = part.split(':', 1)
                    style_dict[prop.strip()] = value.strip()
        
        # Get current fill/stroke values
        current_fill = style_dict.get('fill') or element.get('fill', '')
        current_stroke = style_dict.get('stroke') or element.get('stroke', '')
        
        # Apply dimming - REPLACE the values
        if current_fill and current_fill != 'none':
            style_dict['fill'] = '#D0D0D0'
        if current_stroke and current_stroke != 'none':
            style_dict['stroke'] = '#D0D0D0'
        style_dict['opacity'] = '0.2'
        
        # Rebuild style string
        new_style = ';'.join(f"{prop}:{value}" for prop, value in style_dict.items())
        element.set('style', new_style)

    def highlight_target(self, svg_copy, target_element):
        """Recursively find and highlight the target element in the copied SVG."""
        def process_element(element):
            # Check if this is our target element
            if self.is_same_element(element, target_element):
                return True
            
            # Check children first
            target_found_in_children = False
            for child in element:
                if process_element(child):
                    target_found_in_children = True
            
            # If target found in children, keep this element unchanged (it's an ancestor)
            if target_found_in_children:
                return True

            if self.is_drawable_element(element):
                self.dim_element(element)
            
            return False
        
        # Start processing from the root
        process_element(svg_copy)

class BoundingBox(BaseAnalysis):
    """Strategy to compute the bounding box of an SVG element."""
    
    def __init__(self, padding_factor=0.0):
        super().__init__()
        self.browser = setup_browser()
        self.padding_factor = padding_factor
        self.html_content = RENDER_HTML_SCRIPT
        self.js = RENDER_JS_SCRIPT
    
    def get_system_message(self):
        return RENDER_SYSTEM_PROMPT
    
    def get_prompt_template(self):
        return RENDER_BBOX_PROMPT
    
    def get_group_template(self):
        return RENDER_GROUP_PROMPT
    
    def analyze(self, svg_copy, svg_element=None):
        bbox = self.get_bbox(svg_copy, svg_element)
        if bbox:
            self.add_bbox_visualization(svg_copy, bbox)
        return svg_copy
    
    def get_bbox(self, svg_root, element):
        """Get the bounding box of an SVG element using browser."""
        svg_string = ET.tostring(svg_root, encoding='unicode')

        element_id = element.get('id')
        if not element_id:
            element_id = f"target_element_{id(element)}"
            element.set('id', element_id)
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(self.html_content.format(svg_string=svg_string))
            temp_file = f.name
        
        # Load file and execute javascript to get bounding box
        self.browser.get(f'file://{os.path.abspath(temp_file)}')
        bbox = self.browser.execute_script(self.js.format(element_id=element_id))

        os.unlink(temp_file)
        
        return bbox
    
    def add_bbox_visualization(self, svg_root, bbox):
        """Add a visual bounding box rectangle to the SVG."""
        # Padding around the bounding box
        padding = max(bbox['width'], bbox['height']) * self.padding_factor
        x, y = bbox['x'] - padding, bbox['y'] - padding
        width, height = bbox['width'] + 2 * padding, bbox['height'] + 2 * padding

        bbox_rect = ET.Element(f"{{{self.ns}}}rect")
        bbox_rect.set('x', str(x))
        bbox_rect.set('y', str(y))
        bbox_rect.set('width', str(width))
        bbox_rect.set('height', str(height))
        bbox_rect.set('fill', 'none')
        bbox_rect.set('stroke', '#FF0000')
        bbox_rect.set('stroke-width', '1')
        
        svg_root.insert(0, bbox_rect)

class ZoomIn(BoundingBox):
    """Strategy to zoom in on a specific SVG element."""
    
    def __init__(self, padding_factor=0.1):
        super().__init__()
        self.padding_factor = padding_factor
    
    def get_system_message(self):
        return RENDER_SYSTEM_PROMPT

    def get_prompt_template(self):
        return RENDER_ZOOM_PROMPT
    
    def get_group_template(self):
        return RENDER_GROUP_PROMPT
    
    def analyze(self, svg_copy, svg_element=None):
        if svg_element is None:
            return svg_copy
            
        # Get bounding box and apply zoom
        bbox = self.get_bbox(svg_copy, svg_element)
        if bbox:
            self.set_viewbox(svg_copy, bbox)
        
        return svg_copy
    
    def set_viewbox(self, svg_root, bbox):
        """Set the SVG viewBox to zoom in on the bounding box."""
        padding = max(bbox['width'], bbox['height']) * self.padding_factor
        x, y = bbox['x'] - padding, bbox['y'] - padding
        width, height = bbox['width'] + 2 * padding, bbox['height'] + 2 * padding
        
        svg_root.set('viewBox', f"{x} {y} {width} {height}")        

class ZoomInHighlight(ZoomIn, Highlight):
    """Strategy to zoom in and highlight a specific SVG element."""
    
    def __init__(self, padding_factor=0.1):
        super().__init__(padding_factor)
    
    def get_system_message(self):
        return RENDER_SYSTEM_PROMPT
    
    def get_prompt_template(self):
        return RENDER_ZOOMHIGHLIGHT_PROMPT
    
    def get_group_template(self):
        return RENDER_GROUP_PROMPT
 
    def analyze(self, svg_copy, svg_element=None):
        if svg_element is None:
            return svg_copy
            
        # Get bounding box and apply zoom
        self.highlight_target(svg_copy, svg_element)
        bbox = self.get_bbox(svg_copy, svg_element)
        if bbox:
            self.set_viewbox(svg_copy, bbox)
        return svg_copy
    
class Outline(Highlight):
    """Strategy to outline the target element with a prominent stroke."""
    
    def __init__(self,):
        super().__init__()
    
    def get_system_message(self):
        return RENDER_SYSTEM_PROMPT
    
    def get_prompt_template(self):
        return RENDER_OUTLINE_PROMPT
    
    def get_group_template(self):
        return RENDER_GROUP_PROMPT
    
    def analyze(self, svg_copy, svg_element=None):
        if svg_element is None:
            return svg_copy
        self.outline_target(svg_copy, svg_element)

        return svg_copy
    
    def outline_target(self, svg_copy, target_element):
        def process_element(element):
            if self.is_same_element(element, target_element):
                self.outline_elment(element)
                return True
            
            # Check if target is in children
            target_found_in_children = False
            for child in element:
                if process_element(child):
                    target_found_in_children = True
            
            # If target found in children, keep this element unchanged (it's an ancestor)
            if target_found_in_children:
                return True
            
            # If this element is drawable and we want to dim others
            if self.is_drawable_element(element):
                self.dim_element(element)
            
            return False
        
        process_element(svg_copy)
    
    def outline_elment(self, element):
        """Add or enhance the stroke of the target element."""        
        element.set('stroke', '#FF0000')  # Red outline
        element.set('stroke-width', str(1.0))
        element.set('stroke-opacity', '1.0')
        element.set('stroke-linejoin', 'round')
        element.set('stroke-linecap', 'round')
        