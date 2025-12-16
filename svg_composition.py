import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from svg_decomposition import pretty_print_html


class PropertyContext:
    """Tracks inherited properties during tree traversal"""
    
    def __init__(self):
        self.transforms: List[str] = []
        self.classes: List[str] = []
        self.opacity: float = 1.0
        self.fill: Optional[str] = None
        self.stroke: Optional[str] = None
        self.stroke_width: Optional[str] = None
        self.fill_opacity: float = 1.0
        self.stroke_opacity: float = 1.0
        self.stroke_linecap: Optional[str] = None
        self.stroke_linejoin: Optional[str] = None
        self.clip_path: Optional[str] = None
        self.mask: Optional[str] = None
        self.filter: Optional[str] = None
    
    def copy(self):
        ctx = PropertyContext()
        ctx.transforms = self.transforms.copy()
        ctx.classes = self.classes.copy()
        ctx.opacity = self.opacity
        ctx.fill = self.fill
        ctx.stroke = self.stroke
        ctx.stroke_width = self.stroke_width
        ctx.stroke_linecap = self.stroke_linecap
        ctx.stroke_linejoin = self.stroke_linejoin
        ctx.fill_opacity = self.fill_opacity
        ctx.stroke_opacity = self.stroke_opacity
        ctx.clip_path = self.clip_path
        ctx.mask = self.mask
        ctx.filter = self.filter
        return ctx
    
    def get_composed_transform(self):
        if not self.transforms:
            return None
        return ' '.join(self.transforms)

class SemanticElement:
    """Element with all inherited properties baked in"""
    
    def __init__(self, element: ET.Element, original_order: int, semantic_class: str):
        self.element = element
        self.original_order = original_order
        self.semantic_class = semantic_class

class SVGFlattener:
    """Flattens nested SVG with property propagation"""
    
    DRAWABLE_TAGS = {
        'path', 'circle', 'rect', 'ellipse', 'line', 
        'polygon', 'polyline', 'text', 'use', 'image'
    }
    
    INHERITED_ATTRS = {
        'fill', 'stroke', 'stroke-width', 'stroke-linecap', 
        'stroke-linejoin', 'stroke-dasharray', 'stroke-dashoffset',
        'font-family', 'font-size', 'font-weight', 'font-style',
        'text-anchor', 'fill-opacity', 'stroke-opacity'
    }
    
    def __init__(self, svg_string: str, hierarchy_plan: Dict[str, List[str]]):
        """Initialize with SVG string content"""
        self.root = ET.fromstring(svg_string)
        self.flattened: List[SemanticElement] = []
        self.order_counter = 0

        self.semantic_class_names = set(hierarchy_plan.keys())
        for children in hierarchy_plan.values():
            self.semantic_class_names.update(children)
        
    def flatten(self):
        root_context = PropertyContext()
        
        # Create new root SVG with original attributes
        svg_attribs = self.root.attrib.copy()
        new_root = ET.Element('svg', svg_attribs)
        
        # First, copy over non-drawable elements (defs, style, metadata, etc.)
        for child in self.root:
            tag = child.tag.split('}')[-1]
            if tag not in self.DRAWABLE_TAGS and tag != 'g':
                new_root.append(self._deep_copy_element(child))
        
        # Now traverse and flatten drawable elements
        self._traverse(self.root, root_context)
        for sem_elem in self.flattened:
            new_root.append(sem_elem.element)
        
        svg_string = ET.tostring(new_root, encoding='unicode')
        return pretty_print_html(svg_string)

    def _deep_copy_element(self, element: ET.Element):
        new_elem = ET.Element(element.tag, element.attrib.copy())
        new_elem.text = element.text
        new_elem.tail = element.tail
        
        for child in element:
            new_elem.append(self._deep_copy_element(child))
        
        return new_elem

    def _traverse(self, element: ET.Element, context: PropertyContext):
        """Depth-first traversal to preserve paint order"""
        new_context = context.copy()
        self._update_context(element, new_context)
        
        tag = element.tag.split('}')[-1]
        
        if tag in self.DRAWABLE_TAGS:
            self._add_flattened_element(element, new_context)
        
        for child in element:
            self._traverse(child, new_context)
    
    def _update_context(self, element: ET.Element, context: PropertyContext):
        """Update context with element's properties"""
        
        if 'transform' in element.attrib:
            context.transforms.append(element.get('transform'))
        if 'class' in element.attrib:
            classes = element.get('class', '').split()
            context.classes.extend(classes)
        if 'opacity' in element.attrib:
            try:
                context.opacity *= float(element.get('opacity'))
            except ValueError:
                pass
        if 'fill-opacity' in element.attrib:
            try:
                context.fill_opacity *= float(element.get('fill-opacity'))
            except ValueError:
                pass
        if 'stroke-opacity' in element.attrib:
            try:
                context.stroke_opacity *= float(element.get('stroke-opacity'))
            except ValueError:
                pass
        for attr in self.INHERITED_ATTRS:
            if attr in element.attrib:
                value = element.get(attr)
                attr_key = attr.replace('-', '_')
                if hasattr(context, attr_key):
                    setattr(context, attr_key, value)
        if 'clip-path' in element.attrib:
            context.clip_path = element.get('clip-path')
        if 'mask' in element.attrib:
            context.mask = element.get('mask')
        if 'filter' in element.attrib:
            context.filter = element.get('filter')
    
    def _add_flattened_element(self, element: ET.Element, context: PropertyContext):
        """Add element with baked properties"""
        
        # Clone element
        new_elem = ET.Element(element.tag, element.attrib.copy())
        new_elem.text = element.text
        new_elem.tail = element.tail
        
        # Copy children (for text elements with tspan)
        for child in element:
            new_elem.append(child)
        
        composed_transform = context.get_composed_transform()
        if composed_transform:
            elem_transform = new_elem.get('transform', '')
            if elem_transform:
                new_elem.set('transform', f"{composed_transform} {elem_transform}")
            else:
                new_elem.set('transform', composed_transform)
        if context.opacity < 1.0:
            elem_opacity = float(new_elem.get('opacity', '1.0'))
            final_opacity = context.opacity * elem_opacity
            new_elem.set('opacity', f"{final_opacity:.3f}")
        if context.fill_opacity < 1.0:
            elem_fill_opacity = float(new_elem.get('fill-opacity', '1.0'))
            final_fill_opacity = context.fill_opacity * elem_fill_opacity
            if final_fill_opacity < 1.0:
                new_elem.set('fill-opacity', f"{final_fill_opacity:.3f}")
        if context.stroke_opacity < 1.0:
            elem_stroke_opacity = float(new_elem.get('stroke-opacity', '1.0'))
            final_stroke_opacity = context.stroke_opacity * elem_stroke_opacity
            if final_stroke_opacity < 1.0:
                new_elem.set('stroke-opacity', f"{final_stroke_opacity:.3f}")

        if 'fill' not in new_elem.attrib and context.fill:
            new_elem.set('fill', context.fill)
        if 'stroke' not in new_elem.attrib and context.stroke:
            new_elem.set('stroke', context.stroke)
        if 'stroke-width' not in new_elem.attrib and context.stroke_width:
            new_elem.set('stroke-width', context.stroke_width)
        if 'stroke-linecap' not in new_elem.attrib and context.stroke_linecap:
            new_elem.set('stroke-linecap', context.stroke_linecap)
        if 'stroke-linejoin' not in new_elem.attrib and context.stroke_linejoin:
            new_elem.set('stroke-linejoin', context.stroke_linejoin)
        if 'clip-path' not in new_elem.attrib and context.clip_path:
            new_elem.set('clip-path', context.clip_path)
        if 'mask' not in new_elem.attrib and context.mask:
            new_elem.set('mask', context.mask)
        if 'filter' not in new_elem.attrib and context.filter:
            new_elem.set('filter', context.filter)
        
        # Get semantic class from element's class attribute
        element_classes = new_elem.get('class', '').split()
        all_classes = context.classes + element_classes
        
        # Remove duplicates while preserving order
        unique_classes = []
        for cls in all_classes:
            if cls not in unique_classes:
                unique_classes.append(cls)

        # Move semantic classes to the end
        semantic_classes = [cls for cls in unique_classes if cls in self.semantic_class_names]
        for cls in semantic_classes:
            unique_classes.remove(cls)
        unique_classes.extend(semantic_classes)

        if not unique_classes:
            unique_classes.append('')
        
        new_elem.set('class', ' '.join(unique_classes))
        
        flattened_elem = SemanticElement(
            element=new_elem,
            original_order=self.order_counter,
            semantic_class=unique_classes[-1]
        )
        
        self.flattened.append(flattened_elem)
        self.order_counter += 1

class StructuralComposer:
    def __init__(self, svg_string, hierarchy_tree):
        # Parse the SVG string
        root = ET.fromstring(svg_string)
        self.original_svg_attribs = root.attrib.copy()
        self.non_drawable_elements = []
        
        # Extract flattened elements
        self.flattened = []
        order_counter = 0
        
        for element in root:
            tag = element.tag.split('}')[-1]
            
            if tag not in SVGFlattener.DRAWABLE_TAGS:
                self.non_drawable_elements.append(element)
            else:
                classes = element.get('class', '').split()
                semantic_class = classes[-1] if classes else ' '
                
                sem_elem = SemanticElement(
                    element=element, 
                    original_order=order_counter, 
                    semantic_class=semantic_class
                )
                self.flattened.append(sem_elem)
                order_counter += 1
        
        self.hierarchy_tree = hierarchy_tree or {}
        
        # Build index
        self.class_to_elements = defaultdict(list)
        for i, elem in enumerate(self.flattened):
            self.class_to_elements[elem.semantic_class].append(i)
        
        # Precompute bounds for collision detection
        for elem in self.flattened:
            elem.bounds = self._get_element_bounds(elem.element)

        # Build parent-child maps for logical relationships
        self.parent_map = {}  # child -> parent
        self.children_map = defaultdict(list)  # parent -> [children]
        
        for parent, children in self.hierarchy_tree.items():
            self.children_map[parent] = children
            for child in children:
                self.parent_map[child] = parent
    
    def _get_element_bounds(self, element: ET.Element):
        tag = element.tag.split('}')[-1]
        
        try:
            if tag == 'path':
                return self._get_path_bounds(element.get('d', ''))
            elif tag == 'circle':
                cx, cy, r = float(element.get('cx', 0)), float(element.get('cy', 0)), float(element.get('r', 0))
                return (cx - r, cx + r, cy - r, cy + r)
            elif tag == 'rect':
                x, y = float(element.get('x', 0)), float(element.get('y', 0))
                w, h = float(element.get('width', 0)), float(element.get('height', 0))
                return (x, x + w, y, y + h)
            elif tag == 'ellipse':
                cx, cy = float(element.get('cx', 0)), float(element.get('cy', 0))
                rx, ry = float(element.get('rx', 0)), float(element.get('ry', 0))
                return (cx - rx, cx + rx, cy - ry, cy + ry)
            elif tag == 'line':
                x1, y1 = float(element.get('x1', 0)), float(element.get('y1', 0))
                x2, y2 = float(element.get('x2', 0)), float(element.get('y2', 0))
                return (min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2))
        except (ValueError, TypeError):
            pass
        
        return (0, 100, 0, 100) # Default bounds if parsing fails
    
    def _get_path_bounds(self, path_data):
        # Very naive bounding box calculation for paths
        if not path_data:
            return (0, 0, 0, 0)
        
        numbers = re.findall(r'-?\d+\.?\d*', path_data)
        if not numbers:
            return (0, 0, 0, 0)
        
        numbers = [float(n) for n in numbers]
        coords = [(numbers[i], numbers[i + 1]) for i in range(0, len(numbers) - 1, 2)]
        
        if not coords:
            return (0, 0, 0, 0)
        
        xs, ys = [c[0] for c in coords], [c[1] for c in coords]
        
        return (min(xs), max(xs), min(ys), max(ys))
    
    def _overlaps(self, bounds1, bounds2):
        min_x1, max_x1, min_y1, max_y1 = bounds1
        min_x2, max_x2, min_y2, max_y2 = bounds2
        
        return not (max_x1 < min_x2 or min_x1 > max_x2 or 
                   max_y1 < min_y2 or min_y1 > max_y2)
    
    def can_merge_group(self, class_name, indices):
        """Check if elements at indices can be safely grouped (no z-order conflicts)"""
        if len(indices) < 2:
            return True
        
        min_idx, max_idx = min(indices), max(indices)
        target_elements = [self.flattened[i] for i in indices]
        
        # Check for overlaps with intermediate elements of OTHER classes
        for i in range(min_idx + 1, max_idx):
            if i not in indices:
                intermediate_elem = self.flattened[i]
                for target_elem in target_elements:
                    if hasattr(target_elem, 'bounds') and hasattr(intermediate_elem, 'bounds'):
                        if self._overlaps(target_elem.bounds, intermediate_elem.bounds):
                            return False
        
        return True
    
    def find_mergeable_groups(self):
        """Find which elements of each class can be merged together"""
        mergeable_groups = {}
        
        for class_name, indices in self.class_to_elements.items():
            if len(indices) == 1:
                mergeable_groups[class_name] = [indices]
                continue
            
            # Try to merge all
            if self.can_merge_group(class_name, indices):
                mergeable_groups[class_name] = [indices]
            else:
                # Greedy split into minimal groups
                groups = self._find_optimal_split(class_name, indices)
                mergeable_groups[class_name] = groups
        
        return mergeable_groups
    
    def _find_optimal_split(self, class_name: str, indices: List[int]) -> List[List[int]]:
        """Split indices into minimal number of non-overlapping groups"""
        remaining = set(indices)
        groups = []
        
        while remaining:
            current_group = [min(remaining)]
            remaining.remove(min(remaining))
            
            for idx in sorted(remaining):
                if self.can_merge_group(class_name, current_group + [idx]):
                    current_group.append(idx)
            
            for idx in current_group:
                remaining.discard(idx)
            
            groups.append(sorted(current_group))
        
        return groups
    
    def _merge_bounds(self, bounds_list):
        min_x = min(b[0] for b in bounds_list)
        max_x = max(b[1] for b in bounds_list)
        min_y = min(b[2] for b in bounds_list)
        max_y = max(b[3] for b in bounds_list)
        return (min_x, max_x, min_y, max_y)
    
    def build_regrouped_svg(self):
        new_root = ET.Element('svg', self.original_svg_attribs)
        
        # First, add non-drawable elements (style, defs, etc.)
        for elem in self.non_drawable_elements:
            new_root.append(elem)
        
        # Then add the root group with all drawable content
        root_group = ET.SubElement(new_root, 'g')
        root_group.set('class', 'root-animation-group')
        
        # Step 1: Find conflict-free mergeable groups (existing logic)
        mergeable_groups = self.find_mergeable_groups()
        
        # Step 2: Build placement plan with bounds
        placement_plan, group_bounds = [], {}
        for class_name, groups in mergeable_groups.items():
            for group_idx, indices in enumerate(groups):
                elements = [self.flattened[i] for i in indices]
                bounds = self._merge_bounds([e.bounds for e in elements])
                
                group_key = class_name if len(groups) == 1 else f"{class_name}-{group_idx}"
                group_bounds[class_name] = bounds  # Use base name for lookups
                
                placement_plan.append({
                    'order': min(indices),
                    'class': class_name,
                    'group_idx': group_idx,
                    'elements': elements,
                    'total_groups': len(groups),
                    'bounds': bounds
                })
        
        placement_plan.sort(key=lambda x: x['order'])
        
        # Step 3: Build SVG with metadata
        for i, plan in enumerate(placement_plan):
            group = ET.SubElement(root_group, 'g')
            class_name = plan['class']
            is_split = plan['total_groups'] > 1
            
            # Basic attributes
            if is_split:
                group.set('class', f"{class_name}-group-{plan['group_idx']}")
                group.set('data-split', 'true')
            else:
                group.set('class', f'{class_name}-group')

            # # Basic attributes
            # group.set('class', f'{class_name}-group')  # Always use same name
            # if is_split:
            #     group.set('data-split', 'true')  # Keep this flag for metadata
            
            # Core metadata
            group.set('data-semantic-type', class_name)
            group.set('data-paint-order', str(i))
            
            if len(plan['elements']) > 1:
                group.set('data-merged-count', str(len(plan['elements'])))
            
            # Bounds
            bounds = plan['bounds']
            group.set('data-bounds', f'{bounds[0]:.1f},{bounds[2]:.1f},{bounds[1]:.1f},{bounds[3]:.1f}')
            
            # Center point (geometric center of bounds)
            center = ((bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2)
            group.set('data-center', f'{center[0]:.2f},{center[1]:.2f}')

            # Relative center (to top-left of bounds)               
            relative_center = (center[0] - bounds[0], center[1] - bounds[2])
            group.set('data-center-relative', f'{relative_center[0]:.2f},{relative_center[1]:.2f}')


            # Logical relationships from hierarchy tree
            parent = self.parent_map.get(class_name)
            if parent:
                group.set('data-logical-parent', parent)
            
            children = self.children_map.get(class_name, [])
            if children:
                group.set('data-logical-children', ','.join(children))
            
            # Add elements to group
            for elem in sorted(plan['elements'], key=lambda e: e.original_order):
                group.append(elem.element)
        
        ET.indent(new_root, space='  ')
        return ET.tostring(new_root, encoding='unicode')


class CompositionPipeline:
    """Complete pipeline for SVG restructuring with LLM-generated hierarchy"""
    
    def __init__(self, llm, logger):
        """
        Initialize with an LLM for hierarchy generation.
        """
        self.llm = llm
        self.logger = logger
        self.system_prompt = """
        You are a creative and highly skilled animation specialist focused on web-based SVG animations. 
        Your task is to generate clean, efficient, and visually engaging SVG hierarchy structures that facilitate advanced animations.
        """
        self.hierarcy_prompt = """
        You are given a list of semantic class names extracted from an SVG file.
        Your task is to organize these class names into a hierarchical tree structure that reflects their logical relationships for animation purposes.

        This is the list of semantic class names:
        {class_names}

        Guidelines:
        - Group related elements under common parent classes (e.g., 'wing-left' and 'wing-right' under 'body').
        - Use meaningful parent class names that represent the collective function or role of their children.
        - Do not create unnecessary levels of hierarchy; keep it as flat as possible while maintaining logical relationships.
        - Do not create cycles in the hierarchy.

        Please respond in the following JSON format:
        ```json
        {{
            "parent_class_1": ["child_class_a", "child_class_b"],
            "parent_class_2": ["child_class_c"],
            ...
        }}
        ```
        """
    
    def _generate_hierarchy_tree(self, svg_string, plan):
        """
        Use LLM to generate hierarchy tree from SVG and animation plan.
        
        Args:
            svg_string: SVG string content
            plan: Animation plan dictionary containing semantic class names
            
        Returns:
            Hierarchy tree dict mapping parent -> [children]
        """
        # Extract all semantic class names from the plan
        class_names = list(plan.keys())

        formatted_prompt = self.hierarcy_prompt.format(class_names = class_names)
        response = self.llm.invoke(formatted_prompt)

        chat_prompt = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=[
                {'type': 'text', 'text': formatted_prompt},
            ]),
        ]

        # Generate the HTML and CSS code using the LLM
        response = self.llm.invoke(chat_prompt)
        hierarchy_string = response.content.strip().replace('```', '').replace('json', '').strip()
        self.logger.info(f"[SemanticHierarchy]\n{response}")

        try:
            hierarchy_dict = json.loads(hierarchy_string)
        except json.JSONDecodeError as e:
            return self._generate_hierarchy_tree(svg_string, plan)
        
        return hierarchy_dict
    
    def restructure(self, svg_string, plan):
        hierarchy_tree = self._generate_hierarchy_tree(svg_string, plan)
        
        flattener = SVGFlattener(svg_string, plan)
        flattened_svg_string = flattener.flatten()

        composer = StructuralComposer(flattened_svg_string, hierarchy_tree)
        result = composer.build_regrouped_svg()

        # # Debug output
        # with open("debug_flattened.svg", "w", encoding="utf-8") as f:
        #     f.write(flattened_svg_string)
        # with open("debug_restructured.svg", "w", encoding="utf-8") as f:
        #     f.write(result)

        return result

# Module is intended for import; no top-level execution.