#######################################################
# ------------------ VLM TEMPLATES ------------------ #
#######################################################

PARSER_SYSTEM_PROMPT = \
"""You are an expert who can analyze SVG files and categorize SVG elements based on their semantics."""

PARSER_PROMPT = \
"""You will be shown two images:

1. The first image is a rendering of the full SVG file.
2. The second image is a rendering of an SVG element within the full SVG, while other elements are dimmed to gray (#D0D0D0).

Your task is to analyze the SVG element (shown in the second image) in the context of the full SVG, and determine its semantic category.
The second image can be a component of a larger semantic category in the list.
In that case, please select the semantic category that encompasses the element in the context of the full SVG.

Choose one category from the following list:
{semantic_categories}
{group}

Please respond in the following JSON format:
```json
{{
    "category": "semantic category",
    "reasoning": "Your reasoning for the category selection"
}}
```
"""

PARSER_GROUP_PROMPT = \
"""If the element is a combination of several semantic categories in the list, respond with "group".
Keep in mind that SVGs often have combined outlines of multiple elements.
You should categorize these elements as a "group" if they outline multiple semnatic categories.
"""

PLANNER_SYSTEM_PROMPT = \
"""You are a talented animator. 
Your task is make detailed plans for each elements in the SVG image to make a lively animation. """

PLANNER_PROMPT = \
"""This is an SVG image.

Your task is to generate animation plans for the individual elements in the image based on the following instruction:
{instruction}

The goal is to create a **high-quality, smooth, and visually engaging SVG animation** suitable for web display.

Please follow these guidelines:
- **Animate elements individually or in thoughtfully grouped sets**. Each group should share similar motion or timing.
- **Avoid awkward or robotic movement** unless intentional. The animation should feel natural and dynamic.
- An element can be animated through changes in attributes like position, scale, rotation, color, opacity, path, etc.
- In case elements of similar type (e.g. trees, stars, clouds) is expected to have different animations, treat them as **distinct elements** (e.g., `left_tree`, `foreground_star`, `background_cloud`), but if they should share an animation, treat them as a single element. 
- Keep the number of elements in a manageable range (e.g., 5-10) so that the animation is not overly complex.
- Avoid using generic or SVG/HTML tag-based names (e.g., `circle`, `rect`, `path`, `body`). Instead, use meaningful identifiers based on position or visual role. Also do not use special characters that could interfere with JSON formatting or directory paths (e.g., `#`, `.`, `/`, `\\`, `:`). 
- If an element has no animation, it should be explicitly stated as such.
- Please do not plan accessibility features or interactivity.
- Do not include any runtime-only classes (e.g., .impact, .flight, .play) in your plan. The animation must animate immediately on page load with no manual steps or JS triggers.

Please respond in the following JSON format:
```json
{{
    element_name_1: Animation plan for element_name_1,
    element_name_2: Animation plan for element_name_2,
    ...
    element_name_n: Animation plan for element_name_n,
}}
```
"""

GENERATOR_SYSTEM_PROMPT = \
"""You are a creative and highly skilled animation specialist focused on web-based SVG animations. 
Your task is to generate clean, efficient, and visually engaging CSS code that brings animation plans to life. 
Animations should be smooth, expressive, and well-suited for modern, interactive web environments. 
"""

GENERATOR_PROMPT = \
"""You are a CSS animation expert tasked with creating animations for SVG elements.
The first image is the entire SVG file.
We have animated the following elements in the SVG:
```html
{previous_html}
```

Now, we are currently focusing on animating the `{class_name}` class within the SVG, which is rendered in the second image.

Animation plan for `{class_name}` is as follows:
{animation_plan}

Please generate CSS animation code for the SVG element with class '{class_name}'.

Requirements:
- Create keyframe animations that are timed and executed harmoniously with existing animations in the SVG.
- Animation should be smooth, optimized, and appropriate for web performance.
- Style should be elegant and subtle unless dramatic effects are specifically requested.
- Avoid naming conflicts with existing keyframes or animation properties.
- Include compact comments regarding coherence with other animations where relevant.
- Coordinates and transform origins must be derived based on the actual layout of the entire SVG. Account for the spatial relationship between {class_name} and other animated elements to avoid visual collisions, clipping, or misalignment. Use relative positions where appropriate.
- Refrain from modifying or duplicating any existing CSS code.
- Be mindful of the performance implications of your animations, especially for complex SVGs with multiple animated elements.
- Make all animations self-contained. Do NOT gate keyframes behind runtime-only classes (e.g., .impact, .flight, .play). The delivered file must animate immediately on page load with no manual steps.

Collision avoidance considerations:
- Never write 'transform' inside @keyframes. Write Custom properties only.
- Use the lanes pattern with these naming convention: --{class_name}-tx1/tx2, --{class_name}-ty1/ty2, --{class_name}-rot1/rot2, --{class_name}-sx1/sx2, --{class_name}-sy1/sy2, --{class_name}-op1/op2, --{class_name}-blur1/blur2, --{class_name}-stroke1/stroke2, --{class_name}-bright1/bright2.
- If these @property declarations or the .{class_name} composer rule are missing, add them ONCE.
- Put new motion on the next free lane(s). Do NOT edit existing lanes.
- Use animation-* longhand. If multiple animations, provide comma-separated lists with aligned indexes.

Please respond in the following format:
```html
<style>
    /* CSS code goes here */
</style>
```
"""

########################################################
# --------------- Evaluation Templates --------------- #
########################################################

VIDEO_QUALITY_PROMPT = \
"""You are evaluating whether a video matches a given text description.

Text Description: "{text_prompt}"

Here are frames sampled from the video. Evaluate how well it depicts the given description.

Where the score represents:
- 90-100: Perfect match, video clearly depicts the description
- 70-89: Good match, most elements are present
- 50-69: Partial match, some elements are present
- 30-49: Weak match, few elements present
- 0-29: No match, video does not depict the description.

Provide your response in EXACTLY this json format:
```json
{{
    "score": score_value,
    "reasoning": "brief explanation"
}}
```
"""

VIDEO_BINARY_PROMPT = \
"""You are evaluating whether a video depicts a given text description.

Text Description: "{text_prompt}"

Here are frames sampled from the video. Based on these frames, does the video depict the description?

Provide your response in EXACTLY this json format:
```json
{{
    "answer": "Yes" or "No",
    "reasoning": "brief explanation"
}}
```
"""

########################################################
# ------------------ VLM STRATEGIES ------------------ #
########################################################

RENDER_SYSTEM_PROMPT = \
"""You are an expert who can analyze SVG files and categorize SVG elements based on their semantics."""

RENDER_VANILLA_PROMPT = \
"""You will be shown two images:

1. The first image is a rendering of the full SVG file.
2. The second image is a rendering of a specific SVG element within the full SVG.

Your task is to analyze the SVG element (shown in the second image) in the context of the full SVG, and determine its semantic category.
The second image can be a component of a larger semantic category in the list.
In that case, please select the semantic category that encompasses the element in the context of the full SVG.
If there are no visible bounding boxes, it means the element is not visible in the SVG, respond with "group".

Choose one category from the following list:
{semantic_categories}
{group}

Please respond in the following JSON format:
```json
{{
    "category": "semantic category",
    "reasoning": "Your reasoning for the category selection"
}}
```
"""

RENDER_GROUP_PROMPT = \
"""If the element is a combination of several semantic categories in the list, respond with "group".
Keep in mind that SVGs often have combined outlines of multiple elements.
You should categorize these elements as a "group" if they outline multiple semnatic categories.
"""

RENDER_HIGHLIGHT_PROMPT = \
"""You will be shown two images:

1. The first image is a rendering of the full SVG file.
2. The second image is a rendering of an SVG element within the full SVG, while other elements are dimmed to gray (#D0D0D0).

Your task is to analyze the SVG element (shown in the second image) in the context of the full SVG, and determine its semantic category.
The second image can be a component of a larger semantic category in the list.
In that case, please select the semantic category that encompasses the element in the context of the full SVG.

Choose one category from the following list:
{semantic_categories}
{group}

Please respond in the following JSON format:
```json
{{
    "category": "semantic category",
    "reasoning": "Your reasoning for the category selection"
}}
```
"""

RENDER_BBOX_PROMPT = \
"""You will be shown two images:

1. The first image is a rendering of the full SVG file.
2. The second image highlights a specific SVG element within the full SVG, using a red bounding box around it.

Your task is to analyze the SVG element (shown in the second image) in the context of the full SVG, and determine its semantic category.
The second image can be a component of a larger semantic category in the list.
In that case, please select the semantic category that encompasses the element in the context of the full SVG.
If there are no visible bounding boxes, it means the element is not visible in the SVG, respond with "group".

Choose one category from the following list:
{semantic_categories}
{group}

Please respond in the following JSON format:
```json
{{
    "category": "semantic category",
    "reasoning": "Your reasoning for the category selection"
}}
```
"""

RENDER_ZOOM_PROMPT = \
"""You will be shown two images:

1. The first image is a rendering of the full SVG file.
2. The second image zooms in on a specific SVG element within the full SVG. 

Your task is to analyze the SVG element (shown in the second image) in the context of the full SVG, and determine its semantic category.
The second image can be a component of a larger semantic category in the list.
In that case, please select the semantic category that encompasses the element in the context of the full SVG.

Choose one category from the following list:
{semantic_categories}
{group}

Please respond in the following JSON format:
```json
{{
    "category": "semantic category",
    "reasoning": "Your reasoning for the category selection"
}}
```
"""

RENDER_ZOOMHIGHLIGHT_PROMPT = \
"""You will be shown two images:

1. The first image is a rendering of the full SVG file.
2. The second image zooms in on a specific SVG element within the full SVG. Other elements are dimmed to gray (#D0D0D0).

Your task is to analyze the SVG element (shown in the second image) in the context of the full SVG, and determine its semantic category.
The second image can be a component of a larger semantic category in the list.
In that case, please select the semantic category that encompasses the element in the context of the full SVG.

Choose one category from the following list:
{semantic_categories}
{group}

Please respond in the following JSON format:
```json
{{
    "category": "semantic category",
    "reasoning": "Your reasoning for the category selection"
}}
```
"""

RENDER_OUTLINE_PROMPT = \
"""
You will be shown two images:

1. The first image is a rendering of the full SVG file.
2. The second image shows the same SVG where one element has a prominent outline/stroke highlighting it.

Your task is to analyze the outlined SVG element in the context of the full SVG, and determine its semantic category.
The outlined element can be a component of a larger semantic category in the list.
In that case, please select the semantic category that encompasses the element in the context of the full SVG.

Choose one category from the following list:
{semantic_categories}
{group}

Please respond in the following JSON format:
```json
{{
    "category": "semantic category",
    "reasoning": "Your reasoning for the category selection"
}}
```
"""



###########################################################
# ------------------ RENDERING SCRIPTS ------------------ #
############################################################

RENDER_HTML_SCRIPT = \
"""<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ margin: 0; padding: 0; }}
        svg {{ display: block; }}
    </style>
</head>
<body>
    {svg_string}
</body>
</html>
"""

RENDER_JS_SCRIPT = \
"""
const element = document.getElementById('{element_id}');
if (!element) return null;

try {{
    const rect = element.getBoundingClientRect();
    const svg = element.ownerSVGElement || element.closest('svg');
    
    if (!svg) {{
        // Fallback for non-SVG elements
        return {{
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height,
            right: rect.right,
            bottom: rect.bottom
        }};
    }}
    
    const svgRect = svg.getBoundingClientRect();
    
    // Convert screen coordinates to SVG coordinate system
    const svgPoint1 = svg.createSVGPoint();
    svgPoint1.x = rect.left - svgRect.left;
    svgPoint1.y = rect.top - svgRect.top;
    
    const svgPoint2 = svg.createSVGPoint();
    svgPoint2.x = rect.right - svgRect.left;
    svgPoint2.y = rect.bottom - svgRect.top;
    
    // Transform to SVG coordinates (handles all nested transforms)
    const ctm = svg.getScreenCTM();
    if (ctm) {{
        const inverse = ctm.inverse();
        const transformedPoint1 = svgPoint1.matrixTransform(inverse);
        const transformedPoint2 = svgPoint2.matrixTransform(inverse);
        
        const x = Math.min(transformedPoint1.x, transformedPoint2.x);
        const y = Math.min(transformedPoint1.y, transformedPoint2.y);
        const width = Math.abs(transformedPoint2.x - transformedPoint1.x);
        const height = Math.abs(transformedPoint2.y - transformedPoint1.y);
        
        return {{
            x: x,
            y: y,
            width: width,
            height: height,
            right: x + width,
            bottom: y + height,
            method: 'svg_transform'
        }};
    }}
    
    // Fallback to getBBox if transform fails
    const bbox = element.getBBox();
    return {{
        x: bbox.x,
        y: bbox.y,
        width: bbox.width,
        height: bbox.height,
        right: bbox.x + bbox.width,
        bottom: bbox.y + bbox.height,
        method: 'fallback_bbox'
    }};
    
}} catch (e) {{
    return null;
}}
"""