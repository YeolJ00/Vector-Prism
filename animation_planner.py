import json
import os

import jsonlines
from langchain_core.messages import HumanMessage, SystemMessage

from svg_decomposition import svg_to_png
from prompts.llm_templates import PLANNER_SYSTEM_PROMPT, PLANNER_PROMPT


class AnimationPlanner:
    def __init__(self, vlm, logger, load_plan=None):
        self.vlm = vlm
        self.logger = logger
        self.test_plan_dir = load_plan

        self.system_prompt = PLANNER_SYSTEM_PROMPT
        self.animation_prompt_template = PLANNER_PROMPT

    def set_svg_file(self, svg_file):
        """Set the SVG file for the animation planner."""
        self.svg_file = svg_file
        self.base64_svg = svg_to_png(svg_file)

    def load(self, svg_file, instruction):
        """Load an existing animation plan from a JSON file."""
        if not self.test_plan_dir or not os.path.exists(self.test_plan_dir):
            return None
        with jsonlines.open(self.test_plan_dir, 'r') as reader:
            for item in reader:
                if item['svg_file'] == svg_file and item['instruction'] == instruction:
                    return item['plan']
        return None

    def sort_jsonl_file(self):
        """Sort the JSONL file by 'svg_file' key before saving."""
        if not self.test_plan_dir or not os.path.exists(self.test_plan_dir):
            return
        
        # Read the existing JSONL content
        with jsonlines.open(self.test_plan_dir, 'r') as reader:
            data = list(reader)
        
        # Sort the data by 'svg_file' in ascending order (A-Z)
        data.sort(key=lambda x: x['svg_file'])
        
        # Write the sorted data back to the file
        with jsonlines.open(self.test_plan_dir, 'w') as writer:
            for item in data:
                writer.write(item)

    def save(self, svg_file, instruction, plan):
        """Save the current animation plan to a JSON file."""
        if not self.test_plan_dir:
            return
        if os.path.exists(self.test_plan_dir):
            with jsonlines.open(self.test_plan_dir, 'r') as reader:
                for item in reader:
                    if item['svg_file'] == svg_file and item['instruction'] == instruction:
                        # Already exists, do not duplicate
                        return
        with jsonlines.open(self.test_plan_dir, 'a') as writer:
            writer.write({
                'svg_file': svg_file,
                'instruction': instruction,
                'plan': plan
            })
            
        self.sort_jsonl_file()

    def plan(self, svg_file, instruction):
        if self.test_plan_dir:
            animation_plans = self.load(svg_file, instruction)
            if animation_plans:
                return animation_plans

        self.logger.info(f"[AnimationPlanner] Planning animation for instruction:\n{instruction}")
        self.set_svg_file(svg_file)

        formatted_prompt = self.animation_prompt_template.format(
            instruction=instruction
        )
        image_payload = [{'type': 'image_url', 'image_url': {"url": f"data:image/jpeg;base64,{self.base64_svg}"}},]
        chat_prompt = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=[
                {'type': 'text', 'text': formatted_prompt},
                *image_payload
            ])
        ]

        response = self.vlm.invoke(chat_prompt)
        animation_plan_string = response.content.strip().replace('```', '').replace('json', '').strip()
        self.logger.info(f"[AnimationPlanner]\n{animation_plan_string}")

        try:
            animation_plans = json.loads(animation_plan_string)
        except json.JSONDecodeError as e:
            return self.plan(svg_file, instruction)
        
        self.save(svg_file, instruction, animation_plans)
        
        return animation_plans

# Module is intended to be imported and used by other scripts; no CLI here.