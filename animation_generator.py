from langchain_core.messages import HumanMessage, SystemMessage
from prompts.llm_templates import GENERATOR_SYSTEM_PROMPT, GENERATOR_PROMPT

class AnimationGenerator:
    def __init__(self, vlm, logger):
        self.vlm = vlm
        self.logger = logger

        self.system_prompt = GENERATOR_SYSTEM_PROMPT
        self.animation_prompt_template = GENERATOR_PROMPT

    def animate(self, content, previous_html):
        formatted_prompt = self.animation_prompt_template.format(
            class_name=content['class_name'],
            viewbox=content['viewbox'],
            animation_plan=content['animation_plan'],
            previous_html=previous_html,
        )

        image_payload = [
            {'type': 'image_url', 'image_url': {"url": f"data:image/jpeg;base64,{content['big_picture']}"}},
            {'type': 'image_url', 'image_url': {"url": f"data:image/jpeg;base64,{content['little_picture']}"}},
        ]

        chat_prompt = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=[
                {'type': 'text', 'text': formatted_prompt},
                *image_payload,
            ]),
        ]

        # Generate the HTML and CSS code using the LLM
        response = self.vlm.invoke(chat_prompt)
        response = response.content.replace("```html", "").replace("```", "").strip()
        self.logger.info(f"[AnimationGenerator]\n{response}")

        try:
            # Extract the CSS code from the response
            css_code = response.split("<style>")[1].split("</style>")[0].strip()
            return css_code
        except IndexError:
            return self.animate(content, previous_html)

# Module is intended to be imported and used by `main.py` or other callers.