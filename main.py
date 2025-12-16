import argparse

from animation_generator import AnimationGenerator
from animation_planner import AnimationPlanner
from animation_state import HTMLPalette
from svg_decomposition import MultiSVGParser
from utils.setup import (get_models, select_svg_file, set_api_key,
                         setup_logging, title_screen)


def argument_parser():
    parser = argparse.ArgumentParser(description="Animated SVG Generator")
    parser.add_argument('--exp_name', type=str, default='test', help='Experiment name')

    parser.add_argument('--test_json', type=str, default="svg/test.jsonl", help='Path to the test JSONL file')
    parser.add_argument('--test_plan_json', type=str, default="logs/plans.jsonl", help='Path to the test plan JSONL file')

    parser.add_argument('--model_name', type=str, default='gpt-5-mini', help='Model name for LLM and VLM')
    parser.add_argument('--model_provider', type=str, default='openai', help='Model provider (e.g., openai, azure)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for model sampling')

    parser.add_argument('--burn_in', type=int, default=2, help='Number of burn-in iterations for SVG parser')
    
    args = parser.parse_args()
    return args

def main(args):
    # Set up logging
    logger = setup_logging(args)
    logger.info("Starting the animation generation process.")
    set_api_key()

    # print("|> Initialize Models")
    llm, vlm = get_models(args.model_name, args.model_provider, args.temperature, logger=logger)

    # print("|> Load SVG File")
    svg_path, instruction = select_svg_file(args.test_json)

    # print("|> Plan Animations")
    planner = AnimationPlanner(vlm, logger, load_plan=args.test_plan_json)
    plans = planner.plan(svg_file=svg_path, instruction=instruction)

    logger.info("|> Tagging Semantics")
    svg_parser = MultiSVGParser(vlm, logger)
    svg_parser.set_parser(svg_path, plans)
    if (svg_string:=svg_parser.load_tagged_svg()) is None:
        svg_parser.burn_in(burn_in_count=args.burn_in)
        svg_string = svg_parser.tag_semantics()

    logger.info("|> Generate HTML and CSS scripts")
    animation_state = HTMLPalette(svg_string, logger)
    animator = AnimationGenerator(vlm, logger)
    # reviewer = AnimationReviewer(vlm, logger)
    for class_name in plans.keys():
        logger.info(f"  |=> Animating {class_name}")
        content = {
            'class_name': class_name,
            'viewbox': animation_state.get_viewbox(),
            'animation_plan': plans[class_name],
            'big_picture': svg_parser.full_svg_base64,
            'little_picture': svg_parser.render_by_class(class_name),
        }
        css = animator.animate(content, animation_state.get_current_html())
        animation_state.merge(css)

    animation_state.save_to_html()


if __name__ == "__main__":
    title_screen()
    args = argument_parser()
    main(args)
