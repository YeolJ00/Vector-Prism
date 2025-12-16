import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime
import math

import jsonlines
from langchain.chat_models import init_chat_model
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

def title_screen():
    title = r"""
    ___    __          _____                  ________       _____                   
    __ |  / /____________  /______________    ___  __ \_________(_)_____________ ___ 
    __ | / /_  _ \  ___/  __/  __ \_  ___/    __  /_/ /_  ___/_  /__  ___/_  __ `__ \
    __ |/ / /  __/ /__ / /_ / /_/ /  /        _  ____/_  /   _  / _(__  )_  / / / / /
    _____/  \___/\___/ \__/ \____//_/         /_/     /_/    /_/  /____/ /_/ /_/ /_/                                                                                                                                                    
    """
    print(title)

def print_fixed_columns(items, num_cols=3, col_width=None):
    """Print items in a fixed number of columns, filling vertically first."""
    if not items:
        return
    
    col_width = max(len(str(item)) for item in items) + 2 if col_width is None else col_width
    rows_per_col = math.ceil(len(items) / num_cols)
    for row in range(rows_per_col):
        line_parts = []
        for col in range(num_cols):
            index = col * rows_per_col + row
            if index < len(items):
                item_text = str(items[index])
                if len(item_text) > col_width: # Truncate if too long
                    item_text = item_text[:col_width-3] + "..."
            else:
                item_text = "" 
            
            if col < num_cols - 1:
                line_parts.append(f"{item_text:<{col_width}} | ")
            else:
                line_parts.append(item_text)
        
        print("".join(line_parts).rstrip())

def set_api_key():
    # Set your API keys here
    # This is just a placeholder; replace with your actual API key
    # os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"
    # os.environ["ANTHROPIC_API_KEY"] = "your_anthropic_api_key_here"
    pass

def get_models(model_name='gpt-5-mini', model_provider='openai', temperature=1.0, logger=None):
    vlm = init_chat_model(model_name, model_provider=model_provider, temperature=temperature)
    llm = init_chat_model(model_name, model_provider=model_provider, temperature=temperature)
    if logger:
        logger.info(f"VLM: {model_provider} {model_name} @ {temperature}")
        logger.info(f"LLM: {model_provider} {model_name} @ {temperature}")
    return llm, vlm

def setup_logging(args):
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    exp_dir = f"logs/{args.exp_name}-{current_time}"
    os.makedirs(exp_dir, exist_ok=True)

    logging.basicConfig(
        filename=f"{exp_dir}/info.log",
        filemode='a',
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('langchain').setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.exp_dir = exp_dir
    logger.exp_name = args.exp_name
    
    return logger

def select_svg_file(test_json):
    svg_instructions = {}
    with jsonlines.open(test_json, 'r') as reader:
        for item in reader:
            svg_instructions.setdefault(item['svg_file'], []).append(item['instruction'])

    print("Please select an SVG file from the following options:")
    svg_files = list(svg_instructions.keys())
    print_fixed_columns([f"{i+1:2d}. {file}" for i, file in enumerate(svg_files)], num_cols=3, col_width=50)
    choice = int(input("Enter the number corresponding to your choice: ")) - 1
    svg_path = list(svg_instructions.keys())[choice]

    print("Please select an instruction from the following options:")
    for i, instruction in enumerate(svg_instructions[svg_path], start=1):
        print(f"{i}. {instruction}")
    instruction_choice = int(input("Enter the number corresponding to your choice: ")) - 1
    instruction = svg_instructions[svg_path][instruction_choice]

    return svg_path, instruction

@contextmanager
def suppress_selenium_logs():
    """Completely suppress all Selenium and Chrome output."""
    # Redirect stdout and stderr to devnull
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

@suppress_selenium_logs()
def setup_browser():
    options = Options()
    
    # Headless and performance options
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    
    # Disable EVERYTHING
    options.add_argument('--disable-logging')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-web-security')
    options.add_argument('--disable-features=VizDisplayCompositor,TranslateUI,BlinkGenPropertyTrees')
    options.add_argument('--disable-ipc-flooding-protection')
    options.add_argument('--disable-background-timer-throttling')
    options.add_argument('--disable-backgrounding-occluded-windows')
    options.add_argument('--disable-renderer-backgrounding')
    options.add_argument('--disable-field-trial-config')
    options.add_argument('--disable-back-forward-cache')
    options.add_argument('--disable-background-networking')
    options.add_argument('--disable-breakpad')
    options.add_argument('--disable-component-extensions-with-background-pages')
    options.add_argument('--disable-client-side-phishing-detection')
    options.add_argument('--disable-default-apps')
    options.add_argument('--disable-dev-tools')
    options.add_argument('--disable-hang-monitor')
    options.add_argument('--disable-prompt-on-repost')
    options.add_argument('--disable-sync')
    options.add_argument('--metrics-recording-only')
    options.add_argument('--no-default-browser-check')
    options.add_argument('--no-first-run')
    options.add_argument('--password-store=basic')
    options.add_argument('--use-mock-keychain')
    options.add_argument('--log-level=3')
    options.add_argument('--silent')
    options.add_argument('--quiet')
    
    # Disable specific features that cause startup messages
    options.add_argument('--disable-features=AudioServiceOutOfProcess')
    options.add_argument('--disable-features=MediaRouter')
    options.add_argument('--disable-speech-api')
    options.add_argument('--disable-voice-input')
    
    # Even more suppression
    options.add_experimental_option('excludeSwitches', [
        'enable-automation', 
        'enable-logging',
        'enable-blink-features=AutomationControlled'
    ])
    options.add_experimental_option('useAutomationExtension', False)
    
    # Suppress Chrome logs completely
    service = Service(log_path=os.devnull)
    
    browser = webdriver.Chrome(service=service, options=options)
    
    # Disable logging in browser console too
    browser.execute_cdp_cmd('Log.enable', {})
    browser.execute_cdp_cmd('Runtime.enable', {})

    return browser

