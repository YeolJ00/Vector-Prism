from bs4 import BeautifulSoup
from svg_decomposition import pretty_print_html

class HTMLPalette:
    """
    This class merges the HTML and CSS code into a single HTML file.
    It also includes the SVG code inside the HTML file.
    """
    def __init__(self, svg_string, logger):
        # Initialize the HTML and CSS code
        self.main_branch = f"""
        <!DOCTYPE html>
        <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>SVG Animation</title>
                <style>
                    /* CSS code will be added here */
                </style>
            </head>
            <body>
                {svg_string}
            </body>
        </html>
        """
        self.initial_branch = self.main_branch
        self.logger = logger

    def get_viewbox(self):
        """
        Get the viewbox dimensions from the SVG code.
        
        Returns:
            tuple: A tuple containing the x, y, width, and height of the viewbox.
        """
        soup = BeautifulSoup(self.main_branch, 'html.parser')
        svg_tag = soup.find('svg')
        if svg_tag and 'viewBox' in svg_tag.attrs:
            viewbox = svg_tag['viewBox'].split()
            x, y, width, height = map(float, viewbox)
            return (x, y, width, height)
        else:
            return (0, 0, 8, 8)

    def get_current_html(self):
        """
        Get the current HTML code from the HTML file.
        
        Returns:
            str: The current HTML code.
        """
        soup = BeautifulSoup(self.main_branch, 'html.parser')
        return str(soup)

    def get_current_css(self):
        """
        Get the current CSS code from the HTML file.
        
        Returns:
            str: The current CSS code.
        """
        soup = BeautifulSoup(self.main_branch, 'html.parser')
        style_tag = soup.find('style')
        return style_tag.string if style_tag else ""

    def commit(self, css_string):

        html = self.initial_branch.replace(
            "/* CSS code will be added here */",
            f"/*Newly added CSS code */\n{css_string}"
        )

        # Pretty print the HTML
        html = pretty_print_html(html, parser='html.parser')

        return html

    def merge(self, css_string):
        """
        Merge the HTML and CSS code into a single HTML file.
        This alters the instance variables.

        The CSS code is added inside <style> tags. 
        The SVG code is added inside the <svg> tag.
        Args:
            svg_string (str): The generated HTML code for the animation.
            css_string (str): The generated CSS code for the animation.
        Returns:
            None
        """
        html = self.main_branch.replace(
            "/* CSS code will be added here */",
            f"\n{css_string}\n/* CSS code will be added here */"
        )

        # Pretty print the HTML
        html = pretty_print_html(html, parser='html.parser')
        self.main_branch = html

        return
    
    def save_to_html(self):
        file_path = self.logger.exp_dir + "/result.html"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.main_branch)
        self.logger.info(f"[HTMLPalette] Saved HTML to {file_path}")
        
        return