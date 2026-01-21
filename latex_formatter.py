"""
ScribeTeX LaTeX Formatter
Intelligent formatting and cleanup for LaTeX code.
"""
import re
import logging
from typing import List

logger = logging.getLogger(__name__)


class LaTeXFormatter:
    """
    An intelligent formatter for LaTeX code that understands document structure.
    """
    
    def __init__(self):
        # Environments where internal content should not be indented
        self.no_internal_indent = {
            'equation', 'equation*', 'align', 'align*', 'verbatim',
            'verbatim*', 'gather', 'gather*', 'multline', 'multline*',
            'lstlisting', 'minted'
        }
        self.indent_space = '    '

    def format_latex(self, latex_string: str) -> str:
        """Apply cleaning and formatting rules to raw LaTeX string."""
        logger.debug("Starting LaTeX formatting process")
        
        cleaned_latex = self._clean_input(latex_string)
        lines = cleaned_latex.split('\n')
        formatted_lines = self._format_lines(lines)
        joined_latex = '\n'.join(formatted_lines)
        final_latex = self._apply_regex_spacing(joined_latex)
        
        # Final cleanup for excessive newlines
        final_latex = re.sub(r'\n{3,}', '\n\n', final_latex).strip() + '\n'
        
        logger.debug("Finished LaTeX formatting process")
        return final_latex

    def _clean_input(self, latex_string: str) -> str:
        """Remove markdown fences and normalise newlines."""
        latex_string = latex_string.strip()
        
        # Remove markdown code fences
        if latex_string.startswith("```latex"):
            latex_string = latex_string[len("```latex"):].strip()
        elif latex_string.startswith("```"):
            latex_string = latex_string[len("```"):].strip()
        
        if latex_string.endswith("```"):
            latex_string = latex_string[:-len("```")].strip()
        
        # Normalise line endings
        latex_string = latex_string.replace('\r\n', '\n').replace('\r', '\n')
        
        return latex_string

    def _format_lines(self, lines: List[str]) -> List[str]:
        """Format indentation using a stack to track nested environments."""
        formatted_lines = []
        env_stack = []
        
        begin_env_pattern = re.compile(r'^\s*\\begin\{([\w\*]+)\}')
        end_env_pattern = re.compile(r'^\s*\\end\{([\w\*]+)\}')

        for line in lines:
            stripped_line = line.strip()
            
            if not stripped_line:
                # Preserve single blank lines
                if formatted_lines and formatted_lines[-1].strip():
                    formatted_lines.append('')
                continue

            # Check for \end{} - reduce indent before this line
            end_match = end_env_pattern.match(stripped_line)
            if end_match:
                if env_stack:
                    env_stack.pop()

            current_indent_level = len(env_stack)

            # De-indent content inside math environments
            if env_stack and env_stack[-1] in self.no_internal_indent:
                if not (begin_env_pattern.match(stripped_line) or end_env_pattern.match(stripped_line)):
                    current_indent_level = max(0, current_indent_level - 1)

            formatted_lines.append(self.indent_space * current_indent_level + stripped_line)

            # Check for \begin{} - increase indent after this line
            begin_match = begin_env_pattern.match(stripped_line)
            if begin_match:
                env_stack.append(begin_match.group(1))

        return formatted_lines

    def _apply_regex_spacing(self, latex_string: str) -> str:
        """Apply regex rules for consistent vertical and horizontal spacing."""
        
        # Ensure proper spacing around major environments
        environments = [
            'equation', 'align', 'gather', 'figure', 'table',
            'itemize', 'enumerate', 'verbatim', 'lstlisting',
            'theorem', 'lemma', 'proof', 'definition', 'example'
        ]
        
        for env in environments:
            # Add blank line before \begin{}
            latex_string = re.sub(
                r'\n*(\\begin\{' + env + r'[*]?\})',
                r'\n\n\1',
                latex_string
            )
            # Add blank line after \end{}
            latex_string = re.sub(
                r'(\\end\{' + env + r'[*]?\})\n*',
                r'\1\n\n',
                latex_string
            )

        # Ensure proper spacing around display math
        latex_string = re.sub(r'\n*(\\\[)', r'\n\n\1', latex_string)
        latex_string = re.sub(r'(\\\])\n*', r'\1\n\n', latex_string)

        # Add newline after sectioning commands
        latex_string = re.sub(
            r'(\\(?:chapter|section|subsection|subsubsection)\*?\{.*?\})\n*',
            r'\1\n',
            latex_string
        )

        # Add space around inline math if needed
        latex_string = re.sub(
            r'([a-zA-Z0-9\]\}])(\$.*?[^\\]\$)',
            r'\1 \2',
            latex_string
        )
        latex_string = re.sub(
            r'(\$.*?[^\\]\$)([a-zA-Z0-9\[\{])',
            r'\1 \2',
            latex_string
        )

        return latex_string


def remove_latex_wrapper(latex_content: str) -> str:
    """
    Strip the preamble and document wrappers, returning only the body content.
    Used when stitching multiple LaTeX responses together.
    """
    # Search for \begin{document} and take everything after it
    begin_doc_match = re.search(r"\\begin\{document\}", latex_content, re.DOTALL)
    if not begin_doc_match:
        return latex_content.strip()

    start_index = begin_doc_match.end()
    content_after_begin = latex_content[start_index:]

    # Search for \end{document} and take everything before it
    end_doc_match = re.search(r"\\end\{document\}", content_after_begin, re.DOTALL)
    if not end_doc_match:
        return content_after_begin.strip()

    end_index = end_doc_match.start()
    return content_after_begin[:end_index].strip()
