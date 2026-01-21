"""
ScribeTeX Converter Module
Handles conversion of documents/images to LaTeX using various LLM providers.

Supported providers (January 2026):
- OpenAI: GPT-4.1 (non-reasoning), GPT-5.x (reasoning)
- Google: Gemini 2.5 Flash/Pro, Gemini 3 (preview)
- Anthropic: Claude 4.5 Sonnet/Opus/Haiku

Uses the latest SDK patterns for each provider.
"""
import os
import base64
import logging
import zipfile
from typing import List, Tuple, Optional
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image
from docx import Document as DocxDocument

from config import Config
from latex_formatter import LaTeXFormatter, remove_latex_wrapper
from compile_pdf import compile_tex_to_pdf, LatexCompilationError

logger = logging.getLogger(__name__)
latex_formatter = LaTeXFormatter()

# ============================================================================
# Model Configuration
# ============================================================================

# OpenAI reasoning models require different API parameters
OPENAI_REASONING_MODELS = {'gpt-5', 'gpt-5.1', 'gpt-5.2', 'o1', 'o3', 'o4-mini'}


def _is_openai_reasoning_model(model: str) -> bool:
    """Check if the OpenAI model is a reasoning model."""
    model_lower = model.lower()
    return any(model_lower.startswith(rm) for rm in OPENAI_REASONING_MODELS)


# ============================================================================
# LLM Prompt
# ============================================================================

def _get_llm_prompt() -> str:
    """Return the system prompt for LaTeX conversion."""
    return r"""You are an AI assistant specialised in converting images and text into complete, structured LaTeX documents. Your task is to analyse the provided content and generate a fully formatted LaTeX document. Follow these guidelines strictly:

1. Provide ONLY the LaTeX code as your response. Do not include any explanations, comments, or markdown formatting around the LaTeX code.

2. Begin the document with \documentclass and include all necessary packages based on the content (e.g., amsmath, amssymb, amsthm, hyperref, graphicx, tikz) in the preamble.

3. Structure the document with \begin{document} and \end{document}.

4. Process all content in the order provided, using appropriate sectioning commands to organise content.

5. Convert all content accurately, including text, mathematical equations, tables, and references.

6. For diagrams and figures:
   - Attempt to recreate simple diagrams using TikZ where possible.
   - For complex diagrams that cannot be accurately recreated, insert a placeholder:
     \begin{figure}[h]
     \centering
     \fbox{\parbox{0.8\textwidth}{\centering\vspace{2cm}[Diagram: brief description]\vspace{2cm}}}
     \caption{Description of the diagram}
     \end{figure}

7. Use appropriate LaTeX environments and commands for different types of content (e.g., equation, align, tabular, itemize, enumerate).

8. Maintain the logical flow and coherence of the original content.

9. Ensure the output is a complete, compilable LaTeX document.

10. The LaTeX code should start immediately at the beginning of your response and end at the very end, without any surrounding text."""


# ============================================================================
# File Processing Utilities
# ============================================================================

def get_file_type(filename: str) -> Optional[str]:
    """Determine the type of file based on its extension."""
    ext = Path(filename).suffix.lower()
    for file_type, extensions in Config.ALLOWED_EXTENSIONS.items():
        if ext in extensions:
            return file_type
    return None


def extract_images_from_pdf(pdf_path: str, output_dir: str, dpi: int = 200) -> List[str]:
    """Extract images from a PDF file."""
    os.makedirs(output_dir, exist_ok=True)
    image_paths = []
    
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_path = os.path.join(output_dir, f"page_{i + 1}.jpg")
        img.save(img_path, "JPEG", quality=85)
        image_paths.append(img_path)
    doc.close()
    
    logger.info(f"Extracted {len(image_paths)} pages from PDF")
    return image_paths


def read_text_file(file_path: str) -> str:
    """Read content from a text-based file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def read_docx_file(file_path: str) -> str:
    """Extract text content from a DOCX file."""
    doc = DocxDocument(file_path)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return '\n\n'.join(paragraphs)


def prepare_image_for_api(image_path: str) -> Tuple[str, str]:
    """Prepare an image for API submission. Returns (base64_data, media_type)."""
    ext = Path(image_path).suffix.lower()
    media_type_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
        '.tiff': 'image/tiff',
    }
    media_type = media_type_map.get(ext, 'image/jpeg')
    
    with open(image_path, 'rb') as f:
        base64_data = base64.b64encode(f.read()).decode('utf-8')
    
    return base64_data, media_type


# ============================================================================
# OpenAI API (January 2026)
# Models: gpt-4.1 (non-reasoning), gpt-5.x (reasoning)
# ============================================================================

def _call_openai_api(image_paths: List[str], api_key: str, text_content: str = "") -> str:
    """
    Send images to OpenAI API and get LaTeX response.
    
    Automatically handles both reasoning models (GPT-5.x, o-series) and 
    non-reasoning models (GPT-4.1) with correct parameters.
    """
    from openai import OpenAI
    
    client = OpenAI(api_key=api_key)
    model = Config.OPENAI_MODEL

    # Build content parts
    content_parts = [{
        "type": "text",
        "text": "Convert the following content into a complete LaTeX document."
    }]
    
    # Add text content if provided
    if text_content:
        content_parts.append({
            "type": "text",
            "text": f"Text content:\n{text_content}"
        })

    # Add images
    for img_path in image_paths:
        base64_image, _ = prepare_image_for_api(img_path)
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })

    messages = [
        {"role": "system", "content": _get_llm_prompt()},
        {"role": "user", "content": content_parts}
    ]

    logger.info(f"Sending request to OpenAI API (model: {model}, images: {len(image_paths)})")

    # Use different parameters based on model type
    if _is_openai_reasoning_model(model):
        # Reasoning models: use max_completion_tokens and disable reasoning for vision tasks
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=16384,
            reasoning_effort="none"  # Disable reasoning overhead for vision tasks
        )
    else:
        # Non-reasoning models: use max_tokens
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=16384
        )

    logger.info("Received response from OpenAI API")
    return response.choices[0].message.content


# ============================================================================
# Google Gemini API (January 2026)
# Uses the new unified google-genai SDK (not deprecated google-generativeai)
# Models: gemini-2.5-flash, gemini-2.5-pro, gemini-3-flash-preview
# ============================================================================

def _call_google_api(image_paths: List[str], api_key: str, text_content: str = "") -> str:
    """
    Send images to Google Gemini API and get LaTeX response.
    
    Uses the new unified google-genai SDK (January 2026).
    The old google-generativeai package was deprecated November 2025.
    """
    from google import genai
    from google.genai import types
    
    client = genai.Client(api_key=api_key)
    model = Config.GOOGLE_MODEL

    # Build content parts
    contents = [_get_llm_prompt()]
    
    # Add instruction
    contents.append("Convert the following content into a complete LaTeX document.")
    
    # Add text content if provided
    if text_content:
        contents.append(f"Text content:\n{text_content}")

    # Add images
    for img_path in image_paths:
        with open(img_path, 'rb') as f:
            img_bytes = f.read()
        _, media_type = prepare_image_for_api(img_path)
        contents.append(types.Part.from_bytes(data=img_bytes, mime_type=media_type))

    logger.info(f"Sending request to Google Gemini API (model: {model}, images: {len(image_paths)})")

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            max_output_tokens=16384,
            temperature=0.3
        )
    )

    logger.info("Received response from Google Gemini API")
    return response.text


# ============================================================================
# Anthropic Claude API (January 2026)
# Models: claude-sonnet-4-5, claude-opus-4-5, claude-haiku-4-5
# ============================================================================

def _call_anthropic_api(image_paths: List[str], api_key: str, text_content: str = "") -> str:
    """
    Send images to Anthropic Claude API and get LaTeX response.
    
    Uses the Anthropic Python SDK with vision support.
    All Claude 4.5 models support vision natively.
    """
    import anthropic
    
    client = anthropic.Anthropic(api_key=api_key)
    model = Config.ANTHROPIC_MODEL

    # Build content parts
    content_parts = []
    
    # Add images first (Claude prefers images before text questions)
    for img_path in image_paths:
        base64_data, media_type = prepare_image_for_api(img_path)
        content_parts.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64_data
            }
        })
    
    # Add instruction text
    instruction = "Convert the above images into a complete LaTeX document."
    if text_content:
        instruction = f"Text content:\n{text_content}\n\n{instruction}"
    
    content_parts.append({
        "type": "text",
        "text": instruction
    })

    logger.info(f"Sending request to Anthropic API (model: {model}, images: {len(image_paths)})")

    response = client.messages.create(
        model=model,
        max_tokens=16384,
        system=_get_llm_prompt(),
        messages=[
            {"role": "user", "content": content_parts}
        ]
    )

    logger.info("Received response from Anthropic API")
    
    # Extract text from response
    return response.content[0].text


# ============================================================================
# Main Conversion Logic
# ============================================================================

def call_llm_api(image_paths: List[str], provider: str, api_key: str, text_content: str = "") -> str:
    """Route to the appropriate LLM API based on provider."""
    if provider == 'google':
        return _call_google_api(image_paths, api_key, text_content)
    elif provider == 'anthropic':
        return _call_anthropic_api(image_paths, api_key, text_content)
    else:  # Default to OpenAI
        return _call_openai_api(image_paths, api_key, text_content)


def convert_to_latex(
    input_file_path: str,
    session_id: str,
    original_filename: str,
    provider: str,
    api_key: str
) -> dict:
    """
    Main conversion function. Handles all supported file types.
    
    Args:
        input_file_path: Path to the uploaded file
        session_id: User session ID for file isolation
        original_filename: Original name of the uploaded file
        provider: LLM provider ('openai', 'google', 'anthropic')
        api_key: API key for the provider
        
    Returns:
        dict with 'output_file_rel_path' and 'download_filename'
    """
    logger.info(f"Starting conversion: {original_filename} using {provider}")
    
    file_type = get_file_type(original_filename)
    base_output_dir = os.path.join(Config.OUTPUT_BASE_FOLDER, session_id)
    images_dir = os.path.join(base_output_dir, 'images_for_ai')
    os.makedirs(images_dir, exist_ok=True)
    
    image_paths = []
    text_content = ""
    
    # Process based on file type
    if file_type == 'pdf':
        image_paths = extract_images_from_pdf(input_file_path, images_dir)
    
    elif file_type == 'image':
        # Single image - copy to images directory
        import shutil
        dest_path = os.path.join(images_dir, 'image_1.jpg')
        
        # Convert to JPEG for consistency
        img = Image.open(input_file_path)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        img.save(dest_path, 'JPEG', quality=85)
        image_paths = [dest_path]
    
    elif file_type == 'text':
        text_content = read_text_file(input_file_path)
    
    elif file_type == 'document':
        ext = Path(original_filename).suffix.lower()
        if ext in ['.docx', '.doc']:
            text_content = read_docx_file(input_file_path)
        else:
            # For other document types, try reading as text
            text_content = read_text_file(input_file_path)
    
    else:
        raise ValueError(f"Unsupported file type: {original_filename}")
    
    # Chunk images for large documents (25 pages per chunk)
    chunk_size = 25
    raw_latex_responses = []
    
    if image_paths:
        image_chunks = [image_paths[i:i + chunk_size] for i in range(0, len(image_paths), chunk_size)]
        
        for i, chunk in enumerate(image_chunks):
            logger.info(f"Processing chunk {i + 1}/{len(image_chunks)} ({len(chunk)} images)")
            response = call_llm_api(chunk, provider, api_key, text_content if i == 0 else "")
            raw_latex_responses.append(response)
    else:
        # Text-only conversion
        response = call_llm_api([], provider, api_key, text_content)
        raw_latex_responses.append(response)
    
    # Stitch LaTeX documents together
    if len(raw_latex_responses) == 1:
        final_latex_content = raw_latex_responses[0]
    else:
        import re
        first_doc = raw_latex_responses[0]
        begin_doc_match = re.search(r"\\begin\{document\}", first_doc, re.DOTALL)
        
        if begin_doc_match:
            preamble = first_doc[:begin_doc_match.end()]
            first_content = remove_latex_wrapper(first_doc)
            final_latex_content = preamble + "\n" + first_content
            
            for next_doc in raw_latex_responses[1:]:
                next_content = remove_latex_wrapper(next_doc)
                final_latex_content += "\n" + next_content
            
            final_latex_content += "\n\\end{document}"
        else:
            final_latex_content = "\n".join(raw_latex_responses)
    
    # Format the LaTeX
    formatted_latex = latex_formatter.format_latex(final_latex_content)
    
    # Save .tex file
    base_filename = Path(original_filename).stem
    tex_filename = f"{base_filename}_converted.tex"
    tex_abs_path = os.path.join(base_output_dir, tex_filename)
    
    with open(tex_abs_path, "w", encoding="utf-8") as f:
        f.write(formatted_latex)
    
    logger.info(f"LaTeX file saved: {tex_filename}")
    
    # Optionally compile to PDF
    files_to_include = [tex_filename]
    
    if Config.COMPILE_PDF_ENABLED:
        logger.info("Attempting PDF compilation...")
        try:
            compiled_pdf_path = compile_tex_to_pdf(tex_abs_path, base_output_dir)
            pdf_filename = os.path.basename(compiled_pdf_path)
            files_to_include.append(pdf_filename)
            logger.info(f"PDF compiled successfully: {pdf_filename}")
        except LatexCompilationError as e:
            logger.warning(f"PDF compilation failed: {e}")
        except Exception as e:
            logger.warning(f"PDF compilation error: {e}")
    
    # Create ZIP archive
    zip_name = f"{base_filename}_scribetex_output.zip"
    zip_path = os.path.join(base_output_dir, zip_name)
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_to_add in files_to_include:
            abs_fp = os.path.join(base_output_dir, file_to_add)
            if os.path.exists(abs_fp):
                zf.write(abs_fp, file_to_add)
    
    logger.info(f"Conversion complete. Output: {zip_name}")
    
    return {
        "output_file_rel_path": zip_name,
        "download_filename": zip_name
    }
