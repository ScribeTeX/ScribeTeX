# ScribeTeX

Convert handwritten notes, PDFs, images, and documents to LaTeX.

ScribeTeX uses AI vision models to read your content and produce clean, compilable LaTeX code. It supports multiple LLM providers and file formats.

## Features

- **Multiple file formats**: PDF, images (PNG, JPG, GIF, WebP), text files (TXT, Markdown), documents (DOCX)
- **Three LLM providers**: OpenAI, Google Gemini, Anthropic Claude
- **Automatic PDF compilation**: Optionally compile the output to PDF (requires pdflatex)
- **Session isolation**: Each user's files are kept separate
- **Chunked processing**: Large documents are split into manageable chunks

## Supported Models (January 2026)

| Provider | Non-Reasoning | Reasoning |
|----------|---------------|-----------|
| OpenAI | gpt-4.1, gpt-4.1-mini | gpt-5, gpt-5.1, gpt-5.2 |
| Google | gemini-2.5-flash, gemini-2.5-pro | gemini-3-flash-preview |
| Anthropic | claude-sonnet-4-5, claude-haiku-4-5 | claude-opus-4-5 (extended thinking) |

The application automatically handles the different API parameters required for reasoning vs non-reasoning models.

## Requirements

- Python 3.10 or higher
- An API key from at least one provider (OpenAI, Google, or Anthropic)
- pdflatex (optional, for PDF compilation)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/scribetex.git
cd scribetex
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy the example environment file and add your API key:
```bash
cp .env.example .env
```

5. Edit `.env` and set your configuration:
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
```

6. Run the application:
```bash
python app.py
```

7. Open http://localhost:8000 in your browser.

## Configuration

All configuration is done through environment variables. See `.env.example` for the full list.

### Provider Selection

Set `LLM_PROVIDER` to one of: `openai`, `google`, or `anthropic`.

### Model Selection

Each provider has a default model, but you can override it:

```bash
# OpenAI
OPENAI_MODEL=gpt-4.1

# Google (uses the new google-genai SDK)
GOOGLE_MODEL=gemini-2.5-flash

# Anthropic
ANTHROPIC_MODEL=claude-sonnet-4-5
```

### PDF Compilation

If you have a LaTeX distribution installed (TeX Live, MiKTeX), the application will compile the output to PDF. To disable this:

```bash
COMPILE_PDF_ENABLED=False
```

## Usage

1. Go to the upload page
2. Drag and drop your file or click to browse
3. Review the details and click "Start Conversion"
4. Wait for the conversion to complete
5. Download the ZIP file containing your `.tex` file (and `.pdf` if compilation succeeded)

## File Format Notes

- **PDF**: Each page is converted to an image and sent to the AI
- **Images**: Sent directly to the AI
- **Text files**: Content is extracted and sent as text
- **DOCX**: Text is extracted (formatting may be lost)

## Troubleshooting

### Google API: "google-generativeai" errors

The old `google-generativeai` package was deprecated in November 2025. This project uses the new `google-genai` SDK. If you have the old package installed:

```bash
pip uninstall google-generativeai
pip install google-genai
```

### OpenAI: "max_tokens is not supported"

This error occurs with reasoning models (GPT-5.x). The application handles this automatically, but ensure you're using the latest version of the code.

### PDF compilation fails

- Check that `pdflatex` is installed and in your PATH
- Try running `pdflatex --version` in your terminal
- Set `COMPILE_PDF_ENABLED=False` to disable PDF compilation

## Project Structure

```
scribetex/
├── app.py              # Flask application
├── routes.py           # HTTP endpoints
├── converter.py        # LLM API calls and conversion logic
├── latex_formatter.py  # LaTeX formatting and cleanup
├── compile_pdf.py      # pdflatex wrapper
├── config.py           # Configuration
├── logging_config.py   # Logging setup
├── requirements.txt    # Dependencies
├── templates/          # HTML templates
├── static/             # CSS and assets
├── .env.example        # Example environment file
├── LICENSE             # Apache 2.0 licence
└── NOTICE              # Attribution notice
```

## API Costs

Costs vary by provider and model. Rough estimates per page:

| Provider | Model | Approximate Cost |
|----------|-------|------------------|
| OpenAI | gpt-4.1 | $0.01-0.03 |
| Google | gemini-2.5-flash | $0.001-0.005 |
| Anthropic | claude-sonnet-4-5 | $0.01-0.02 |

Check each provider's pricing page for current rates.

## Licence

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

**Attribution required**: If you use this software, please credit the original authors in your documentation or application.

## Authors

- [Manav Madan Rawal](https://github.com/Manav02012002)
- [Shamant Shreedhar Dixit](https://github.com/shamantdixit)

## Contributing

Contributions are welcome. Please open an issue to discuss your proposed changes before submitting a pull request.
