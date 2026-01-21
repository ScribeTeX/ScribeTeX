"""
ScribeTeX Routes
HTTP endpoints for file upload, conversion, and download.
"""
import os
import uuid
import shutil
import logging
from pathlib import Path

from flask import (
    Blueprint, render_template, request, send_file, redirect, url_for,
    session, jsonify, flash, current_app
)
import fitz

from config import Config
from converter import convert_to_latex, get_file_type

main_bp = Blueprint('main', __name__)
logger = logging.getLogger(__name__)


def _get_session_dir(base_folder_key: str, session_id: str) -> str:
    """Get the absolute path for a session-specific directory."""
    project_root = current_app.root_path
    base_folder = current_app.config[base_folder_key]
    session_dir = os.path.join(project_root, base_folder, session_id)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def _get_api_key(provider: str) -> str:
    """Get the API key for the specified provider."""
    key_map = {
        'openai': 'OPENAI_API_KEY',
        'google': 'GOOGLE_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY'
    }
    config_key = key_map.get(provider, 'OPENAI_API_KEY')
    return current_app.config.get(config_key)


def _get_page_count(file_path: str, file_type: str) -> int:
    """Get the number of pages/items in a file."""
    if file_type == 'pdf':
        with fitz.open(file_path) as doc:
            return len(doc)
    elif file_type == 'image':
        return 1
    else:
        # For text files, estimate based on content length
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            # Rough estimate: 1 page per 3000 characters
            return max(1, len(content) // 3000)
        except:
            return 1


@main_bp.route('/', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in the request.', 'error')
            return redirect(request.url)
        
        uploaded_file = request.files['file']
        
        if uploaded_file.filename == '':
            flash('No file selected. Please choose a file.', 'error')
            return redirect(request.url)
        
        # Check file extension
        file_type = get_file_type(uploaded_file.filename)
        if not file_type:
            allowed = ', '.join(Config.get_all_allowed_extensions())
            flash(f'Unsupported file type. Allowed: {allowed}', 'error')
            return redirect(request.url)
        
        # Create session if needed
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        sid = session['session_id']
        upload_dir = _get_session_dir('UPLOAD_BASE_FOLDER', sid)
        
        # Sanitise filename
        safe_name = ''.join(c for c in uploaded_file.filename if c.isalnum() or c in '._-')
        file_abs_path = os.path.join(upload_dir, safe_name)
        uploaded_file.save(file_abs_path)
        
        # Get page count
        page_count = _get_page_count(file_abs_path, file_type)
        
        # Store session data
        session.update({
            'file_abs_path': file_abs_path,
            'original_filename': uploaded_file.filename,
            'file_type': file_type,
            'page_count': page_count
        })
        
        return redirect(url_for('main.confirm_page'))
    
    # GET request - show upload form
    allowed_extensions = Config.get_all_allowed_extensions()
    return render_template('index.html', allowed_extensions=allowed_extensions)


@main_bp.route('/confirm')
def confirm_page():
    """Show confirmation page before conversion."""
    required_keys = ['original_filename', 'page_count', 'file_type']
    if not all(k in session for k in required_keys):
        flash("Session expired. Please upload your file again.", "error")
        return redirect(url_for('main.upload_file'))
    
    provider = Config.LLM_PROVIDER
    model_map = {
        'openai': Config.OPENAI_MODEL,
        'google': Config.GOOGLE_MODEL,
        'anthropic': Config.ANTHROPIC_MODEL
    }
    
    return render_template(
        'confirm.html',
        original_filename=session['original_filename'],
        page_count=session['page_count'],
        file_type=session['file_type'],
        provider=provider,
        model=model_map.get(provider, 'Unknown')
    )


@main_bp.route('/process', methods=['POST'])
def process_conversion():
    """Start the conversion process."""
    sid = session.get('session_id')
    if not sid or 'file_abs_path' not in session:
        flash("Session expired. Please start again.", 'error')
        return redirect(url_for('main.upload_file'))
    
    # Clear previous output
    output_dir = _get_session_dir('OUTPUT_BASE_FOLDER', sid)
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check API key
    provider = Config.LLM_PROVIDER
    api_key = _get_api_key(provider)
    
    if not api_key:
        flash(f'API key for {provider} is not configured. Check your .env file.', 'error')
        return redirect(url_for('main.confirm_page'))
    
    # Store provider info in session
    session['provider'] = provider
    session['api_key'] = api_key
    
    return redirect(url_for('main.processing_page'))


@main_bp.route('/processing')
def processing_page():
    """Show processing page with spinner."""
    if 'file_abs_path' not in session:
        flash("Session expired. Please start again.", 'error')
        return redirect(url_for('main.upload_file'))
    
    return render_template(
        'processing.html',
        filename=session.get('original_filename', 'Unknown')
    )


@main_bp.route('/api/convert', methods=['POST'])
def api_convert():
    """API endpoint to perform the actual conversion."""
    sid = session.get('session_id')
    if not sid:
        return jsonify({'success': False, 'error': 'Session expired'}), 400
    
    file_path = session.get('file_abs_path')
    original_filename = session.get('original_filename')
    provider = session.get('provider', Config.LLM_PROVIDER)
    api_key = session.get('api_key') or _get_api_key(provider)
    
    if not file_path or not os.path.exists(file_path):
        return jsonify({'success': False, 'error': 'File not found'}), 400
    
    if not api_key:
        return jsonify({'success': False, 'error': f'API key for {provider} not configured'}), 400
    
    try:
        result = convert_to_latex(
            input_file_path=file_path,
            session_id=sid,
            original_filename=original_filename,
            provider=provider,
            api_key=api_key
        )
        
        session['output_rel_path'] = result['output_file_rel_path']
        session['download_filename'] = result['download_filename']
        
        return jsonify({
            'success': True,
            'redirect_url': url_for('main.success_page')
        })
    
    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@main_bp.route('/success')
def success_page():
    """Show success page with download link."""
    if 'output_rel_path' not in session:
        flash("No conversion result found.", 'error')
        return redirect(url_for('main.upload_file'))
    
    return render_template(
        'success.html',
        filename=session.get('download_filename', 'output.zip')
    )


@main_bp.route('/download')
def download_file():
    """Download the converted file."""
    sid = session.get('session_id')
    output_rel_path = session.get('output_rel_path')
    download_filename = session.get('download_filename')
    
    if not sid or not output_rel_path:
        flash("Download not available.", 'error')
        return redirect(url_for('main.upload_file'))
    
    output_dir = _get_session_dir('OUTPUT_BASE_FOLDER', sid)
    file_path = os.path.join(output_dir, output_rel_path)
    
    if not os.path.exists(file_path):
        flash("File not found.", 'error')
        return redirect(url_for('main.upload_file'))
    
    return send_file(
        file_path,
        as_attachment=True,
        download_name=download_filename
    )
