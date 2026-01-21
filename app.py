"""
ScribeTeX - Convert handwritten notes and documents to LaTeX
Main Flask application factory.
"""
import logging
import uuid
from flask import Flask, render_template, g
from flask_session import Session
import datetime

from config import Config
from logging_config import setup_logging

setup_logging()


def create_app(config_class=Config):
    """Create and configure the Flask application."""
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_class)
    Session(app)

    @app.before_request
    def before_request_log():
        g.request_id = str(uuid.uuid4())

    from routes import main_bp
    app.register_blueprint(main_bp)

    register_error_handlers(app)

    @app.context_processor
    def inject_current_year():
        return {'current_year': datetime.datetime.now().year}

    app.logger.info("ScribeTeX application started successfully.")
    return app


def register_error_handlers(app):
    """Register error handlers for common HTTP errors."""
    @app.errorhandler(404)
    def page_not_found_error(error):
        return render_template('errors/404.html'), 404

    @app.errorhandler(413)
    def request_entity_too_large_error(error):
        max_size_mb = app.config.get('MAX_CONTENT_LENGTH', 0) / (1024 * 1024)
        return render_template('errors/413.html', max_size_mb=max_size_mb), 413

    @app.errorhandler(500)
    def internal_server_error(error):
        return render_template('errors/500.html'), 500


if __name__ == '__main__':
    flask_app = create_app()
    flask_app.run(host='0.0.0.0', port=8000, debug=flask_app.config['DEBUG'])
