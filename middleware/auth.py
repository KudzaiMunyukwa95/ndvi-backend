"""
🔒 Authentication Middleware for Flask API
"""

from functools import wraps
from flask import request, jsonify
import os
import logging

logger = logging.getLogger(__name__)


def require_auth(f):
    """Decorator to require authentication token for route access"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            logger.warning('Unauthorized access - Missing authorization header')
            return jsonify({
                'success': False,
                'error': 'Unauthorized',
                'message': 'Missing authorization header'
            }), 401
        
        token = auth_header.replace('Bearer ', '') if auth_header.startswith('Bearer ') else auth_header
        admin_token = os.environ.get('ADMIN_TOKEN')
        
        if not admin_token:
            logger.error('⚠️ CRITICAL: ADMIN_TOKEN not configured')
            return jsonify({
                'success': False,
                'error': 'Server configuration error'
            }), 500
        
        if token != admin_token:
            logger.warning(f'Invalid token from IP: {request.remote_addr}')
            return jsonify({
                'success': False,
                'error': 'Unauthorized',
                'message': 'Invalid authentication token'
            }), 401
        
        logger.info(f'Authenticated request to {request.path}')
        return f(*args, **kwargs)
    
    return decorated_function


def log_authentication_status():
    """Log authentication status on startup"""
    admin_token = os.environ.get('ADMIN_TOKEN')
    if admin_token:
        logger.info('🔒 Authentication: ENABLED')
    else:
        logger.error('⚠️ Authentication: DISABLED - ADMIN_TOKEN not set!')
