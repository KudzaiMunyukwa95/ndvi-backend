"""
🔒 Authentication Middleware for FastAPI
Converted from Flask to maintain exact same authentication logic
"""

import os
import logging
from functools import wraps
from typing import Callable
from fastapi import Request, HTTPException, status

logger = logging.getLogger(__name__)


def log_authentication_status():
    """Log authentication status on startup"""
    admin_token = os.environ.get('ADMIN_TOKEN')
    if admin_token:
        logger.info('🔒 Authentication: ENABLED')
    else:
        logger.error('⚠️ Authentication: DISABLED - ADMIN_TOKEN not set!')


def validate_token(token: str) -> bool:
    """Validate the provided authentication token against ADMIN_TOKEN"""
    admin_token = os.environ.get('ADMIN_TOKEN')
    
    if not admin_token:
        logger.error('⚠️ CRITICAL: ADMIN_TOKEN not configured')
        return False
    
    return token == admin_token


def require_auth(func: Callable):
    """
    Decorator to require authentication token for route access
    Maintains exact same behavior as Flask version
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract request from args (FastAPI passes it as first argument)
        request = None
        for arg in args:
            if isinstance(arg, Request):
                request = arg
                break
        
        # Also check kwargs
        if not request:
            request = kwargs.get('request')
        
        if not request:
            logger.error("Request object not found in decorator arguments")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    'success': False,
                    'error': 'Server configuration error'
                }
            )
        
        # Extract Authorization header
        auth_header = request.headers.get("Authorization")
        
        if not auth_header:
            logger.warning('Unauthorized access - Missing authorization header')
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    'success': False,
                    'error': 'Unauthorized',
                    'message': 'Missing authorization header'
                }
            )
        
        # Extract token from "Bearer <token>" format, or use as-is
        token = auth_header.replace('Bearer ', '') if auth_header.startswith('Bearer ') else auth_header
        
        # Check if ADMIN_TOKEN is configured
        admin_token = os.environ.get('ADMIN_TOKEN')
        
        if not admin_token:
            logger.error('⚠️ CRITICAL: ADMIN_TOKEN not configured')
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    'success': False,
                    'error': 'Server configuration error'
                }
            )
        
        # Validate the token
        if token != admin_token:
            client_ip = request.client.host if request.client else 'unknown'
            logger.warning(f'Invalid token from IP: {client_ip}')
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    'success': False,
                    'error': 'Unauthorized',
                    'message': 'Invalid authentication token'
                }
            )
        
        logger.info(f'Authenticated request to {request.url.path}')
        
        # Call the actual endpoint function
        return await func(*args, **kwargs)
    
    return wrapper
