"""
🔒 Authentication Middleware for FastAPI
"""

from fastapi import Header, HTTPException, status
import os
import logging

logger = logging.getLogger(__name__)

async def verify_auth(authorization: str = Header(None)):
    """FastAPI dependency to require authentication token"""
    if not authorization:
        logger.warning('Unauthorized access - Missing authorization header')
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header"
        )
    
    token = authorization.replace('Bearer ', '').strip() if authorization.startswith('Bearer ') else authorization.strip()
    # Also strip any potential quotes if they were added to the environment variable
    token = token.strip('"').strip("'")
    
    admin_token = os.environ.get('ADMIN_TOKEN')
    if admin_token:
        admin_token = admin_token.strip().strip('"').strip("'")
    
    if not admin_token:
        logger.error('⚠️ CRITICAL: ADMIN_TOKEN not configured')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error"
        )
    
    if token != admin_token:
        logger.warning(f'Invalid token provided. Received: {token[:5]}..., Expected: {admin_token[:5]}...')
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    
    return True

def log_authentication_status():
    """Log authentication status on startup"""
    admin_token = os.environ.get('ADMIN_TOKEN')
    if admin_token:
        logger.info('🔒 Authentication: ENABLED')
    else:
        logger.error('⚠️ Authentication: DISABLED - ADMIN_TOKEN not set!')

