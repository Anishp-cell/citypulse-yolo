from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
import os
import warnings

# Secret key to encode the JWT
_DEFAULT_KEY = "your-super-secret-key-change-in-prod"
SECRET_KEY = os.getenv("SECRET_KEY", _DEFAULT_KEY)
if SECRET_KEY == _DEFAULT_KEY:
    warnings.warn(
        "⚠️  SECRET_KEY is set to the insecure default value. "
        "Set the SECRET_KEY environment variable before deploying to production.",
        RuntimeWarning,
        stacklevel=2,
    )
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
