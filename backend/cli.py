"""Admin CLI.

    python -m backend.cli create-admin --email you@example.com
"""

import argparse
import getpass

from backend.db import make_engine, make_sessionmaker
from backend.models import User
from backend.security import hash_password
from backend.settings import Settings


def create_admin(settings: Settings, email: str, password: str) -> None:
    email = email.strip().lower()
    if len(password) < 8:
        raise SystemExit("password must be at least 8 characters")
    engine = make_engine(settings.database_url)
    with make_sessionmaker(engine)() as db:
        user = db.query(User).filter(User.email == email).first()
        if user is None:
            user = User(email=email, full_name="Administrator")
            db.add(user)
        user.hashed_password = hash_password(password)
        user.role = "admin"
        user.is_active = True
        db.commit()
    engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(prog="backend.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("create-admin", help="create or promote an admin account")
    p.add_argument("--email", required=True)
    p.add_argument("--password", help="omit to be prompted")
    args = parser.parse_args()
    if args.cmd == "create-admin":
        password = args.password or getpass.getpass("Password: ")
        create_admin(Settings(), args.email, password)
        print(f"admin ready: {args.email.lower()}")


if __name__ == "__main__":
    main()
