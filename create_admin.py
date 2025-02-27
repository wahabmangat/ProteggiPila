from model import session, User
from werkzeug.security import generate_password_hash

# Define admin credentials
admin_username = "admin"
admin_password = "Proteggipila123"
admin_role = "admin"

# Hash the password
hashed_password = generate_password_hash(admin_password)

# Create and add the admin user
admin_user = User(username=admin_username, password=hashed_password, role=admin_role)
session.add(admin_user)
session.commit()

print(f"Admin user '{admin_username}' created successfully!")
