# Download latest version of Python
wget https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz

# Extract Python
tar -xzf Python-3.12.0.tgz

# Download eigen3
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz

# Extract eigen3
tar -xzf eigen-3.4.0.tar.gz
mv eigen-3.4.0 eigen3

# Set OS_NAME environment variable
export OS_NAME="Wisteria"
