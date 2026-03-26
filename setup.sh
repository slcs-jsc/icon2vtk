#!/bin/bash

set -e  # stop on error

echo -e "\n🔧 Checking environment..."

# Check if 'module' command exists...
if command -v module &> /dev/null; then
    echo -e "\n📦 Module system detected → loading Python environment"
    module purge
    module load GCC Python
    MODULE_LOADED=true
else
    echo -e "\n⚠️  No module system found → assuming Python is available"
    MODULE_LOADED=false
fi

ENV_NAME=".venv"
echo -e "\n📁 Creating virtual environment: $ENV_NAME"

if [ ! -d "$ENV_NAME" ]; then
    python3 -m venv $ENV_NAME
else
    echo -e "\n⚠️  Environment already exists, reusing it"
fi

echo -e "\n🚀 Activating environment..."
source $ENV_NAME/bin/activate

echo -e "\n⬆️ Upgrading pip..."
pip install --upgrade pip

echo -e "\n📦 Installing requirements..."
pip install -r requirements.txt

echo -e "\n✅ Setup complete!"
echo -e "\n👉 To activate later, run:"

if [ "$MODULE_LOADED" = true ]; then
    echo "   module purge ; module load GCC Python"
fi

echo "   source $ENV_NAME/bin/activate"
