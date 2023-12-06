#!/bin/bash

# Check if openfabric-pysdk is installed
if ! pip show openfabric-pysdk >/dev/null 2>&1; then
    echo "Installing openfabric-pysdk..."
    pip install openfabric-pysdk==0.1.11
else
    echo "openfabric-pysdk is already installed."
fi

# Check if other requirements are installed
if ! pip show -r requirements.txt >/dev/null 2>&1; then
    echo "Installing requirements..."
    pip install -r requirements.txt
else
    echo "Requirements are already installed."
fi

# Run the application
python ./ignite.py
