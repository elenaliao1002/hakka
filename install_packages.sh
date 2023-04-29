#!/bin/bash

echo "Installing apt dependencies..."
sudo apt-get update
sudo apt-get install -y $(cat packages.txt)
echo "Apt dependencies installed."
