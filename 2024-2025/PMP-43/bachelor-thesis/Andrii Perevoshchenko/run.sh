#!/bin/bash

if [[ "$(docker images -q advection-diffusion 2> /dev/null)" == "" ]]; then
    echo "Building Docker image..."
    docker build -t advection-diffusion .
fi

echo "Running example with no stabilization..."
docker run --rm -v $(pwd):/app advection-diffusion conda run -n fenics python advection_diffusion.py --example example3 --stabilization none

echo "Running example with artificial diffusion..."
docker run --rm -v $(pwd):/app advection-diffusion conda run -n fenics python advection_diffusion.py --example example3 --stabilization artificial --alpha 1.0

echo "Running example with streamline diffusion..."
docker run --rm -v $(pwd):/app advection-diffusion conda run -n fenics python advection_diffusion.py --example example3 --stabilization streamline --delta 0.5

echo "Running example with bubble functions..."
docker run --rm -v $(pwd):/app advection-diffusion conda run -n fenics python advection_diffusion.py --example example3 --stabilization bubble 