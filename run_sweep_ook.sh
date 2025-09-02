#!/bin/bash

# Keras3 Edge Baseline - WandB Sweep Runner for Ook (x86_64 + RTX 4050)
# This script runs hyperparameter sweeps using Docker containers

set -e

# Configuration
SWEEP_CONFIG="sweep_config.json"
PROJECT="keras3_edge_baseline"
ENTITY="hug"
NODE_NAME="ook"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Keras3 Edge Baseline - Ook Sweep Runner${NC}"
echo -e "${YELLOW}Node: ${NODE_NAME} (x86_64 + RTX 4050)${NC}"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${RED}❌ Error: .env file not found${NC}"
    echo "Please create a .env file with your WANDB_API_KEY:"
    echo "echo 'WANDB_API_KEY=your_key_here' > .env"
    exit 1
fi

# Check if sweep config exists
if [ ! -f "$SWEEP_CONFIG" ]; then
    echo -e "${RED}❌ Error: Sweep configuration file '$SWEEP_CONFIG' not found${NC}"
    exit 1
fi

# Function to run sweep with specific backend
run_sweep() {
    local backend=$1
    local container_name="vit:ook-${backend}"
    
    echo -e "${YELLOW}🏗️  Building ${backend} container...${NC}"
    docker build -f "Dockerfile.ook.${backend}" -t "$container_name" .
    
    echo -e "${GREEN}🔍 Creating WandB sweep for ${backend} backend...${NC}"
    
    # Create sweep and capture sweep ID
    SWEEP_ID=$(docker run --rm -v $PWD:/app -e WANDB_API_KEY="$(grep WANDB_API_KEY .env | cut -d '=' -f2)" \
        "$container_name" python -c "
import json
import wandb
import os

# Load sweep config
with open('/app/$SWEEP_CONFIG', 'r') as f:
    sweep_config = json.load(f)

# Set backend-specific configuration
sweep_config['parameters']['keras_backend'] = {'value': '$backend'}
sweep_config['parameters']['node_name'] = {'value': '$NODE_NAME'}

# Adjust batch size based on backend (PyTorch uses more memory)
if '$backend' == 'torch':
    sweep_config['parameters']['batch_size'] = {'values': [32, 64, 128]}
else:
    sweep_config['parameters']['batch_size'] = {'values': [64, 128, 256, 512]}

# Initialize wandb and create sweep
wandb.login()
sweep_id = wandb.sweep(sweep_config, project='$PROJECT', entity='$ENTITY')
print(f'SWEEP_ID:{sweep_id}')
")
    
    # Extract sweep ID from output
    SWEEP_ID=$(echo "$SWEEP_ID" | grep "SWEEP_ID:" | cut -d':' -f2)
    
    if [ -z "$SWEEP_ID" ]; then
        echo -e "${RED}❌ Failed to create sweep for ${backend}${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✅ Created sweep: ${SWEEP_ID}${NC}"
    echo -e "${YELLOW}🏃 Starting sweep agent for ${backend}...${NC}"
    echo -e "${YELLOW}📊 View sweep at: https://wandb.ai/${ENTITY}/${PROJECT}/sweeps/${SWEEP_ID}${NC}"
    
    # Start sweep agent
    docker run --rm --gpus all \
        -v $PWD:/app \
        -e WANDB_API_KEY="$(grep WANDB_API_KEY .env | cut -d '=' -f2)" \
        "$container_name" \
        wandb agent "${ENTITY}/${PROJECT}/${SWEEP_ID}"
}

# Main execution
case "${1:-both}" in
    "jax")
        echo -e "${YELLOW}🎯 Running JAX backend sweep only${NC}"
        run_sweep "jax"
        ;;
    "pytorch"|"torch")
        echo -e "${YELLOW}🎯 Running PyTorch backend sweep only${NC}"
        run_sweep "pytorch"
        ;;
    "both"|"")
        echo -e "${YELLOW}🎯 Running sweeps for both backends${NC}"
        echo -e "${YELLOW}Starting with JAX backend...${NC}"
        run_sweep "jax" &
        JAX_PID=$!
        
        sleep 10  # Stagger the starts
        
        echo -e "${YELLOW}Starting with PyTorch backend...${NC}"
        run_sweep "pytorch" &
        PYTORCH_PID=$!
        
        echo -e "${GREEN}🏃 Both sweep agents started in parallel${NC}"
        echo -e "${YELLOW}JAX PID: $JAX_PID${NC}"
        echo -e "${YELLOW}PyTorch PID: $PYTORCH_PID${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop all agents${NC}"
        
        # Wait for both processes
        wait $JAX_PID $PYTORCH_PID
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [jax|pytorch|both|help]"
        echo ""
        echo "Options:"
        echo "  jax        Run sweep with JAX backend only"
        echo "  pytorch    Run sweep with PyTorch backend only"
        echo "  both       Run sweeps with both backends in parallel (default)"
        echo "  help       Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0              # Run both backends"
        echo "  $0 jax          # Run JAX only"
        echo "  $0 pytorch      # Run PyTorch only"
        exit 0
        ;;
    *)
        echo -e "${RED}❌ Unknown option: $1${NC}"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac

echo -e "${GREEN}🎉 Ook sweep completed!${NC}"