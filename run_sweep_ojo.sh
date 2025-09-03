#!/bin/bash

# Keras3 Edge Baseline - WandB Sweep Runner for Ojo (ARM64 + Jetson AGX Orin)
# This script runs hyperparameter sweeps using Docker containers

set -e

# Configuration
SWEEP_CONFIG="sweep_config.json"
NODE_NAME="ojo"

# Load env (WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT)
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi

# Resolve WandB settings
PROJECT="${WANDB_PROJECT:-keras3_edge_baseline}"
ENTITY="${WANDB_ENTITY:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Keras3 Edge Baseline - Ojo Sweep Runner${NC}"
echo -e "${YELLOW}Node: ${NODE_NAME} (ARM64 + Jetson AGX Orin)${NC}"

# Check if .env file exists
if [ -z "${WANDB_API_KEY:-}" ]; then
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
    local container_name="vit:ojo-${backend}"

    # Normalize backend label for Keras (torch vs pytorch)
    local keras_backend="$backend"
    if [ "$backend" = "pytorch" ] || [ "$backend" = "torch" ]; then
        keras_backend="torch"
    fi
    
    echo -e "${YELLOW}🏗️  Building ${backend} container...${NC}"
    docker build -f "Dockerfile.ojo.${backend}" -t "$container_name" .
    
    echo -e "${GREEN}🔍 Creating WandB sweep for ${backend} backend...${NC}"
    
    # Create sweep and capture sweep ID
    if [ -z "$ENTITY" ]; then
        echo -e "${RED}❌ Error: WANDB_ENTITY not set${NC}"
        echo "Set WANDB_ENTITY in .env (e.g., your username or team)."
        exit 1
    fi

    SWEEP_ID=$(docker run --rm -v $PWD:/workspace -e WANDB_API_KEY="$WANDB_API_KEY" \
        "$container_name" python3 -c "
import json
import wandb
import os

# Load sweep config
with open('/workspace/$SWEEP_CONFIG', 'r') as f:
    sweep_config = json.load(f)

# Set backend-specific configuration
sweep_config['parameters']['keras_backend'] = {'value': '$keras_backend'}
sweep_config['parameters']['node_name'] = {'value': '$NODE_NAME'}

# Set sweep name with backend and node suffix
sweep_config['name'] = f\"{sweep_config.get('name', 'keras3_edge_baseline')}_{'$NODE_NAME'}_{'$keras_backend'}\"

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
    docker run --rm -it --runtime nvidia --gpus all \
        -v $PWD:/workspace \
        -e WANDB_API_KEY="$WANDB_API_KEY" \
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
        echo -e "${YELLOW}🎯 Running sweeps for both backends sequentially${NC}"
        echo -e "${YELLOW}Starting with PyTorch backend (better Jetson support)...${NC}"
        run_sweep "pytorch"
        
        echo -e "${YELLOW}PyTorch backend completed. Starting JAX backend...${NC}"
        run_sweep "jax"
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [jax|pytorch|both|help]"
        echo ""
        echo "Options:"
        echo "  jax        Run sweep with JAX backend only"
        echo "  pytorch    Run sweep with PyTorch backend only"
        echo "  both       Run sweeps with both backends sequentially (default)"
        echo "  help       Show this help message"
        echo ""
        echo "Note: This script is optimized for Jetson AGX Orin with:"
        echo "  - Smaller batch sizes (16-128 vs 32-512)"
        echo "  - Reduced model sizes for memory efficiency"
        echo "  - Fewer epochs per run (15 vs 20)"
        echo ""
        echo "Examples:"
        echo "  $0              # Run both backends"
        echo "  $0 pytorch      # Run PyTorch only (recommended for Jetson)"
        echo "  $0 jax          # Run JAX only"
        exit 0
        ;;
    *)
        echo -e "${RED}❌ Unknown option: $1${NC}"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac

echo -e "${GREEN}🎉 Ojo sweep completed!${NC}"
