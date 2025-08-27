#!/usr/bin/env python3
"""
WandB Quick Setup Script
Help install and configure WandB
"""

import subprocess
import sys
import os

def check_wandb_installation():
    """Check if wandb is installed"""
    try:
        import wandb
        print("✅ WandB is installed")
        return True
    except ImportError:
        print("❌ WandB is not installed")
        return False

def install_wandb():
    """Install wandb"""
    print("Installing WandB...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
        print("✅ WandB installation successful")
        return True
    except subprocess.CalledProcessError:
        print("❌ WandB installation failed")
        return False

def setup_wandb():
    """Setup wandb login"""
    print("\n🔧 Setting up WandB...")
    print("Please visit https://wandb.ai to create an account (if you don't have one)")
    print("Then run the following command to login:")
    print("\nwandb login")
    print("\nOr enter your API key here directly (you can find it at https://wandb.ai/settings):")
    
    try:
        # Try interactive login
        result = subprocess.run(["wandb", "login"], check=False)
        if result.returncode == 0:
            print("✅ WandB login successful")
            return True
        else:
            print("⚠️  Login may be incomplete, please check")
            return False
    except FileNotFoundError:
        print("⚠️  Cannot find wandb command, please restart terminal or reinstall")
        return False

def main():
    print("🎯 WandB Quick Setup")
    print("="*40)
    
    # Check installation
    if not check_wandb_installation():
        if not install_wandb():
            print("\n❌ Installation failed, please install manually:")
            print("pip install wandb")
            return
    
    # Setup login
    print("\n🔐 Configuring WandB login...")
    setup_wandb()
    
    print("\n✅ Setup completed!")
    print("\n🚀 Now you can run:")
    print("python upload_to_wandb.py")
    print("\nThis will upload your training results to WandB for visualization and analysis!")

if __name__ == "__main__":
    main() 