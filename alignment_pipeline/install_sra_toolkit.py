#!/usr/bin/env python3
"""
Quick SRA Toolkit installer script.
Convenience script around the full installer in utils/.
"""

import sys
import os

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

try:
    from sra_toolkit_installer import SRAToolkitInstaller
    
    print("🧬 RNA-seq Pipeline - SRA Toolkit Installer")
    print("=" * 50)
    
    installer = SRAToolkitInstaller()
    
    # Check current status
    status = installer.check_installation()
    
    if status['installed'] and status['functional']:
        print("✅ SRA Toolkit is already installed and functional!")
        installer.print_status()
    else:
        print("📥 Installing SRA Toolkit...")
        success = installer.download_and_install()
        
        if success:
            print("🔧 Configuring environment...")
            installer.configure_environment()
            print("\n✅ SRA Toolkit installation completed!")
            installer.print_status()
        else:
            print("❌ Installation failed!")
            sys.exit(1)

except Exception as e:
    print(f"❌ Error: {e}")
    print("Try running: python utils/sra_toolkit_installer.py --install")
    sys.exit(1)
