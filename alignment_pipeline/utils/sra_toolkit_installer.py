#!/usr/bin/env python3
"""
SRA Toolkit Installation Manager for Hoffman2

Automatically downloads and installs the correct version of SRA Toolkit
for the current platform, with special handling for Hoffman2 cluster compatibility.
"""

import os
import sys
import logging
import platform
import subprocess
import urllib.request
import tarfile
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SRAToolkitInstaller:
    """Manages SRA Toolkit installation for different platforms."""
    
    def __init__(self, install_dir=None):
        """
        Initialize the SRA Toolkit installer.
        
        Args:
            install_dir: Directory to install SRA Toolkit (auto-detected if None)
        """
        if install_dir is None:
            install_dir = self._detect_install_dir()
        
        self.install_dir = Path(install_dir)
        self.install_dir.mkdir(parents=True, exist_ok=True)
        
        # Platform detection
        self.platform_info = self._detect_platform()
        logger.info(f"Detected platform: {self.platform_info['os']} {self.platform_info['arch']}")
        
        # SRA Toolkit version and URLs
        self.sra_version = "2.10.8"  # Known stable version for Hoffman2
        self.download_urls = {
            'linux_x64': f'https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/{self.sra_version}/sratoolkit.{self.sra_version}-ubuntu64.tar.gz',
            'linux_arm64': f'https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/{self.sra_version}/sratoolkit.{self.sra_version}-ubuntu64.tar.gz',  # Use x64 version
            'mac_x64': f'https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/{self.sra_version}/sratoolkit.{self.sra_version}-mac64.tar.gz',
            'mac_arm64': f'https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/{self.sra_version}/sratoolkit.{self.sra_version}-mac64.tar.gz',
            'windows_x64': f'https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/{self.sra_version}/sratoolkit.{self.sra_version}-win64.zip'
        }
        
        # Expected installation paths
        platform_key = f"{self.platform_info['os']}_{self.platform_info['arch']}"
        self.toolkit_dirname = {
            'linux_x64': f'sratoolkit.{self.sra_version}-ubuntu64',
            'linux_arm64': f'sratoolkit.{self.sra_version}-ubuntu64',
            'mac_x64': f'sratoolkit.{self.sra_version}-mac64',
            'mac_arm64': f'sratoolkit.{self.sra_version}-mac64',
            'windows_x64': f'sratoolkit.{self.sra_version}-win64'
        }.get(platform_key, f'sratoolkit.{self.sra_version}-ubuntu64')
        
        self.toolkit_path = self.install_dir / self.toolkit_dirname
        self.bin_path = self.toolkit_path / 'bin'
        
        # Key binaries to check
        binary_ext = '.exe' if self.platform_info['os'] == 'windows' else ''
        self.key_binaries = [
            f'fasterq-dump{binary_ext}',
            f'fastq-dump{binary_ext}',
            f'vdb-config{binary_ext}',
            f'prefetch{binary_ext}'
        ]
    
    def _detect_platform(self):
        """Detect the current platform and architecture."""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        # Normalize OS
        if system == 'darwin':
            os_name = 'mac'
        elif system == 'linux':
            os_name = 'linux'
        elif system == 'windows':
            os_name = 'windows'
        else:
            logger.warning(f"Unknown OS: {system}, defaulting to linux")
            os_name = 'linux'
        
        # Normalize architecture
        if machine in ['x86_64', 'amd64']:
            arch = 'x64'
        elif machine in ['arm64', 'aarch64']:
            arch = 'arm64'
        else:
            logger.warning(f"Unknown architecture: {machine}, defaulting to x64")
            arch = 'x64'
        
        return {'os': os_name, 'arch': arch}
    
    def _detect_install_dir(self):
        """Auto-detect appropriate installation directory."""
        # Try user home first
        home_dir = Path.home()
        
        # Common installation directories
        candidates = [
            home_dir,  # User home directory
            home_dir / 'tools',
            home_dir / 'software',
            Path('/usr/local'),  # System-wide (if writable)
            Path('/opt'),        # Alternative system-wide
            Path.cwd()           # Current directory as fallback
        ]
        
        for candidate in candidates:
            try:
                # Test if we can write to this directory
                test_file = candidate / '.write_test'
                test_file.touch()
                test_file.unlink()
                return str(candidate)
            except (PermissionError, OSError):
                continue
        
        # Fallback to current directory
        return str(Path.cwd())
    
    def check_installation(self):
        """Check if SRA Toolkit is already installed and functional."""
        if not self.toolkit_path.exists():
            return {
                'installed': False,
                'functional': False,
                'version': None,
                'path': None,
                'missing_binaries': self.key_binaries
            }
        
        # Check if binaries exist
        missing_binaries = []
        existing_binaries = []
        
        for binary in self.key_binaries:
            binary_path = self.bin_path / binary
            if binary_path.exists():
                existing_binaries.append(binary)
            else:
                missing_binaries.append(binary)
        
        # Test fasterq-dump functionality if it exists
        functional = False
        version = None
        
        fasterq_path = self.bin_path / self.key_binaries[0]  # fasterq-dump
        if fasterq_path.exists():
            try:
                result = subprocess.run(
                    [str(fasterq_path), '--version'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    functional = True
                    # Extract version from output
                    for line in result.stdout.split('\n'):
                        if 'fasterq-dump' in line and 'version' in line:
                            version = line.strip()
                            break
            except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
                pass
        
        return {
            'installed': len(missing_binaries) == 0,
            'functional': functional,
            'version': version,
            'path': str(self.toolkit_path),
            'bin_path': str(self.bin_path),
            'missing_binaries': missing_binaries,
            'existing_binaries': existing_binaries
        }
    
    def download_and_install(self, force=False):
        """Download and install SRA Toolkit."""
        logger.info(f"Installing SRA Toolkit {self.sra_version}...")
        
        # Check if already installed
        status = self.check_installation()
        if status['installed'] and status['functional'] and not force:
            logger.info("✅ SRA Toolkit already installed and functional")
            return True
        
        # Determine download URL
        platform_key = f"{self.platform_info['os']}_{self.platform_info['arch']}"
        download_url = self.download_urls.get(platform_key)
        
        if not download_url:
            logger.error(f"No download URL available for platform: {platform_key}")
            return False
        
        # Create temporary download directory
        temp_dir = self.install_dir / 'temp_sra_download'
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Download
            filename = download_url.split('/')[-1]
            download_path = temp_dir / filename
            
            logger.info(f"Downloading from: {download_url}")
            logger.info(f"Saving to: {download_path}")
            
            urllib.request.urlretrieve(download_url, download_path)
            logger.info("✅ Download completed")
            
            # Extract
            logger.info("Extracting SRA Toolkit...")
            
            if filename.endswith('.tar.gz'):
                with tarfile.open(download_path, 'r:gz') as tar:
                    tar.extractall(self.install_dir)
            elif filename.endswith('.zip'):
                import zipfile
                with zipfile.ZipFile(download_path, 'r') as zip_ref:
                    zip_ref.extractall(self.install_dir)
            else:
                logger.error(f"Unsupported archive format: {filename}")
                return False
            
            logger.info("✅ Extraction completed")
            
            # Verify installation
            status = self.check_installation()
            if status['installed'] and status['functional']:
                logger.info("✅ SRA Toolkit installation verified")
                logger.info(f"   Installed at: {status['path']}")
                logger.info(f"   Version: {status['version']}")
                return True
            else:
                logger.error("❌ Installation verification failed")
                logger.error(f"   Missing binaries: {status['missing_binaries']}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to download/install SRA Toolkit: {e}")
            return False
        
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Could not clean up temp directory: {e}")
    
    def configure_environment(self):
        """Configure environment variables for SRA Toolkit."""
        logger.info("Configuring SRA Toolkit environment...")
        
        status = self.check_installation()
        if not status['installed']:
            logger.error("Cannot configure - SRA Toolkit not installed")
            return False
        
        # Add to PATH
        bin_path = str(self.bin_path)
        current_path = os.environ.get('PATH', '')
        
        if bin_path not in current_path:
            os.environ['PATH'] = f"{bin_path}{os.pathsep}{current_path}"
            logger.info(f"✅ Added SRA Toolkit to PATH: {bin_path}")
        else:
            logger.info("✅ SRA Toolkit already in PATH")
        
        # Verify binaries are accessible
        for binary in ['fasterq-dump', 'vdb-config']:
            try:
                result = subprocess.run([binary, '--version'], 
                                      capture_output=True, timeout=5)
                if result.returncode == 0:
                    logger.info(f"✅ {binary} available: {shutil.which(binary)}")
                else:
                    logger.warning(f"⚠️  {binary} not working properly")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning(f"⚠️  {binary} not found in PATH")
        
        return True
    
    def print_status(self):
        """Print concise installation status."""
        status = self.check_installation()
        
        print("\nSRA Toolkit Status:")
        if status['installed'] and status['functional']:
            print("✅ INSTALLED AND FUNCTIONAL")
            if status['version']:
                print(f"   Version: {status['version']}")
        elif status['installed']:
            print("⚠️  INSTALLED BUT NOT FUNCTIONAL")
        else:
            print("❌ NOT INSTALLED")
            print(f"   Install with: python {__file__} --install")
        
        if status['missing_binaries']:
            print(f"   Missing: {', '.join(status['missing_binaries'])}")
        
        print(f"   Path: {self.bin_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Install and manage SRA Toolkit')
    parser.add_argument('--install', action='store_true', help='Install SRA Toolkit')
    parser.add_argument('--status', action='store_true', help='Show installation status')
    parser.add_argument('--configure', action='store_true', help='Configure environment variables')
    parser.add_argument('--force', action='store_true', help='Force reinstall even if already installed')
    parser.add_argument('--install-dir', help='Override installation directory')
    
    args = parser.parse_args()
    
    # Initialize installer
    try:
        installer = SRAToolkitInstaller(install_dir=args.install_dir)
    except Exception as e:
        logger.error(f"Failed to initialize SRA Toolkit installer: {e}")
        sys.exit(1)
    
    # Execute requested action
    if args.install:
        success = installer.download_and_install(force=args.force)
        if success:
            installer.configure_environment()
        sys.exit(0 if success else 1)
    
    elif args.configure:
        success = installer.configure_environment()
        sys.exit(0 if success else 1)
    
    elif args.status:
        installer.print_status()
        sys.exit(0)
    
    else:
        # Default to status if no action specified
        installer.print_status()
        sys.exit(0)

if __name__ == "__main__":
    main()
