#!/usr/bin/env python3
"""
Launcher script for the Retirement Spending Calculator Dashboard
Run this script to start the spending-focused retirement calculator
"""

import subprocess
import sys
import os

def main():
    """Launch the retirement spending dashboard"""
    dashboard_path = os.path.join(os.path.dirname(__file__), 'retirement_spending_dashboard.py')

    print("Starting Retirement Spending Calculator Dashboard...")
    print(f"Dashboard location: {dashboard_path}")
    print("Opening in your default browser...")
    print("Use Ctrl+C to stop the server")
    print("-" * 50)

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            dashboard_path,
            "--server.port", "8502",  # Different port from main dashboard
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\nShutting down Retirement Spending Dashboard...")
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()