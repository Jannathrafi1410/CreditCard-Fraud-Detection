"""
Hadoop Distributed File System (HDFS) Utilities and Command Interface.
Provides cross-platform execution for HDFS commands, with automatic fallback
to an HDFS-compliant local storage simulator when running on a standalone laptop.
"""

import os
import shutil
import subprocess
from typing import List, Dict, Any, Optional


class HDFSManager:
    """
    Manages HDFS operations (mkdir, put, ls, rm, cat).
    Automatically detects if native Hadoop HDFS CLI is accessible.
    If not, operates in simulated local HDFS sandbox mode ('data/hdfs_storage/').
    """
    def __init__(self, local_sandbox_dir: str = "data/hdfs_storage"):
        self.local_sandbox = local_sandbox_dir
        self.has_native_hdfs = self._check_native_hdfs()
        if not self.has_native_hdfs:
            os.makedirs(self.local_sandbox, exist_ok=True)
            print(f"[HDFSManager] Native Hadoop not in PATH. Operating in HDFS Sandbox mode: '{self.local_sandbox}'")
        else:
            print("[HDFSManager] Native Hadoop environment detected.")

    def _check_native_hdfs(self) -> bool:
        """Checks if 'hdfs' or 'hadoop' CLI is available in PATH."""
        try:
            res = subprocess.run(["hdfs", "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return res.returncode == 0
        except Exception:
            return False

    def mkdir(self, hdfs_path: str) -> bool:
        """Creates directory in HDFS (hdfs dfs -mkdir -p <hdfs_path>)."""
        clean_path = hdfs_path.lstrip("/\\")
        print(f"[HDFS] Executing: hdfs dfs -mkdir -p /{clean_path}")

        if self.has_native_hdfs:
            cmd = ["hdfs", "dfs", "-mkdir", "-p", f"/{clean_path}"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            return res.returncode == 0
        else:
            target = os.path.join(self.local_sandbox, clean_path)
            os.makedirs(target, exist_ok=True)
            return True

    def put(self, local_src: str, hdfs_dest: str) -> bool:
        """Copies file to HDFS (hdfs dfs -put <local_src> <hdfs_dest>)."""
        clean_dest = hdfs_dest.lstrip("/\\")
        print(f"[HDFS] Executing: hdfs dfs -put {local_src} /{clean_dest}")

        if not os.path.exists(local_src):
            raise FileNotFoundError(f"Source file not found: {local_src}")

        if self.has_native_hdfs:
            cmd = ["hdfs", "dfs", "-put", "-f", local_src, f"/{clean_dest}"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            return res.returncode == 0
        else:
            dest_dir = os.path.join(self.local_sandbox, clean_dest)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir, exist_ok=True)
            filename = os.path.basename(local_src)
            dest_file = os.path.join(dest_dir, filename)
            shutil.copy2(local_src, dest_file)
            print(f"[HDFS Sandbox] Successfully copied {local_src} -> {dest_file}")
            return True

    def ls(self, hdfs_path: str = "/") -> List[str]:
        """Lists directory contents in HDFS (hdfs dfs -ls <hdfs_path>)."""
        clean_path = hdfs_path.lstrip("/\\")
        print(f"[HDFS] Executing: hdfs dfs -ls /{clean_path}")

        if self.has_native_hdfs:
            cmd = ["hdfs", "dfs", "-ls", f"/{clean_path}"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            return res.stdout.splitlines()
        else:
            target = os.path.join(self.local_sandbox, clean_path)
            if not os.path.exists(target):
                return []
            items = os.listdir(target)
            results = []
            for item in items:
                full = os.path.join(target, item)
                size = os.path.getsize(full) if os.path.isfile(full) else 0
                item_type = "-rw-r--r--" if os.path.isfile(full) else "drwxr-xr-x"
                results.append(f"{item_type}  1 hdfs hadoop  {size:>10}  {item}")
            return results

    def get_equivalent_cluster_commands(self) -> str:
        """Returns standard bash commands for running on an actual Apache Hadoop cluster."""
        return """
# ==============================================================================
# Native Apache Hadoop HDFS Cluster Commands Reference for Viva/Demo
# ==============================================================================
# 1. Format HDFS namenode (initial setup only):
hdfs namenode -format

# 2. Start HDFS daemons (NameNode and DataNode):
start-dfs.sh

# 3. Create fraud detection input and output directories:
hdfs dfs -mkdir -p /fraud_detection/input
hdfs dfs -mkdir -p /fraud_detection/output

# 4. Upload raw dataset to distributed storage:
hdfs dfs -put -f data/raw/creditcard.csv /fraud_detection/input/

# 5. Verify file placement and replica status:
hdfs dfs -ls -h /fraud_detection/input/

# 6. Execute distributed PySpark pipeline on the cluster:
spark-submit \\
  --master yarn \\
  --deploy-mode client \\
  --driver-memory 4G \\
  --executor-memory 4G \\
  --num-executors 4 \\
  src/spark_pipeline.py \\
  --input hdfs:///fraud_detection/input/creditcard.csv \\
  --output hdfs:///fraud_detection/output/
"""


if __name__ == "__main__":
    hdfs = HDFSManager()
    hdfs.mkdir("/fraud_detection/input")
    hdfs.mkdir("/fraud_detection/output")
    test_file = "data/sample/sample_transactions.csv"
    if os.path.exists(test_file):
        hdfs.put(test_file, "/fraud_detection/input")
        print("\n--- HDFS Directory Listing ---")
        for line in hdfs.ls("/fraud_detection/input"):
            print(line)
    print(hdfs.get_equivalent_cluster_commands())
