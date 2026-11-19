from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _java_home() -> Path:
    if os.environ.get("JAVA_HOME"):
        return Path(os.environ["JAVA_HOME"])
    found = sorted(TOOLS.glob("jdk-*"))
    if not found:
        raise RuntimeError("No JDK: set JAVA_HOME or unpack a JDK 17 or 21 into tools/ (see README).")
    return found[-1]


def _windows_shim(java_home: Path) -> Path:
    """Compile the small file-system class once, and return its folder for the classpath."""
    import pyspark

    src = Path(__file__).resolve().parent / "spark_win" / "NoNativeLocalFS.java"
    out = TOOLS / "spark_win"
    if not (out / "bft" / "NoNativeLocalFS.class").exists():
        out.mkdir(parents=True, exist_ok=True)
        cp = str(Path(pyspark.__file__).parent / "jars" / "*")
        subprocess.run([str(java_home / "bin" / "javac.exe"), "-nowarn", "-cp", cp, "-d", str(out), str(src)], check=True)
    return out


def get_spark(threads: int = 0, driver_memory: str = "8g", shuffle_partitions: int | None = None,
              app: str = "bft65b", extra: dict | None = None):
    """threads=0 means all logical CPUs. The session uses UTC, so dates never shift by time zone."""
    java_home = _java_home()
    os.environ["JAVA_HOME"] = str(java_home)
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    from pyspark.sql import SparkSession

    b = (SparkSession.builder.master(f"local[{threads or '*'}]").appName(app)
         .config("spark.driver.memory", driver_memory)
         .config("spark.sql.session.timeZone", "UTC")
         .config("spark.ui.showConsoleProgress", "false")
         .config("spark.sql.execution.arrow.pyspark.enabled", "true"))
    if shuffle_partitions:
        b = b.config("spark.sql.shuffle.partitions", str(shuffle_partitions))
    if os.name == "nt":
        shim = _windows_shim(java_home)
        b = (b.config("spark.driver.extraClassPath", str(shim))
             .config("spark.hadoop.fs.file.impl", "bft.NoNativeLocalFS")
             .config("spark.hadoop.fs.file.impl.disable.cache", "true"))
    for k, v in (extra or {}).items():
        b = b.config(k, v)
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
