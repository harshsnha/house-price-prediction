"""Creates the four files the internship form asks for, correctly named, inside submission/.

Usage: python make_submission.py --name Harsh_Sinha --college "Your College Name"
(--name goes into the file names, so use underscores instead of spaces)
"""
import argparse
import os
import shutil
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--college", default="Your College Name")
args = ap.parse_args()

out_dir = os.path.join(BASE, "submission")
os.makedirs(out_dir, exist_ok=True)
display_name = args.name.replace("_", " ")

subprocess.run([sys.executable, os.path.join(BASE, "generate_report.py"), "--name", display_name,
                "--college", args.college,
                "--out", os.path.join(out_dir, f"{args.name}_ProjectReport.docx")], check=True)
shutil.copy(os.path.join(BASE, "notebooks", "house_price_analysis.ipynb"),
            os.path.join(out_dir, f"{args.name}_HousePricePrediction.ipynb"))
shutil.copy(os.path.join(BASE, "requirements.txt"), os.path.join(out_dir, "requirements.txt"))
shutil.copy(os.path.join(BASE, "README.md"), os.path.join(out_dir, "README.md"))
print("ready:", sorted(os.listdir(out_dir)))
