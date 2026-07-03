"""
Re-run Phase 6 transfer validation and capture full output to file.
"""
import sys
import os

# Redirect stdout to file
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, text):
        for f in self.files:
            f.write(text)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

output_file = open('experiments/results/phase6_output.txt', 'w')
sys.stdout = Tee(sys.__stdout__, output_file)
sys.stderr = Tee(sys.__stderr__, output_file)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.cross_dataset_transfer import perform_transfer_validation

perform_transfer_validation()

output_file.close()
print("\nFull output saved to experiments/results/phase6_output.txt")
