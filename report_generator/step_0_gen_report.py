# ------------------------------------------------------------------------------
#
#
#                                 P̶̖̥͈͈͇̼͇̈́̈́̀͋͒̽͊͘͠h̴͙͈̦͗́̓a̴̧͗̾̀̅̇ḡ̶͓̭̝͓͎̰͓̦̭̎́͊̐̂͒͠ơ̵̘͌̿̽̑̈̾Ś̴̠́̓̋̂̃͒͆̚t̴̲͓̬͎̾͑͆͊́̕a̸͍̫͎̗̞͆̇̀̌̑͂t̸̳̼̘͕̯̠̱̠͂̔̒̐̒̕͝͝
#
#
#                                PhagoStat
#                Advanced Phagocytic Activity Analysis Tool
# ------------------------------------------------------------------------------
# Copyright (C) 2023 Mehdi OUNISSI <mehdi.ounissi@icm-institute.org>
#               Sorbonne University, Paris Brain Institute - ICM, CNRS, Inria,
#               Inserm, AP-HP, Paris, 75013, France.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ------------------------------------------------------------------------------
# Note on Imported Packages:
# The packages used in this work are imported as is and not modified. If you
# intend to use, modify, or distribute any of these packages, please refer to
# the requirements.txt file and the respective package licenses.
# ------------------------------------------------------------------------------

from utils import phagocytosis_evolution_to_csv_pdf
from multiprocessing import Pool
from natsort import natsorted
from glob import glob
from os import getpid
import time
import os

def processing_function(cpu_id):
    print("Process started : ", getpid())
    for i in range(limit_list[cpu_id], limit_list[cpu_id+1]):
        try:
            cell_tk_csv_path = glob(os.path.join(os.path.dirname(phago_csv_paths[i]),  'cell_tracking_report.csv'))
            if len(cell_tk_csv_path) == 1:
                phagocytosis_evolution_to_csv_pdf(phago_csv_paths[i],
                                                  cell_tk_csv_path[0],
                                                  output_path=report_repository_dir,
                                                  keep_until=200,
                                                  time_per_frame=2)
        except IndexError: pass

if __name__ == '__main__':
    # Start the timer
    start_time = time.perf_counter()

    # Keep data until time point
    keep_until=200

    # Time in min per frame (data point)
    time_per_frame = 2

    # Path to the analyzed cells (microglia) video microscopy
    analysed_data_abs_path  = os.path.join('root', 'microglia_video_microscopy')

    # Path to store reports
    report_repository_dir = os.path.join('root', 'report_repository')

    # Loading csv files for all scenes
    phago_csv_paths = natsorted(glob(os.path.join(analysed_data_abs_path, '*','*','scene_*', 'CSVs','phagocytosis_evolution.csv')))

    # CPU count for multi-task
    cpu_count = multiprocessing.cpu_count()

    # Defining images per CPU
    limit_list = []

    for cpu_id in range(cpu_count):
        limit_list.append(int(len(phago_csv_paths)/cpu_count)*cpu_id)
    limit_list.append(len(phago_csv_paths))

    with Pool() as pool:
        pool.map(processing_function, [cpu_id for cpu_id in range(cpu_count)])
    
    # Stop the timer and calculate elapsed time
    elapsed_time = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")