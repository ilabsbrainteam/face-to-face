import os
import mne
from f2f_helpers import load_paths, get_skip_regexp

# config paths
data_root, subjects_dir, results_dir = load_paths()
param_dir = os.path.join('..', 'params')
plot_dir = os.path.join(results_dir, 'figs', 'parcellations')
for _dir in (plot_dir,):
    os.makedirs(_dir, exist_ok=True)

# load other config values
Brain = mne.viz.get_brain_class()
parcellation = 'aparc_sub'

brain = Brain(
    'fsaverage', hemi='split', surf='pial', size=(1200, 900),
    cortex='low_contrast', views=['lateral', 'medial'], background='white',
    subjects_dir=subjects_dir)
regexp = get_skip_regexp()
labels = mne.read_labels_from_annot(
    'fsaverage', parcellation, regexp=regexp, subjects_dir=subjects_dir)
label_dict = {label.name: label for label in labels}

colors = ('r', 'orange', 'y', 'g', 'b', 'purple', 'pink', 'brown') * 2
borders = (False,) * 8 + (True,) * 8

raise RuntimeError()
# convenience variables. Run blocks below interactively as needed after
# changing these variables.
bilateral = 'supramarginal'
left_hemi = 'parsopercularis'

# bilateral
for n, (col, bord) in enumerate(zip(colors, borders), start=1):
    for hemi in ('rh', 'lh'):
        try:
            brain.add_label(label_dict[f'{bilateral}_{n}-{hemi}'],
                            color=col, borders=bord, alpha=0.5)
        except KeyError:
            continue

# left hemi only
for n, (col, bord) in enumerate(zip(colors, borders), start=1):
    brain.add_label(label_dict[f'{left_hemi}_{n}-lh'],
                    color=col, borders=bord, alpha=0.5)
