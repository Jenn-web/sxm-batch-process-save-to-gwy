# SXM Batch Process and Save to GWY

Batch process STM data in `.sxm` format from a directory and save all to original location in `.gwy` format.

## Features

- **Recursive file discovery**: Finds all `.sxm` files in a directory and its subdirectories
- **Comprehensive processing pipeline**:
  - Level data by mean plane subtraction
  - Align rows using various methods (mean, median, match height)
  - Parabolic background subtraction
  - Remove scars (anomalous lines)
- **Metadata preservation**: Saves as much information as possible from the original file
- **In-situ saving**: New `.gwy` files are saved in the same location as the source `.sxm` files
- **Compatible output**: Files can be read by the `gwyfile` module and processed further normally

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Jenn-web/sxm-batch-process-save-to-gwy.git
cd sxm-batch-process-save-to-gwy
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Python Script

```python
import sxm_batch_processor as sbp

# Process all .sxm files in a directory
created_files = sbp.batch_process_directory(
    "path/to/your/data",
    level_plane=True,
    align_rows='mean',
    parabolic_sub=True,
    remove_scars_flag=True,
    verbose=True
)

print(f"Created {len(created_files)} .gwy files")
```

### Jupyter Notebook

Open and run `batch_process_example.ipynb` for interactive examples and detailed usage.

## Usage

### Module Functions

#### `find_sxm_files(directory)`
Find all `.sxm` files in directory and subdirectories.

#### `sxm_to_gwy(sxm_file, **options)`
Convert a single `.sxm` file to `.gwy` format with processing.

**Parameters:**
- `sxm_file`: Path to input `.sxm` file
- `output_file`: Path to output `.gwy` file (optional, defaults to same name with `.gwy` extension)
- `level_plane`: Apply mean plane subtraction (default: `True`)
- `align_rows`: Row alignment method - `'mean'`, `'median'`, `'match_height'`, or `None` (default: `'mean'`)
- `parabolic_sub`: Apply parabolic subtraction (default: `True`)
- `remove_scars_flag`: Remove scars (default: `True`)
- `scar_threshold`: Threshold for scar detection in standard deviations (default: `3.0`)

#### `batch_process_directory(directory, **options)`
Process all `.sxm` files in directory tree and save as `.gwy` files.

**Parameters:**
- `directory`: Root directory to search for `.sxm` files
- Same processing options as `sxm_to_gwy()`
- `verbose`: Print progress messages (default: `True`)

**Returns:** List of paths to created `.gwy` files

### Processing Options

#### Row Alignment Methods
- `'mean'`: Subtract mean value of each row (removes horizontal stripes)
- `'median'`: Subtract median value of each row (more robust to outliers)
- `'match_height'`: Match row heights at edges (good for thermal drift)
- `None`: Skip row alignment

#### Scar Removal
Detects and removes anomalous horizontal and vertical lines using statistical analysis. The `scar_threshold` parameter controls sensitivity (lower = more sensitive).

## Examples

### Process with custom settings

```python
import sxm_batch_processor as sbp

# Use median row alignment and higher scar sensitivity
sbp.batch_process_directory(
    "data/",
    level_plane=True,
    align_rows='median',
    parabolic_sub=True,
    remove_scars_flag=True,
    scar_threshold=2.5,
    verbose=True
)
```

### Process a single file

```python
import sxm_batch_processor as sbp
from pathlib import Path

input_file = Path("data/sample.sxm")
output_file = sbp.sxm_to_gwy(input_file)
print(f"Saved to {output_file}")
```

### Verify output files

```python
import gwyfile

# Read a created .gwy file
gwy_obj = gwyfile.load("data/sample.gwy")

# Access data
for key in gwy_obj.keys():
    if key.endswith('/data'):
        datafield = gwy_obj[key]
        print(f"Shape: {datafield.data.shape}")
        print(f"Physical size: {datafield.xreal} x {datafield.yreal}")
```

## Requirements

- Python 3.8+
- nanonispy >= 1.1.0
- gwyfile >= 0.2.0
- numpy >= 1.20.0
- jupyter >= 1.0.0 (for notebook examples)

## Notes

- Original `.sxm` files are not modified
- `.gwy` files are saved in the same directory as the source files
- Metadata from `.sxm` files is preserved in the `.gwy` files
- Output files are compatible with Gwyddion and can be processed further using gwyfile

## License

MIT License
