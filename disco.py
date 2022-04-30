import os, sys

# Set base project directory to current working directory
PROJECT_DIR = os.path.abspath(os.getcwd())

# Import DD helper modules
sys.path.append(PROJECT_DIR)
import dd, dd_args

# Unsure about these:
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

# Load parameters
pargs = dd_args.arg_configuration_loader()

# Setup folders
folders = dd.setupFolders(is_colab=dd.detectColab(), PROJECT_DIR=PROJECT_DIR, pargs=pargs)

# Load Models
dd.loadModels(folders)

# Report System Details
dd.systemDetails(pargs)

# Get CUDA Device
device = dd.getDevice(pargs)

dd.start_run(pargs=pargs, folders=folders, device=device, is_colab=dd.detectColab())
