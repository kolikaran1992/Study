from pathlib import Path

root_path_obj = Path(__file__).absolute().parent

path_to_logs = root_path_obj.joinpath('logs')