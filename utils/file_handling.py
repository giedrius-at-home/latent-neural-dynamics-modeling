from pathlib import Path
import json
from scipy.io import loadmat


def read_json(file_path: str) -> dict:
    with open(file_path, "r") as f:
        contents = json.load(f)
    return contents


def load_mat_into_dict(mat_file: str) -> dict:
    data_dict = loadmat(mat_file)
    return data_dict


def list_files(folder_path: Path) -> list[str]:
    return [str(item) for item in folder_path.iterdir()]


def get_child_subchilds_tuples(
    parent_folder: Path, full_paths: bool = False
) -> list[tuple[str, ...]]:
    """
    Return tuples capturing only the deepest hierarchical directory chains starting at
    parent_folder. That is, for every terminal (leaf) directory reachable from
    parent_folder, return the full chain from parent to that leaf.

    Example (names, not full paths):
    parent/
      └── a/
          └── b/
              └── c/
    -> [
        ("parent", "a", "b", "c"),
       ]

    With siblings:
    parent/
      └── a/
          ├── x/
          │   └── y/
          └── m/
              └── n/
    -> [
        ("parent", "a", "x", "y"),
        ("parent", "a", "m", "n"),
       ]

    If full_paths is True, each element is the full string path.
    """

    def recurse(folder: Path, lineage: list[str]) -> list[tuple[str, ...]]:
        leaf_chains: list[tuple[str, ...]] = []
        subdirs = [child for child in folder.iterdir() if child.is_dir()]
        if not subdirs:
            # Current folder is a leaf; return the lineage as a completed chain
            # Only include chains that have at least two elements (parent and one child)
            if len(lineage) >= 2:
                leaf_chains.append(tuple(lineage))
            return leaf_chains
        for child in subdirs:
            child_part = str(child) if full_paths else child.name
            child_lineage = lineage + [child_part]
            leaf_chains.extend(recurse(child, child_lineage))
        return leaf_chains

    root_part = str(parent_folder) if full_paths else parent_folder.name
    return recurse(parent_folder, [root_part])
