"""
Compute map to map distances on ground truth versus submission volumes.
"""

from .._map_to_map.map_to_map_distance_matrix import run

def main():
    run()
    return


if __name__ == "__main__":
    main()
