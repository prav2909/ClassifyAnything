import argparse, os, sys
import os
from classification.detr import DETR
from datetime import datetime

def decode_operations(operations, save_path):
    """
    Decode the list of operations and create instances of the DETR class for each operation.

    Parameters:
    operations : list of str
        List of operations to run (e.g., 'object_detection', 'segmentation').
    save_path : str
        Path to save the results.

    Returns:
    list of DETR
        List of DETR instances configured for the specified operations.
    """
    operations_list = []
    if 'object_detection' in operations:
        operations_list.append(
            DETR(image_path=os.path.join(args.image_path, args.image_name), save_path=save_path, save=args.save_results)
            )
    if 'segmentation' in operations:
        operations_list.append(
            DETR(image_path=os.path.join(args.image_path, args.image_name), save_path=save_path, save=args.save_results, segmentation=True)
            )
    return operations_list

def runner(operations_list):
    """
    Run the specified operations.

    Parameters:
    operations_list : list of Models
        List of Model instances to run.
    """
    for a_operation in operations_list:
        a_operation.run()
        a_operation.show_results()
    pass
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script reads all the operations you want to run and runs it and saves the results.")
    parser.add_argument("-o", "--operations", type=str, nargs='+', help="Operations to run, comma seperated vals")
    parser.add_argument("-ipath", "--image_path", type=str, help="Image path")
    parser.add_argument("-iname", "--image_name", type=str, help="Image Name with file type extension")
    parser.add_argument("-save", "--save_results", action='store_true', help="Save Results?")

    # Parse the arguments
    args = parser.parse_args()

    save_path = os.path.join(os.getcwd(),'results', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(save_path, exist_ok=True)

    # Decode the operations and create DETR instances
    operations_list = decode_operations(args.operations, save_path)

    # Run the models - Inference time!
    runner(operations_list)

    
