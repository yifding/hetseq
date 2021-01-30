import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--distributed_no_spawn', action='store_true',
                       help='do not spawn multiple processes even if multiple GPUs are visible')
    args = parser.parse_args()
    print(args.distributed_no_spawn)