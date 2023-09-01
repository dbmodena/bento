import os
import argparse
import subprocess
from multiprocessing import Pool


def build(library, args):
    """
    Install a docker container
    """
    print(f'Building {library}...')
    if args is not None and len(args) != 0:
        q = " ".join(["--build-arg " + x.replace(" ", "\\ ") for x in args])
    else:
        q = ""

    try:
        subprocess.check_call(
            'docker build %s --rm -t df-benchmarks-%s -f'
            ' install/Dockerfile.%s .' % (q, library, library), shell=True)
        return {library: 'success'}
    except subprocess.CalledProcessError:
        return {library: 'fail'}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--algorithm',
        metavar='NAME',
        help='build only the named algorithm image',
        default=None)
    parser.add_argument(
        '--build-arg',
        help='pass given args to all docker builds',
        nargs="+")
    args = parser.parse_args()
    
    result = subprocess.run(["docker", "image", "inspect", "df-benchmarks"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(result)
    # Check the return code
    if result.returncode != 0:
        print('Downloading ubuntu container...')
        subprocess.check_call('docker pull ubuntu:18.04', shell=True)

    print('Building base image...')
    subprocess.check_call('docker build --rm -t df-benchmarks -f install/Dockerfile .', shell=True)

    if args.algorithm:
        tags = [args.algorithm]
    else:
        tags = [fn.split('.')[-1] for fn in os.listdir('install') if fn.startswith('Dockerfile.')]

    print(tags)
    print('Building algorithm images...')

    install_status = [build(tag, args.build_arg) for tag in tags]

    print('\n\nInstall Status:\n' + '\n'.join(str(algo) for algo in install_status))


    '''
        for tag in tags:
        result = subprocess.run(["docker", "image", "inspect", "df-benchmarks-{tag}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result != 0:
            install_status = [build(tag, args.build_arg) for tag in tags]
            print('\n\nInstall Status:\n' + '\n'.join(str(algo) for algo in install_status))
    '''