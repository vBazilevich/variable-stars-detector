import os
import tarfile
from ftplib import FTP
import requests
import tqdm
from multiprocessing import Pool, Process
import functools

processes = []
OGLE_III_FTP = "ftp.astrouw.edu.pl"
OUT_DIR = "../data/OGLE3"
PHOT_FILE = 'phot.tar.gz'

SKIP_LIST = ['bright_lmc', 'remarks.txt', 'unclassified']

def load_file(file, url_pfx, vt):
    url = url_pfx + file
    fname = os.path.basename(file)
    filepath = os.path.join(OUT_DIR, vt, fname)
    with open(filepath, 'wb') as f:
        r = requests.get(url)
        if r.ok:
            f.write(r.content)
        else:
            print(f'Failed to download file from {url}')

def load_tarball(path, url_pfx, directory, vt, forked=False):
    filepath = os.path.join(path, f'{directory}-{vt}-{PHOT_FILE}')
    with open(filepath, 'wb') as f:
        r = requests.get(url_pfx + f'{vt}/{PHOT_FILE}', stream=True)
        size = int(r.headers.get('content-length', 0))
        if not forked:
            with tqdm.tqdm(desc=f'{directory}-{vt}', total=size, unit='B', unit_scale=True) as pbar:
                for data in r.iter_content(65536):
                    pbar.update(len(data))
                    f.write(data)
        else:
            for data in r.iter_content(65536):
                f.write(data)


    with tarfile.open(filepath) as f:
        # dn is different - only I band photometry without a band-specific subfolder
        phot_dir = 'phot/I/' if vt != 'dn' else 'phot/'
        i_phot = [tarinfo for tarinfo in f.getmembers() if tarinfo.name.startswith(phot_dir)]
        for dat in i_phot:
            dat.name = os.path.basename(dat.name)
            f.extract(dat, path)

    os.remove(filepath)


def crawl_dir(ftp, directory):
    ftp.cwd(directory)
    variability_types = ftp.nlst()
    url_pfx = f'https://{OGLE_III_FTP}/ogle/ogle3/OIII-CVS/{directory}/'

    for vt in variability_types:
        if vt in SKIP_LIST:
            continue

        path = os.path.join(OUT_DIR, vt)
        if not os.path.exists(path):
            os.makedirs(path)

        # Check if tarball is available
        if any([f.endswith(PHOT_FILE) for f in ftp.nlst(vt)]):
            size = ftp.size(os.path.join(vt, PHOT_FILE))

            if size < 2**25:
                load_tarball(path, url_pfx, directory, vt)
            else:
                # Fork a process for large files
                print(f'{directory}-{vt} will be downloaded in a separate process')
                p = Process(target=load_tarball, args=(path, url_pfx, directory, vt, True))
                p.start()
                processes.append(p)
        else:
            # Fallback to individual photometry files download

            photometry_files = ftp.nlst(os.path.join(vt, 'phot', 'I')) 
            n_files = len(photometry_files)
            worker = functools.partial(load_file, url_pfx=url_pfx, vt=vt)
            with Pool() as p:
                with tqdm.tqdm(total=n_files) as pbar:
                    for _ in p.imap_unordered(worker, photometry_files):
                        pbar.update()
    ftp.cwd("..")

if __name__ == "__main__":
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    ftp = FTP(OGLE_III_FTP, timeout=3600)
    ftp.login()
    ftp.sendcmd('PASV')
    ftp.cwd("ogle/ogle3/OIII-CVS/")

    # Collect list of photometry files
    root_dirs = ftp.nlst()
    for rd in root_dirs:
        if rd in SKIP_LIST:
            continue
        crawl_dir(ftp, rd)

    ftp.quit()

    for p in processes:
        p.join()
