import os
import time
import copy
try:
    import requests
except Exception:
    pass
from sys import stdout

from ..tools import tf


RESOURCES = [
    ["bert-base-zh", "BERT", "Google",
     "https://github.com/google-research/bert",
     "https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip"],

    ["albert-tiny-zh", "ALBERT", "Google",
     "https://github.com/google-research/albert",
     "https://storage.googleapis.com/albert_zh/albert_tiny_zh_google.zip"],
    ["albert-small-zh", "ALBERT", "Google",
     "https://github.com/google-research/albert",
     "https://storage.googleapis.com/albert_zh/albert_small_zh_google.zip"],

    ["albert-base-zh", "ALBERT", "Brightmart",
     "https://github.com/brightmart/albert_zh",
     "https://storage.googleapis.com/albert_zh/albert_base_zh_additional_36k_steps.zip"],
    ["albert-large-zh", "ALBERT", "Brightmart",
     "https://github.com/brightmart/albert_zh",
     "https://storage.googleapis.com/albert_zh/albert_large_zh.zip"],
    ["albert-xlarge-zh", "ALBERT", "Brightmart",
     "https://github.com/brightmart/albert_zh",
     "https://storage.googleapis.com/albert_zh/albert_xlarge_zh_183k.zip"],

    ["bert-wwm-ext-base-zh", "BERT", "HFL",
     "https://github.com/ymcui/Chinese-BERT-wwm",
     "https://drive.google.com/uc?export=download&id=1buMLEjdtrXE2c4G1rpsNGWEx7lUQ0RHi"],
    ["roberta-wwm-ext-base-zh", "BERT", "HFL",
     "https://github.com/ymcui/Chinese-BERT-wwm",
     "https://drive.google.com/uc?export=download&id=1jMAKIJmPn7kADgD3yQZhpsqM-IRM1qZt"],
    ["roberta-wwm-ext-large-zh", "BERT", "HFL",
     "https://github.com/ymcui/Chinese-BERT-wwm",
     "https://drive.google.com/uc?export=download&id=1dtad0FFzG11CBsawu8hvwwzU2R0FDI94"],

    ["macbert-base-zh", "BERT", "HFL",
     "https://github.com/ymcui/MacBERT",
     "https://drive.google.com/uc?export=download&id=1aV69OhYzIwj_hn-kO1RiBa-m8QAusQ5b"],
    ["macbert-large-zh", "BERT", "HFL",
     "https://github.com/ymcui/MacBERT",
     "https://drive.google.com/uc?export=download&id=1lWYxnk1EqTA2Q20_IShxBrCPc5VSDCkT"],

    ["xlnet-base-zh", "XLNet", "HFL",
     "https://github.com/ymcui/Chinese-XLNet",
     "https://drive.google.com/uc?export=download&id=1m9t-a4gKimbkP5rqGXXsEAEPhJSZ8tvx"],
    ["xlnet-mid-zh", "XLNet", "HFL",
     "https://github.com/ymcui/Chinese-XLNet",
     "https://drive.google.com/uc?export=download&id=1342uBc7ZmQwV6Hm6eUIN_OnBSz1LcvfA"],

    ["electra-180g-small-zh", "ELECTRA", "HFL",
     "https://github.com/ymcui/Chinese-ELECTRA",
     "https://drive.google.com/uc?export=download&id=177EVNTQpH2BRW-35-0LNLjV86MuDnEmu"],
    ["electra-180g-small-ex-zh", "ELECTRA", "HFL",
     "https://github.com/ymcui/Chinese-ELECTRA",
     "https://drive.google.com/uc?export=download&id=1NYJTKH1dWzrIBi86VSUK-Ml9Dsso_kuf"],
    ["electra-180g-base-zh", "ELECTRA", "HFL",
     "https://github.com/ymcui/Chinese-ELECTRA",
     "https://drive.google.com/uc?export=download&id=1RlmfBgyEwKVBFagafYvJgyCGuj7cTHfh"],
    ["electra-180g-large-zh", "ELECTRA", "HFL",
     "https://github.com/ymcui/Chinese-ELECTRA",
     "https://drive.google.com/uc?export=download&id=1P9yAuW0-HR7WvZ2r2weTnx3slo6f5u9q"],
]


def list_resources():
    columns = ["Key", "Backbone", "Organization", "Site", "URL"]
    lengths = [len(col) for col in columns]
    resources = copy.deepcopy(RESOURCES)

    # scan for maximum length
    for i in range(len(resources)):
        for j in range(len(columns)):
            lengths[j] = max(lengths[j], len(resources[i][j]))
    seps = ["─" * length for length in lengths]

    # re-scan to modify length
    tf.logging.info("┌─" + "─┬─".join(seps) + "─┐")
    for j in range(len(columns)):
        columns[j] += " " * (lengths[j] - len(columns[j]))
    tf.logging.info("┊ " + " ┊ ".join(columns) + " ┊")
    tf.logging.info("├─" + "─┼─".join(seps) + "─┤")

    for i in range(len(resources)):
        for j in range(len(columns)):
            resources[i][j] += " " * (lengths[j] - len(resources[i][j]))
        tf.logging.info("┊ " + " ┊ ".join(resources[i]) + " ┊")
    tf.logging.info("└─" + "─┴─".join(seps) + "─┘")


def download(key):
    resources = {item[0]: item for item in RESOURCES}
    if key not in resources:
        raise ValueError("Invalid key: %s. Check available resources "
                         "through `uf.list_resources()`." % key)
    url = resources[key][-1]

    # download files
    try:
        if "drive.google.com" in url:
            path = get_download_path(key, ".tar.gz")
            download_from_google_drive(url, path)
        elif "storage.googleapis.com" in url:
            path = get_download_path(key, ".zip")
            download_from_google_apis(url, path)
    except KeyboardInterrupt:
        os.remove(path)


def download_all():
    resources = {item[0]: item for item in RESOURCES}
    for key in resources:
        url = resources[key][-1]

        try:
            if "drive.google.com" in url:
                path = get_download_path(key, ".tar.gz")
                download_from_google_drive(url, path)
            elif "storage.googleapis.com" in url:
                path = get_download_path(key, ".zip")
                download_from_google_apis(url, path)
        except KeyboardInterrupt:
            os.remove(path)
            return


def download_from_google_drive(url, path):
    with open(path, "wb") as writer:
        file_id = url.split("id=")[-1]
        ori_url = url
        url = "https://docs.google.com/uc?export=download"

        session = requests.Session()
        r = session.get(url, params={"id": file_id}, stream=True)

        def _get_confirm_token(response):
            for key, value in response.cookies.items():
                if key.startswith("download_warning"):
                    return value
            return None
        token = _get_confirm_token(r)

        if token:
            params = {"id": file_id, "confirm": token}
            r = session.get(url, params=params, stream=True)
        tf.logging.info("Downloading files from %s" % (ori_url))

        chunk_size = 10240
        acc_size = 0
        cur_size = 0
        speed = None
        last_tic = time.time()
        for chunk in r.iter_content(chunk_size):
            if not chunk:
                continue

            writer.write(chunk)
            cur_size += chunk_size
            acc_size += chunk_size
            tic = time.time()

            if tic - last_tic > 3 or speed is None:
                span = tic - last_tic
                speed = "%.2fMB/s" % (cur_size / span / (1024 ** 2))
                last_tic = tic
                cur_size = 0
            stdout.write(
                "Downloading ... %.2fMB [%s] \r"
                % (acc_size / (1024 ** 2), speed))

        # extract files
        # extract_dir = path.replace(".tar.gz", "")
        # with tarfile.open(path) as tar_file:
        #     tar_file.extractall(extract_dir)
        # os.remove(path)
        tf.logging.info(
            "Succesfully downloaded. Saved into ./%s" % path)


def download_from_google_apis(url, path):
    with requests.get(url, stream=True) as r, open(path, "wb") as writer:
        file_size = int(r.headers["Content-Length"])
        tf.logging.info("Downloading files from %s (%dMB)"
                        % (url, file_size // (1024 ** 2)))

        chunk_size = 10240
        percentage = 0
        cur_size = 0
        percentage_step = chunk_size / file_size
        speed = None
        last_tic = time.time()
        for chunk in r.iter_content(chunk_size):
            if not chunk:
                continue

            writer.write(chunk)
            cur_size += chunk_size
            percentage += percentage_step
            percentage = min(percentage, 1.0)
            tic = time.time()

            if tic - last_tic > 3 or speed is None:
                span = tic - last_tic
                speed = "%.2fMB/s" % (cur_size / span / (1024 ** 2))
                last_tic = tic
                cur_size = 0
            stdout.write(
                "Downloading ... %.2f%% [%s] \r" % (percentage * 100, speed))

        # extract files
        # extract_dir = path.replace(".zip", "")
        # tf.gfile.MakeDirs(extract_dir)
        # with zipfile.ZipFile(path) as zip_file:
        #     zip_file.extractall(extract_dir)
        # os.remove(path)
        tf.logging.info(
            "Succesfully downloaded. Saved into ./%s" % path)


def get_download_path(key, suffix=".zip"):
    new_path = key + suffix
    if not os.path.exists(new_path):
        return new_path
    index = 1
    while True:
        new_path = key + " (%d)" % index + suffix
        if not os.path.exists(new_path):
            return new_path
        index += 1
