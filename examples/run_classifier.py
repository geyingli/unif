import uf
import numpy as np


def get_best_f1(probs, labels, label_index=1):
    assert len(probs) == len(labels)
    probs = np.array(probs)
    labels = np.array(labels)
    num = np.sum(labels == label_index)

    tp = num
    fp = len(labels) - num
    fn = 0
    tn = 0

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    threshold = 0

    ids = sorted(
        list(range(len(probs))), key=lambda i: probs[i])
    for i in ids:
        prob = probs[i]
        label = labels[i]
        if label == label_index:
            tp -= 1
            fn += 1
        elif label != label_index:
            fp -= 1
            tn += 1

        _accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-9)
        _precision = tp / (tp + fp + 1e-9)
        _recall = tp / (tp + fn + 1e-9)
        _f1 = 2 * _precision * _recall / (_precision + _recall + 1e-9)
        if _f1 > f1:
            accuracy = _accuracy
            precision = _precision
            recall = _recall
            f1 = _f1
            threshold = prob

    return (accuracy, precision, recall, f1, threshold, num)


def main():

    uf.set_log("./log")

    # load data
    with open("./data/SST-2/train.tsv", encoding="utf-8") as f:
        X, y = [], []
        for line in f.readlines()[1:]:    # ignore title
            line = line.strip().split("\t")
            X.append(line[0])
            y.append(int(line[1]))
    with open("./data/SST-2/dev.tsv", encoding="utf-8") as f:
        X_dev, y_dev = [], []
        for line in f.readlines()[1:]:    # ignore title
            line = line.strip().split("\t")
            X_dev.append(line[0])
            y_dev.append(int(line[1]))

    # modeling
    model = uf.BERTClassifier(
        config_file="./bert-base-zh/bert_config.json",
        vocab_file="./bert-base-zh/vocab.txt",
        max_seq_length=128,
        init_checkpoint="./bert-base-zh",
        output_dir="outputs",
        gpu_ids="0")

    # training
    for epoch in range(3):
        model.fit(
            X, y,
            batch_size=64,
            target_steps=-(epoch + 1),
            total_steps=-3,
            print_per_secs=5,
            save_per_steps=1000000)
        model.cache("epoch_%d" % epoch)

        probs = model.predict(X_dev)["probs"]
        for i in range(2):
            acc, pre, rec, f1, thresh, num = get_best_f1(
                probs[:, i], y_dev, label_index=i)
            print("[dev] label %d (%d): accuracy %.6f, precision %.6f, "
                  "recall %.6f, f1 %.6f, thresh %s"
                  % (i, num, acc, pre, rec, f1, thresh))

    print("Application finished.")


if __name__ == "__main__":

    main()
