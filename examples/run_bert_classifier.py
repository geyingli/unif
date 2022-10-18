import uf
import numpy as np


def get_best_f1(probs, labels, label_index=1):
    """ Calculate the best f1 by scanning over probabilities. """

    assert len(probs) == len(labels)
    probs = np.array(probs)
    labels = np.array(labels)

    # initialize metrics
    n = np.sum(labels == label_index)
    tp = n
    fp = len(labels) - n
    fn = 0
    tn = 0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1)
    threshold = 0

    ids = sorted(range(len(probs)), key=lambda i: probs[i])
    for i in ids:
        prob = probs[i]
        label = labels[i]
        if label == label_index:
            tp -= 1
            fn += 1
        elif label != label_index:
            fp -= 1
            tn += 1

        _accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        _precision = tp / max(tp + fp, 1)
        _recall = tp / max(tp + fn, 1)
        _f1 = 2 * _precision * _recall / max(_precision + _recall, 1)
        if _f1 > f1:
            accuracy = _accuracy
            precision = _precision
            recall = _recall
            f1 = _f1
            threshold = prob
    return (n, accuracy, precision, recall, f1, threshold)


def main():

    uf.set_log("./log")

    # load data
    X, y = [], []
    X_dev, y_dev = [], []
    with open("sst-2/train.txt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:          # ignore title
                continue
            query, label = line.strip("\n").split("\t")
            X.append(query)
            y.append(int(label))
    with open("sst-2/dev.txt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:          # ignore title
                continue
            query, label = line.strip("\n").split("\t")
            X_dev.append(query)
            y_dev.append(int(label))

    # modeling
    checkpoint_dir = "pretrained/bert-base-zh"
    model = uf.BERTClassifier(
        config_file=f"{checkpoint_dir}/bert_config.json",
        vocab_file=f"{checkpoint_dir}/vocab.txt",
        max_seq_length=32,
        label_size=2,
        init_checkpoint=checkpoint_dir,
        output_dir="bert",
        gpu_ids="0")

    # training
    for epoch in range(3):
        model.fit(
            X, y,
            batch_size=64,
            target_steps=-(epoch + 1),
            total_steps=-3,
            print_per_secs=5,
            save_per_steps=3000)
        model.localize("bp.%d" % epoch)

        # validation
        probs = model.predict(X_dev)["probs"]
        for i in range(2):
            n, acc, pre, rec, f1, thresh = get_best_f1(probs=probs[:, i], labels=y_dev, label_index=i)
            print("[dev] label %d (%d): accuracy %.3f, precision %.3f, recall %.3f, best_f1 %.3f, thresh >%s"
                  % (i, n, acc, pre, rec, f1, thresh))

    print("Application finished.")


if __name__ == "__main__":
    main()
